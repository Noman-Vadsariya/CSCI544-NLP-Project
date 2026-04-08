"""
Simplified hypernetwork for generating LoRA weights from context.

Pipeline: context text -> context encoder -> perceiver aggregator -> MLP head -> LoRA A/B -> apply to frozen LLM
"""

import logging
from dataclasses import dataclass
from functools import partial
from math import sqrt
from typing import Any

import torch
from einops import rearrange, repeat, unpack
from einops.layers.torch import EinMix as Mix
from jaxtyping import Float, Integer
from peft import LoraConfig, LoraRuntimeConfig, PeftConfig, PeftModel
from peft.tuners._buffer_dict import BufferDict
from peft.tuners.tuners_utils import BaseTunerLayer, check_target_module_exists
from peft.utils import PeftType, TaskType
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from simple_hyperlora.idefics2 import Idefics2Perceiver, Idefics2PerceiverConfig
from simple_hyperlora.lora_merger import combine_lora
from simple_hyperlora.lora_ops import apply_lora_to_layers, get_layers, lora_forward_packed

logger = logging.getLogger()

# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

@dataclass
class HypernetConfig:
    latent_size: int
    lora_config: LoraConfig
    base_hidden_size: int
    layer_indices: torch.Tensor
    feature_sizes: tuple[dict[str, int], dict[str, int]]
    # aggregator params
    num_blocks: int
    num_self_attn_per_block: int
    shared_weights: bool
    n_latent_queries: int
    ctx_feature_size: int
    per_rank_gen: bool
    num_pre_head_layers: int
    dropout_rate: float
    use_bias: bool
    layer_to_layer: bool  # whether ctx encoder produces per-layer activations


# ---------------------------------------------------------------------------
# Context Encoders
# ---------------------------------------------------------------------------

class EarlyExitEncoder(nn.Module):
    """Frozen LLM truncated at a given layer."""

    def __init__(self, base_model: PreTrainedModel, layer_idx: int):
        super().__init__()
        # strip to inner model (remove lm_head etc.)
        model = base_model
        while hasattr(model, "model"):
            model = model.model
        model.layers = model.layers[:layer_idx]
        self.base_model = model

    @property
    def config(self):
        return self.base_model.config

    @torch.no_grad()
    def forward(self, **kwargs):
        return self.base_model(**kwargs).last_hidden_state


class PerLayerActivationsEncoder(nn.Module):
    """Frozen LLM returning all hidden states stacked as [bs, n_layers, seq_len, hidden]."""

    def __init__(self, base_model: PreTrainedModel, last_layer: int | None = None):
        super().__init__()
        model = base_model
        while hasattr(model, "model"):
            model = model.model
        if last_layer is not None:
            model.layers = model.layers[: last_layer - 1]
        else:
            model.layers = model.layers[:-1]
        self.base_model = model

    @property
    def config(self):
        return self.base_model.config

    @torch.no_grad()
    def forward(self, **kwargs):
        kwargs["output_hidden_states"] = True
        outputs = self.base_model(**kwargs)
        return torch.stack(outputs.hidden_states, dim=1)


# ---------------------------------------------------------------------------
# Perceiver Aggregator
# ---------------------------------------------------------------------------

class PerceiverAggregator(nn.Module):
    """Perceiver that compresses context features into LoRA embeddings."""

    def __init__(
        self,
        feature_size: int,
        output_size: int,
        num_layers: int,
        num_modules: int,
        per_rank_gen: bool,
        lora_r: int,
        layer_to_layer: bool,
        n_latent_queries: int,
        num_blocks: int,
        num_self_attn_per_block: int,
        shared_weights: bool,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_modules = num_modules
        self.per_rank_gen = per_rank_gen
        self.r = lora_r if per_rank_gen else 1
        self.layer_to_layer = layer_to_layer

        n_output_queries = num_layers * num_modules * self.r
        if layer_to_layer:
            n_output_queries = num_modules * self.r

        encoder_config = Idefics2PerceiverConfig(
            input_size=feature_size,
            num_blocks=num_blocks,
            num_self_attn_per_block=num_self_attn_per_block,
            shared_weights=shared_weights,
            n_latents=n_latent_queries,
            intermediate_size_factor=4,
            hidden_size=output_size,
            attn_implementation="flash_attention_2",
        )
        decoder_config = Idefics2PerceiverConfig(
            input_size=output_size,
            num_blocks=1,
            num_self_attn_per_block=0,
            shared_weights=False,
            n_latents=n_output_queries,
            intermediate_size_factor=4,
            hidden_size=output_size,
            attn_implementation="flash_attention_2",
        )
        self.perceiver = Idefics2Perceiver(encoder_config, decoder_config)

    def forward(
        self,
        ctx_features: Float[Tensor, "bs ..."],
        ctx_attn_mask: Integer[Tensor, "bs seq_len"] | None = None,
        ctx_position_ids: Integer[Tensor, "bs seq_len"] | None = None,
    ):
        if self.layer_to_layer:
            # ctx_features: [bs, num_layers, seq_len, hidden]
            if ctx_attn_mask is not None:
                ctx_attn_mask = repeat(
                    ctx_attn_mask,
                    "bs seq_len -> (num_layers bs) seq_len",
                    num_layers=self.num_layers,
                )
                ctx_features = rearrange(
                    ctx_features,
                    "bs num_layers seq_len d -> (num_layers bs) seq_len d",
                )
            if ctx_position_ids is not None:
                ctx_position_ids = repeat(
                    ctx_position_ids,
                    "1 seq_len -> 1 (num_layers seq_len)",
                    num_layers=self.num_layers,
                )
                ctx_features = rearrange(
                    ctx_features,
                    "1 num_layers seq_len d -> 1 (num_layers seq_len) d",
                )

        x = self.perceiver(ctx_features, ctx_attn_mask, ctx_position_ids)

        if self.layer_to_layer:
            per_layer_size = self.num_modules * self.r
            x = rearrange(
                x,
                "(num_layers bs) per_layer_sz d -> bs (num_layers per_layer_sz) d",
                num_layers=self.num_layers,
                per_layer_sz=per_layer_size,
            )

        lora_x = rearrange(
            x,
            "bs (n_layers n_modules r) d -> bs n_layers n_modules r d",
            n_modules=self.num_modules,
            n_layers=self.num_layers,
            r=self.r,
        )
        if not self.per_rank_gen:
            lora_x = lora_x.squeeze(3)

        return lora_x


# ---------------------------------------------------------------------------
# Residual MLP Block
# ---------------------------------------------------------------------------

class ResMLPBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout_rate: float = 0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Dropout(dropout_rate),
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_size),
            nn.LayerNorm(output_size),
        )

    def forward(self, x):
        return x + self.mlp(x)


# ---------------------------------------------------------------------------
# HyperLoRA
# ---------------------------------------------------------------------------

class HyperLoRA(nn.Module):
    """Generates LoRA A/B weights from context features via perceiver + MLP head."""

    def __init__(self, config: HypernetConfig):
        super().__init__()
        self.config = config
        self.lora_config = config.lora_config
        self.r = self.lora_config.r
        self.layer_indices = config.layer_indices
        self.n_layers = len(self.layer_indices)
        self.target_modules = tuple(sorted(self.lora_config.target_modules))
        self.num_modules = len(self.target_modules)
        self.d_in, self.d_out = config.feature_sizes
        self.d_latent = config.latent_size

        # Aggregator
        self.aggregator = PerceiverAggregator(
            feature_size=config.ctx_feature_size,
            output_size=config.latent_size,
            num_layers=self.n_layers,
            num_modules=self.num_modules,
            per_rank_gen=config.per_rank_gen,
            lora_r=self.r,
            layer_to_layer=config.layer_to_layer,
            n_latent_queries=config.n_latent_queries,
            num_blocks=config.num_blocks,
            num_self_attn_per_block=config.num_self_attn_per_block,
            shared_weights=config.shared_weights,
        )

        # Pre-head MLP layers
        self.layers = nn.Sequential(*[
            ResMLPBlock(self.d_latent, self.d_latent * 4, self.d_latent, config.dropout_rate)
            for _ in range(config.num_pre_head_layers)
        ])

        # Max output dim across modules
        self.d_lora = max(self.d_in[m] + self.d_out[m] for m in self.target_modules)

        # Bias terms (data-independent LoRA initialization)
        self.bias_A = nn.ParameterDict({
            m: nn.Parameter(torch.normal(
                0, 0.2 / (self.d_in[m] * self.r) ** 0.5,
                (self.n_layers, self.r, self.d_in[m]),
            ))
            for m in self.target_modules
        })
        self.bias_B = nn.ParameterDict({
            m: nn.Parameter(torch.zeros((self.n_layers, self.r, self.d_out[m])))
            for m in self.target_modules
        })

        # Learnable scalers
        self.scaler_A = nn.ParameterDict({
            m: nn.Parameter(torch.ones((1, self.n_layers, self.r, 1)))
            for m in self.target_modules
        })
        self.scaler_B = nn.ParameterDict({
            m: nn.Parameter(torch.zeros((1, self.n_layers, self.r, 1)))
            for m in self.target_modules
        })

        # Projection head: latent -> LoRA weights
        n_modules = len(self.target_modules)
        if n_modules == 1:
            self.head = Mix(
                "bs n_layers n_modules r d_latent -> bs n_layers n_modules r d_lora",
                weight_shape="n_layers d_latent d_lora",
                bias_shape=None,
                n_layers=self.n_layers,
                d_latent=self.d_latent,
                r=self.r,
                d_lora=self.d_lora,
            )
        else:
            self.head = Mix(
                "bs n_layers n_modules r d_latent -> bs n_layers n_modules r d_lora",
                weight_shape="n_layers n_modules d_latent d_lora",
                bias_shape=None,
                n_layers=self.n_layers,
                n_modules=n_modules,
                d_latent=self.d_latent,
                r=self.r,
                d_lora=self.d_lora,
            )

    def get_head_bias(self):
        return {
            m: dict(A=self.bias_A[m], B=self.bias_B[m])
            for m in self.target_modules
        }

    def _to_lora_dict(
        self, flat_loras: Float[Tensor, "bs n_layers n_modules r max_io_dim"]
    ) -> dict[str, dict[str, Float[Tensor, "bs n_layers r _"]]]:
        loras = unpack(
            flat_loras,
            [[] for _ in range(len(self.target_modules))],
            "bs n_layers * r max_io_dim",
        )

        lora_dict = dict()
        for module, lora in zip(self.target_modules, loras):
            A, B = unpack(
                lora[..., : self.d_in[module] + self.d_out[module]],
                [[self.d_in[module]], [self.d_out[module]]],
                "bs n_layers r *",
            )
            A = torch.einsum("ijkl,ijkl->ijkl", A, self.scaler_A[module])
            B = torch.einsum("ijkl,ijkl->ijkl", B, self.scaler_B[module])
            lora_dict[module] = dict(A=A, B=B)
        return lora_dict

    def forward(
        self,
        features: Float[Tensor, "bs ..."],
        attn_mask: Integer[Tensor, "bs seq_len"] | None = None,
        position_ids: Integer[Tensor, "bs seq_len"] | None = None,
    ):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            lora_emb = self.aggregator(features, attn_mask, position_ids)

        lora_emb = self.layers(lora_emb)
        norm = torch.norm(lora_emb, dim=-1, keepdim=True)
        norm_lora_emb = lora_emb / norm
        flat_loras = self.head(norm_lora_emb)
        return flat_loras

    def generate_weights(
        self,
        features: Float[Tensor, "bs ..."],
        attn_mask=None,
        position_ids=None,
    ):
        flat_loras = self.forward(features, attn_mask, position_ids)
        return self._to_lora_dict(flat_loras)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_peft_modules(model, peft_config):
    return [
        {"name": name, "module": module}
        for name, module in model.named_modules()
        if name.split(".")[-1] in peft_config.target_modules
        and isinstance(module, BaseTunerLayer)
        and check_target_module_exists(peft_config, name)
    ]


def get_peft_in_out_features(model, peft_config):
    in_features, out_features = dict(), dict()
    for info in get_peft_modules(model, peft_config):
        name = info["name"].split(".")[-1]
        module = info["module"]
        assert isinstance(module.base_layer, nn.Linear)
        if name not in in_features:
            in_features[name] = module.in_features
            out_features[name] = module.out_features
    return in_features, out_features


def get_num_layers(model):
    return len(get_layers(model))


def get_base_model(model):
    while hasattr(model, "model"):
        model = model.model
    return model


def get_model(model_name_or_path, train, requires_grad, peft_config=None, use_flash_attn=True):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if use_flash_attn else "eager",
    )
    if peft_config is not None:
        model = PeftModel(model, peft_config)
    model.train(train)
    for param in model.parameters():
        param.requires_grad = requires_grad
    return model


def get_tokenizer(model_name_or_path, train=True):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="right" if train else "left",
        truncation_side="left",
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


# ---------------------------------------------------------------------------
# ModulatedPretrainedModel (main wrapper)
# ---------------------------------------------------------------------------

class ModulatedPretrainedModel(nn.Module):
    """
    Wraps a frozen PeftModel + context encoder + hypernetwork.

    Forward: encode context -> generate LoRA -> apply to base model -> run forward.
    """

    def __init__(
        self,
        base_model: PeftModel,
        hypernet_config: HypernetConfig,
        ctx_encoder_type: str = "per_layer_activations",
        ctx_encoder_model_name: str | None = None,
        ctx_encoder_last_layer: int | None = None,
        ctx_encoder_layer_idx: int | None = None,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        self.device = base_model.device
        self.peft_config = base_model.peft_config["default"]
        self.hypernet_config = hypernet_config
        self.ctx_encoder_type = ctx_encoder_type
        self.ctx_encoder_model_name = ctx_encoder_model_name or base_model.config.name_or_path
        self.ctx_encoder_last_layer = ctx_encoder_last_layer
        self.ctx_encoder_layer_idx = ctx_encoder_layer_idx
        self.model_accepts_loss_kwargs = True

        self.register_module("base_model", base_model)
        self._init_model(use_flash_attn)
        self._init_head_weights()

    def _init_model(self, use_flash_attn: bool):
        # Disable PEFT adapter forward (we inject our own)
        self.base_model.disable_adapter_layers()

        # Create hypernetwork
        self.hypernet = HyperLoRA(self.hypernet_config).to(self.device).to(torch.float32)

        # Patch linear layers with LoRA forward
        self._patch_lora_forward()

        # Create context encoder
        encoder_model = get_model(
            self.ctx_encoder_model_name,
            train=self.base_model.training,
            requires_grad=False,
            use_flash_attn=use_flash_attn,
        )
        if self.ctx_encoder_type == "per_layer_activations":
            self.ctx_encoder = PerLayerActivationsEncoder(encoder_model, self.ctx_encoder_last_layer)
        elif self.ctx_encoder_type == "early_exit":
            self.ctx_encoder = EarlyExitEncoder(encoder_model, self.ctx_encoder_layer_idx)
        else:
            raise ValueError(f"Unknown ctx_encoder_type: {self.ctx_encoder_type}")

    def _patch_lora_forward(self):
        layers = get_layers(self.base_model)
        for layer_idx in self.hypernet.layer_indices:
            for info in get_peft_modules(layers[layer_idx], self.peft_config):
                module = info["module"]
                if getattr(module, "patched_forward", False):
                    continue
                module.forward_orig = module.forward
                module.patched_forward = True
                module.forward = partial(
                    lora_forward_packed,
                    self=module,
                    lora_dropout_p=self.peft_config.lora_dropout,
                    scaling=self.peft_config.lora_alpha,
                )

    @torch.no_grad()
    def _init_head_weights(self):
        r = self.hypernet_config.lora_config.r
        nn.init.normal_(
            self.hypernet.head.weight,
            mean=0,
            std=0.5 / sqrt(self.hypernet.config.latent_size + self.hypernet.d_lora * r),
        )

    # Delegate properties to base model
    @property
    def config(self):
        return self.base_model.config

    @property
    def generation_config(self):
        return self.base_model.generation_config

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def generate_weights(self, ctx_ids, ctx_attn_mask=None, ctx_position_ids=None):
        with torch.no_grad():
            ctx_features = self.ctx_encoder(
                input_ids=ctx_ids,
                attention_mask=ctx_attn_mask,
                position_ids=ctx_position_ids,
            )
        return self.hypernet.generate_weights(ctx_features, ctx_attn_mask, ctx_position_ids)

    def state_dict(self, *args, **kwargs):
        sd = self.hypernet.state_dict(*args, **kwargs)
        sd["base_model_name_or_path"] = self.base_model.name_or_path
        sd["hypernet_config"] = self.hypernet_config
        sd["ctx_encoder_type"] = self.ctx_encoder_type
        sd["ctx_encoder_model_name"] = self.ctx_encoder_model_name
        sd["ctx_encoder_last_layer"] = self.ctx_encoder_last_layer
        sd["ctx_encoder_layer_idx"] = self.ctx_encoder_layer_idx
        return sd

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        self.base_model_name_or_path = state_dict.pop("base_model_name_or_path")
        self.hypernet_config = state_dict.pop("hypernet_config")
        self.ctx_encoder_type = state_dict.pop("ctx_encoder_type")
        self.ctx_encoder_model_name = state_dict.pop("ctx_encoder_model_name")
        self.ctx_encoder_last_layer = state_dict.pop("ctx_encoder_last_layer")
        self.ctx_encoder_layer_idx = state_dict.pop("ctx_encoder_layer_idx")

        # Remove torch.compile prefix if present
        for k in list(state_dict.keys()):
            if k.startswith("_orig_mod."):
                state_dict[k[len("_orig_mod."):]] = state_dict.pop(k)

        return self.hypernet.load_state_dict(state_dict, strict=True)

    @classmethod
    def from_state_dict(cls, state_dict: dict, train: bool = True, use_flash_attn: bool = True, **kwargs):
        hypernet_config = state_dict["hypernet_config"]
        lora_config = hypernet_config.lora_config
        model_name = state_dict["base_model_name_or_path"]
        base_model = get_model(model_name, train=train, requires_grad=False, peft_config=lora_config, use_flash_attn=use_flash_attn)

        model = cls(
            base_model,
            hypernet_config,
            ctx_encoder_type=state_dict.get("ctx_encoder_type", "per_layer_activations"),
            ctx_encoder_model_name=state_dict.get("ctx_encoder_model_name"),
            ctx_encoder_last_layer=state_dict.get("ctx_encoder_last_layer"),
            ctx_encoder_layer_idx=state_dict.get("ctx_encoder_layer_idx"),
            use_flash_attn=use_flash_attn,
            **kwargs,
        )
        model.load_state_dict(state_dict)
        return model

    def forward(
        self,
        ctx_ids: Integer[Tensor, "n_ctx ctx_len"] | None = None,
        ctx_attn_mask: Integer[Tensor, "n_ctx ctx_len"] | None = None,
        ctx_position_ids: Integer[Tensor, "n_ctx ctx_len"] | None = None,
        n_ctx_chunks: Integer[Tensor, "n_ctx"] | None = None,
        n_queries: Integer[Tensor, "n_ctx"] | None = None,
        return_generated_lora: bool = False,
        *model_inputs_args: Any,
        **model_inputs_kwargs: dict[str, Any],
    ) -> tuple | ModelOutput:
        generated_loras = self.generate_weights(ctx_ids, ctx_attn_mask, ctx_position_ids)

        generated_loras = combine_lora(
            generated_loras,
            n_ctx_chunks,
            lora_bias=self.hypernet.get_head_bias() if self.hypernet.config.use_bias else None,
        )

        position_ids = model_inputs_kwargs.get("position_ids", None)

        if n_queries is None:
            if ctx_position_ids is None:
                n_queries = torch.ones(ctx_ids.shape[0], dtype=torch.int32, device=self.device)
            else:
                n_queries = torch.ones(
                    (ctx_position_ids == 0).sum(), dtype=torch.int32, device=self.device
                )

        apply_lora_to_layers(
            self.base_model,
            self.hypernet.layer_indices,
            generated_loras,
            n_queries,
            position_ids,
        )

        model_outputs = self.base_model(*model_inputs_args, **model_inputs_kwargs)

        if return_generated_lora:
            return model_outputs, generated_loras
        return model_outputs


# Register safe globals for torch.load
torch.serialization.add_safe_globals([
    HypernetConfig, LoraConfig, PeftType, TaskType, LoraRuntimeConfig, set,
])
