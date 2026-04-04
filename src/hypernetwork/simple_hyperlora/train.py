"""
Training entry point for simplified hypernetwork LoRA generation.

Usage (from scratch):
    python -m simple_hyperlora.train \
        --model_name google/gemma-2-2b-it \
        --dataset squad_compact \
        --output_dir train_outputs/simple_run

Usage (from checkpoint):
    python -m simple_hyperlora.train \
        --checkpoint path/to/pytorch_model.bin \
        --dataset squad_compact \
        --output_dir train_outputs/simple_run
"""

import argparse
import logging
import os

import torch
from datasets import disable_caching
from peft.utils import PeftType
from transformers import AutoConfig, TrainingArguments, set_seed

from simple_hyperlora.data import (
    flatten_if_not_packed,
    get_tokenized_dataset,
    pack_datasets,
)
from simple_hyperlora.model import (
    HypernetConfig,
    ModulatedPretrainedModel,
    get_model,
    get_num_layers,
    get_peft_in_out_features,
    get_tokenizer,
)
from simple_hyperlora.trainer import (
    CrossEntropyTrainer,
    DistillationTrainer,
    get_decay_parameter_names,
)

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

disable_caching()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Simple HyperLoRA training")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="Path to pytorch_model.bin checkpoint")
    source.add_argument("--model_name", help="HuggingFace model name (from scratch)")

    # Hypernetwork config (from scratch only)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--target_modules", nargs="+", default=["down_proj"])
    p.add_argument("--ctx_encoder_type", default="per_layer_activations",
                    choices=["per_layer_activations", "early_exit"])
    p.add_argument("--ctx_encoder_model", default=None)
    p.add_argument("--ctx_encoder_layer_idx", type=int, default=None,
                    help="Layer index for early_exit encoder (default: num_layers//4)")
    p.add_argument("--ctx_encoder_last_layer", type=int, default=None,
                    help="Last layer for per_layer_activations (default: -1)")
    p.add_argument("--n_latent_queries", type=int, default=208)
    p.add_argument("--latent_size", type=int, default=512)
    p.add_argument("--num_pre_head_layers", type=int, default=4)
    p.add_argument("--per_rank_gen", action="store_true", default=True)
    p.add_argument("--num_blocks", type=int, default=9)
    p.add_argument("--num_self_attn_per_block", type=int, default=2)
    p.add_argument("--dropout_rate", type=float, default=0.0)

    # Dataset
    p.add_argument("--dataset", required=True, nargs="+")
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--add_ctx_to_chat", action="store_true", default=False)

    # Training
    p.add_argument("--output_dir", default="train_outputs/simple_run")
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_packed_inp_len", type=int, default=4096)
    p.add_argument("--max_packed_ctx_len", type=int, default=4096)

    # Loss
    p.add_argument("--use_kl_loss", action="store_true", default=False)
    p.add_argument("--use_per_ctx_average_loss", action="store_true", default=False)
    p.add_argument("--gen_lora_l1_reg_coef", type=float, default=0.0)

    # Misc
    p.add_argument("--compile", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--bf16", action="store_true", default=torch.cuda.is_available())
    return p.parse_args()


def get_lora_config(model_name, lora_r, target_modules, lora_dropout=0.0):
    from peft import get_peft_config
    return get_peft_config({
        "peft_type": PeftType.LORA,
        "r": lora_r,
        "base_model_name_or_path": model_name,
        "task_type": "CAUSAL_LM",
        "lora_dropout": lora_dropout,
        "lora_alpha": lora_r ** (3 / 2) * 2,
        "target_modules": target_modules,
    })


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, weights_only=False)
        model = ModulatedPretrainedModel.from_state_dict(state_dict, train=True)
        base_name = model.base_model.config.name_or_path
        ctx_name = model.ctx_encoder_model_name or base_name
    else:
        logger.info(f"Initializing from scratch: {args.model_name}")
        lora_config = get_lora_config(args.model_name, args.lora_r, args.target_modules)
        base_model = get_model(args.model_name, train=True, requires_grad=False, peft_config=lora_config)

        base_name = args.model_name
        ctx_name = args.ctx_encoder_model or base_name
        ctx_config = AutoConfig.from_pretrained(ctx_name, trust_remote_code=True)

        layer_to_layer = args.ctx_encoder_type == "per_layer_activations"
        layer_idx = args.ctx_encoder_layer_idx
        if args.ctx_encoder_type == "early_exit" and layer_idx is None:
            layer_idx = ctx_config.num_hidden_layers // 4

        hypernet_config = HypernetConfig(
            latent_size=args.latent_size,
            lora_config=lora_config,
            base_hidden_size=base_model.config.hidden_size,
            layer_indices=torch.arange(get_num_layers(base_model), device=base_model.device),
            feature_sizes=get_peft_in_out_features(base_model, lora_config),
            num_blocks=args.num_blocks,
            num_self_attn_per_block=args.num_self_attn_per_block,
            shared_weights=True,
            n_latent_queries=args.n_latent_queries,
            ctx_feature_size=ctx_config.hidden_size,
            per_rank_gen=args.per_rank_gen,
            num_pre_head_layers=args.num_pre_head_layers,
            dropout_rate=args.dropout_rate,
            use_bias=True,
            layer_to_layer=layer_to_layer,
        )

        model = ModulatedPretrainedModel(
            base_model, hypernet_config,
            ctx_encoder_type=args.ctx_encoder_type,
            ctx_encoder_model_name=ctx_name if ctx_name != base_name else None,
            ctx_encoder_last_layer=args.ctx_encoder_last_layer,
            ctx_encoder_layer_idx=layer_idx,
        )

    # Freeze base + ctx encoder, train only hypernet
    for p in model.ctx_encoder.parameters():
        p.requires_grad = False
    for p in model.base_model.parameters():
        p.requires_grad = False
    model.base_model.gradient_checkpointing_enable()
    if hasattr(model.ctx_encoder, "base_model") and hasattr(model.ctx_encoder.base_model, "gradient_checkpointing_enable"):
        model.ctx_encoder.base_model.gradient_checkpointing_enable()

    if args.compile:
        model.hypernet = torch.compile(model.hypernet, fullgraph=True, mode="max-autotune")

    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Load tokenizers
    tokenizer = get_tokenizer(base_name, train=True)
    ctx_tokenizer = get_tokenizer(ctx_name, train=True)
    model.base_model.config.pad_token_id = tokenizer.pad_token_id

    # Load and tokenize datasets
    ds_dict = {}
    for ds_name in args.dataset:
        logger.info(f"Loading dataset: {ds_name}")
        ds = get_tokenized_dataset(
            ds_name=ds_name,
            split="train",
            tokenizer=tokenizer,
            ctx_tokenizer=ctx_tokenizer,
            add_ctx_to_chat=args.add_ctx_to_chat,
            use_kl_loss=args.use_kl_loss,
        )
        if args.max_train_samples and args.max_train_samples < len(ds):
            ds = ds.select(range(args.max_train_samples))
        ds_dict[ds_name] = ds

    # Pack
    train_ds = pack_datasets(
        ds_dict,
        max_packed_inp_len=args.max_packed_inp_len,
        max_packed_ctx_len=args.max_packed_ctx_len,
        seed=args.seed,
    )

    # Training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=False,
        report_to="wandb" if args.wandb else "none",
        optim="adamw_torch",
        remove_unused_columns=False,
        seed=args.seed,
        run_name=os.path.basename(args.output_dir),
    )

    TrainerCls = DistillationTrainer if args.use_kl_loss else CrossEntropyTrainer
    trainer = TrainerCls(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=flatten_if_not_packed,
        gen_lora_l1_reg_coef=args.gen_lora_l1_reg_coef,
        use_per_ctx_average_loss=args.use_per_ctx_average_loss,
    )
    trainer.get_decay_parameter_names = get_decay_parameter_names

    if args.wandb:
        import wandb
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "simple_hyperlora"),
            name=os.path.basename(args.output_dir),
        )

    trainer.train()
    trainer.save_model()
    logger.info(f"Training complete. Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
