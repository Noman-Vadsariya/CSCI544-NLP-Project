"""Simplified stage 1 training: finetune hypernet on a dataset (1 chunk per context by default).

Example (from checkpoint):
    uv run python3 src/train_stage1.py \
        --checkpoint checkpoints/trained_d2l/gemma_2b_d2l/checkpoint-20000/pytorch_model.bin \
        --dataset squad_compact \
        --output_dir train_outputs/stage1_run

Example (from scratch with a base model):
    uv run python3 src/train_stage1.py \
        --model_name google/gemma-2-2b-it \
        --dataset squad_compact \
        --output_dir train_outputs/stage1_scratch

Example (with KL + LoRA L1 regularization):
    uv run python3 src/train_stage1.py \
        --model_name google/gemma-2-2b-it \
        --dataset hotpotQA_compact \
        --output_dir train_outputs/stage1_hotpot_scratch \
        --use_kl_loss \
        --gen_lora_l1_reg_coef 1e-4
"""

import argparse
import logging
import os
from functools import partial

import numpy as np
import wandb

import torch
from datasets import disable_caching
from transformers import AutoConfig, set_seed
from transformers import TrainingArguments

from ctx_to_lora.configs import (
    HypernetArguments,
    AggregatorArguments,
    CtxEncoderArguments,
)
from ctx_to_lora.data.collator import flatten_if_not_packed
from ctx_to_lora.data.processing import get_tokenized_dataset, pack
from ctx_to_lora.model_loading import (
    get_lora_config,
    get_model_and_tokenizer,
    get_tokenizer,
)
from ctx_to_lora.modeling.hypernet import (
    ModulatedPretrainedModel,
    get_hypernet_config,
)
from ctx_to_lora.metrics import (
    Evaluator,
    compute_metrics,
    compute_per_token_acc,
    compute_perplexity,
    compute_prefix_matching,
)
from ctx_to_lora.trainer import train_model
from ctx_to_lora.utils import compile_linear, log_num_train_params

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OMP_NUM_THREADS"] = "23"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

disable_caching()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1 hypernet training")

    # Model either from --checkpoint OR --model_name (from scratch)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="Path to pytorch_model.bin checkpoint")
    source.add_argument(
        "--model_name",
        help="HuggingFace model name to init from scratch (e.g. google/gemma-2-2b-it)",
    )

    # From-scratch hypernetwork config
    p.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    p.add_argument(
        "--target_modules",
        nargs="+",
        default=["down_proj"],
        help="LoRA target modules",
    )
    p.add_argument(
        "--ctx_encoder_type",
        default="per_layer_activations",
        choices=["early_exit", "embed_only", "per_layer_activations"],
    )
    p.add_argument(
        "--ctx_encoder_model",
        default=None,
        help="Separate ctx encoder model (default: same as base)",
    )
    p.add_argument(
        "--n_latent_queries",
        type=int,
        default=8,
        help="Perceiver latent queries",
    )
    p.add_argument(
        "--latent_size",
        type=int,
        default=512,
        help="Hypernet latent size",
    )
    p.add_argument(
        "--num_blocks",
        type=int,
        default=8,
        help="Number of perceiver blocks",
    )
    p.add_argument(
        "--num_self_attn_per_block",
        type=int,
        default=0,
        help="Number of self-attention layers per perceiver block",
    )
    p.add_argument(
        "--per_layer_processing",
        action="store_true",
        help="Enable per-layer processing in the hypernet head",
    )
    p.add_argument(
        "--quantize_ctx_encoder",
        action="store_true",
        help="4-bit quantize the frozen context encoder to save memory",
    )

    p.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (e.g. squad_compact, pwc_compact)",
    )
    p.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Cap training samples",
    )
    p.add_argument(
        "--val_dataset",
        default=None,
        help="Validation dataset name (default: same as --dataset)",
    )
    p.add_argument(
        "--max_val_samples",
        type=int,
        default=200,
        help="Max validation samples",
    )
    p.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Run validation every N steps (default: same as save_steps)",
    )
    p.add_argument(
        "--output_dir",
        default="train_outputs/stage1",
        help="Output directory",
    )
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)

    # Packing
    p.add_argument("--max_packed_inp_len", type=int, default=4096)
    p.add_argument("--max_packed_ctx_len", type=int, default=4096)

    # Loss / regularization controls
    p.add_argument(
        "--use_kl_loss",
        action="store_true",
        help="Enable KL distillation loss if supported by the dataset/trainer pipeline",
    )
    p.add_argument(
        "--use_per_ctx_average_loss",
        action="store_true",
        help="Enable per-context average loss in the trainer",
    )
    p.add_argument(
        "--gen_lora_l1_reg_coef",
        type=float,
        default=0.0,
        help="L1 regularization coefficient for generated LoRA weights",
    )

    # Torch compile / logging
    p.add_argument("--compile", action="store_true", help="torch.compile the hypernet")
    p.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    p.add_argument("--bf16", action="store_true", default=torch.cuda.is_available())

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, weights_only=False)
        model = ModulatedPretrainedModel.from_state_dict(state_dict, train=True)

        tokenizer = get_tokenizer(model.base_model.config.name_or_path, train=True)
        ctx_name = model.ctx_encoder_args.ctx_encoder_model_name_or_path
        if ctx_name is None:
            ctx_name = model.base_model.config.name_or_path
        ctx_tokenizer = get_tokenizer(ctx_name, train=True)

    else:
        logger.info(f"Initializing from scratch with base model: {args.model_name}")
        lora_config = get_lora_config(
            args.model_name,
            lora_r=args.lora_r,
            target_modules=args.target_modules,
        )
        base_model, tokenizer = get_model_and_tokenizer(
            model_name_or_path=args.model_name,
            train=True,
            requires_grad=False,
            peft_config=lora_config,
        )

        ctx_name = args.ctx_encoder_model or args.model_name
        ctx_encoder_model_config = AutoConfig.from_pretrained(
            ctx_name,
            trust_remote_code=True,
        )
        ctx_tokenizer = get_tokenizer(ctx_name, train=True)

        hypernet_args = HypernetArguments(
            latent_size=args.latent_size,
            per_rank_gen=True,
            per_layer_processing=args.per_layer_processing,
        )
        aggregator_args = AggregatorArguments(
            n_latent_queries=args.n_latent_queries,
            num_blocks=args.num_blocks,
            num_self_attn_per_block=args.num_self_attn_per_block,
        )
        ctx_encoder_args = CtxEncoderArguments(
            ctx_encoder_model_name_or_path=args.ctx_encoder_model,
            ctx_encoder_type=args.ctx_encoder_type,
            quantize_ctx_encoder=args.quantize_ctx_encoder,
        )

        if ctx_encoder_args.layer_idx is None:
            ctx_encoder_args.layer_idx = ctx_encoder_model_config.num_hidden_layers // 4
            logger.info(
                f"Using first {ctx_encoder_args.layer_idx} layers as context encoder"
            )

        hypernet_config = get_hypernet_config(
            base_model,
            ctx_encoder_model_config,
            hypernet_args,
            aggregator_args,
            ctx_encoder_args,
        )
        model = ModulatedPretrainedModel(base_model, hypernet_config, ctx_encoder_args)

    # Freeze base model and ctx encoder; only train hypernet
    for p in model.ctx_encoder.parameters():
        p.requires_grad = False
    for p in model.base_model.parameters():
        p.requires_grad = False

    # Enable gradient checkpointing to reduce activation memory
    model.base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if hasattr(model.ctx_encoder, "gradient_checkpointing_enable"):
        model.ctx_encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    if args.compile:
        model.hypernet.compile(fullgraph=True, mode="max-autotune")

        # Compile ctx encoder and base model to match original train.py
        logger.info("Compiling ctx_encoder linear layers")
        compile_linear(model.ctx_encoder.base_model)

        from peft import PeftModel
        if isinstance(model.base_model, PeftModel):
            base_model_inner = model.base_model.base_model
        else:
            base_model_inner = model.base_model
        logger.info("Compiling base_model")
        base_model_inner.compile(fullgraph=True, mode="max-autotune")

    model.train()
    log_num_train_params(model)

    logger.info(f"Loading dataset: {args.dataset}")
    ctx_max_pos = model.ctx_encoder.config.max_position_embeddings

    train_ds_raw = get_tokenized_dataset(
        ds_name=args.dataset,
        split="train",
        max_qas_len=-1,
        max_qas_per_sample=-1,
        base_model_max_len=model.base_model.config.max_position_embeddings,
        tokenizer=tokenizer,
        ctx_model_max_len=ctx_max_pos,
        ctx_tokenizer=ctx_tokenizer,
        add_ctx_to_chat=False,
        add_negative_prompt=False,
        max_ctx_chunk_len=ctx_max_pos,  # no chunking: chunk_len = max_pos
        min_ctx_chunk_len=-1,
        num_chunk_probs=None,
        max_ctx_chunk_num=None,
        use_kl_loss=args.use_kl_loss,
    )

    if args.max_train_samples and args.max_train_samples < len(train_ds_raw):
        train_ds_raw = train_ds_raw.take(args.max_train_samples)

    logger.info(f"Dataset size before packing: {len(train_ds_raw)}")

    train_ds = pack(
        {args.dataset: train_ds_raw},
        max_packed_inp_len=args.max_packed_inp_len,
        max_packed_ctx_len=args.max_packed_ctx_len,
        max_packed_size=-1,
        seed=args.seed,
        num_proc=4,
    )
    logger.info(f"Dataset size after packing: {len(train_ds)}")

    # --- Validation dataset ---
    val_ds_name = args.val_dataset or args.dataset
    val_ds_raw = get_tokenized_dataset(
        ds_name=val_ds_name,
        split="validation",
        max_qas_len=-1,
        max_qas_per_sample=-1,
        base_model_max_len=model.base_model.config.max_position_embeddings,
        tokenizer=tokenizer,
        ctx_model_max_len=ctx_max_pos,
        ctx_tokenizer=ctx_tokenizer,
        add_ctx_to_chat=False,
        add_negative_prompt=False,
        max_ctx_chunk_len=ctx_max_pos,
        min_ctx_chunk_len=-1,
        num_chunk_probs=None,
        max_ctx_chunk_num=None,
        use_kl_loss=args.use_kl_loss,
    )

    if val_ds_raw is None:
        # No validation split available; fall back to a slice of training data
        logger.info("No validation split found; using a slice of training data")
        val_ds_raw = train_ds_raw.take(args.max_val_samples)
        train_ds_raw = train_ds_raw.skip(args.max_val_samples)

    val_indices = np.random.permutation(len(val_ds_raw))[: args.max_val_samples]
    val_ds = val_ds_raw.select(val_indices)
    val_ds = {val_ds_name: val_ds}
    logger.info(f"Validation dataset size: {len(val_indices)}")

    eval_steps = args.eval_steps or args.save_steps
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=1,  # always 1 with packing
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps",
        eval_steps=eval_steps,
        bf16=args.bf16,
        fp16=False,
        report_to="wandb" if args.wandb else "none",
        optim="adamw_torch",
        remove_unused_columns=False,
        seed=args.seed,
        run_name=os.path.basename(args.output_dir),
    )

    # Trainer-specific custom args
    training_args.gen_lora_l1_reg_coef = args.gen_lora_l1_reg_coef
    training_args.use_per_ctx_average_loss = args.use_per_ctx_average_loss
    training_args.use_kl_loss = args.use_kl_loss

    if args.wandb:
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "ctx_to_lora"),
            name=os.path.basename(args.output_dir),
            config=vars(args),
        )

    train_model(
        model=model,
        training_args=training_args,
        train_dataset=train_ds,
        val_dataset=val_ds,
        train_collator=flatten_if_not_packed,
        compute_metrics=partial(
            compute_metrics,
            evaluator=Evaluator(
                [compute_per_token_acc, compute_prefix_matching, compute_perplexity]
            ),
        ),
    )
    logger.info(f"Training complete. Saved to {args.output_dir}")


if __name__ == "__main__":
    main()