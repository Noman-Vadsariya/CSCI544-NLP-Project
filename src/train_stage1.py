"""Simplified stage 1 training: finetune hypernet on a dataset (1 chunk per context, no distillation).

Usage:
    uv run python3 src/train_stage1.py \
        --checkpoint checkpoints/trained_d2l/gemma_2b_d2l/checkpoint-20000/pytorch_model.bin \
        --dataset squad_compact \
        --output_dir train_outputs/stage1_run \
        --num_train_epochs 3 \
        --learning_rate 2e-5 \
        --gradient_accumulation_steps 8 \
        --max_packed_inp_len 4096 \
        --max_packed_ctx_len 4096
"""
import argparse
import logging
import os
import wandb

import torch
from datasets import disable_caching
from transformers import set_seed
from transformers import TrainingArguments

from ctx_to_lora.data.collator import flatten_if_not_packed
from ctx_to_lora.data.processing import get_tokenized_dataset, pack
from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel
from ctx_to_lora.trainer import train_model
from ctx_to_lora.utils import log_num_train_params

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OMP_NUM_THREADS"] = "23"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1 hypernet training")
    p.add_argument("--checkpoint", required=True, help="Path to pytorch_model.bin checkpoint")
    p.add_argument("--dataset", required=True, help="Dataset name (e.g. squad_compact, pwc_compact)")
    p.add_argument("--max_train_samples", type=int, default=None, help="Cap training samples")
    p.add_argument("--output_dir", default="train_outputs/stage1", help="Output directory")
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)

    # used for training efficiency, packs multple short samples together and uses position ids to seperate them
    p.add_argument("--max_packed_inp_len", type=int, default=4096)
    p.add_argument("--max_packed_ctx_len", type=int, default=4096)  

    # torch compile for additional optimization
    p.add_argument("--compile", action="store_true", help="torch.compile the hypernet")
    p.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    p.add_argument("--bf16", action="store_true", default=torch.cuda.is_available())
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model from checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, weights_only=False)
    model = ModulatedPretrainedModel.from_state_dict(state_dict, train=True)

    tokenizer = get_tokenizer(model.base_model.config.name_or_path, train=True)
    ctx_name = model.ctx_encoder_args.ctx_encoder_model_name_or_path
    if ctx_name is None:
        ctx_name = model.base_model.config.name_or_path
    ctx_tokenizer = get_tokenizer(ctx_name, train=True)

    # freeze base model and ctx encoder, only train hypernet
    for p in model.ctx_encoder.parameters():
        p.requires_grad = False
    for p in model.base_model.parameters():
        p.requires_grad = False

    if args.compile:
        model.hypernet.compile(fullgraph=True, mode="max-autotune")

    model.train()
    log_num_train_params(model)

    # ── Load and tokenize dataset ───────────────────────────────────────
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
        use_kl_loss=False,
    )

    if args.max_train_samples and args.max_train_samples < len(train_ds_raw):
        train_ds_raw = train_ds_raw.take(args.max_train_samples)

    logger.info(f"Dataset size before packing: {len(train_ds_raw)}")

    # Pack dataset
    train_ds = pack(
        {args.dataset: train_ds_raw},
        max_packed_inp_len=args.max_packed_inp_len,
        max_packed_ctx_len=args.max_packed_ctx_len,
        max_packed_size=-1,
        seed=args.seed,
        num_proc=4,
    )
    logger.info(f"Dataset size after packing: {len(train_ds)}")

    # Training args 
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
        bf16=args.bf16,
        fp16=False,
        report_to="wandb" if args.wandb else "none",
        optim="adamw_torch",
        remove_unused_columns=False,
        seed=args.seed,
        run_name=os.path.basename(args.output_dir),
    )

    # these are expected by the trainer
    training_args.gen_lora_l1_reg_coef = 0.0
    training_args.use_per_ctx_average_loss = False
    training_args.use_kl_loss = False

    # Train 
    if args.wandb:
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "ctx_to_lora"),
            name=os.path.basename(args.output_dir),
        )

    train_model(
        model=model,
        training_args=training_args,
        train_dataset=train_ds,
        val_dataset=None,
        train_collator=flatten_if_not_packed,
    )
    logger.info(f"Training complete. Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
