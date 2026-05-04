#!/usr/bin/env bash
# Stage-1 hypernetwork training from scratch on hotpotQA_gold_compact
# (no pre-trained doc2lora checkpoint).
set -euo pipefail

sbatch slurm/run_gpu.sbatch src/hypernetwork/train_stage1.py \
  --model_name google/gemma-2-2b-it \
  --dataset hotpotQA_gold_compact \
  --output_dir train_outputs/stage1_hotpotQA_gold_compact_scratch \
  --num_train_epochs 10 \
  --learning_rate 2e-5 \
  --gradient_accumulation_steps 8 \
  --max_packed_inp_len 4096 \
  --max_packed_ctx_len 3072 \
  --logging_steps 10 \
  --bf16 \
  --wandb \
  --ctx_encoder_type per_layer_activations \
  --num_blocks 9 \
  --per_layer_processing \
  --quantize_ctx_encoder \
  --use_kl_loss \
  --use_per_ctx_average_loss \
  --gen_lora_l1_reg_coef 0.1
