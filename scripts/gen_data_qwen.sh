#!/usr/bin/env bash
# Generate Qwen3-4B-Instruct-2507 teacher responses for hypernet training.
# Outputs land under data/self_gen/Qwen--Qwen3-4B-Instruct-2507_temp_*/...
set -euo pipefail

TEACHER="Qwen/Qwen3-4B-Instruct-2507"

sbatch slurm/run_gpu.sbatch data/self_generate_qa.py \
  --vllm_model "$TEACHER" \
  --ds_names gen_hotpotQA_compact gen_hotpotQA_gold_compact gen_asqa_compact \
  --split train \
  --closed_qa_prob 1.0

sbatch slurm/run_gpu.sbatch data/self_generate_qa.py \
  --vllm_model "$TEACHER" \
  --ds_names gen_combined_gold_dataset \
  --split train \
  --closed_qa_prob 1.0 \
  --do_truncate
