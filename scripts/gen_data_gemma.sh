#!/usr/bin/env bash
# Generate Gemma-2-2b-it teacher responses for hypernet training.
# Outputs land under data/self_gen/google--gemma-2-2b-it_temp_*/...
set -euo pipefail

TEACHER="google/gemma-2-2b-it"

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
