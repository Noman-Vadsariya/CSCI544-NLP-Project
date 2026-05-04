#!/usr/bin/env bash
# Generation-only eval over pre-retrieved SPLADE neural contexts (HotpotQA + ASQA).
# Uses existing retrieval JSONs in $RETRIEVED — does NOT re-run retrieval.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RETRIEVED="$REPO_ROOT/data/retrieved"

GEMMA_D2L_HOTPOT="$REPO_ROOT/train_outputs/stage2_combined_gold_dataset_finetune/pytorch_model.bin"
GEMMA_D2L_ASQA="$REPO_ROOT/train_outputs/stage2_hotpotQA_gold_compact_finetune/pytorch_model.bin"
GEMMA_BASE="$REPO_ROOT/checkpoints/gemma-2-2b-it-bin/pytorch_model.bin"

export XFORMERS_DISABLED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================
# HotpotQA — Neural retrieval, short answers
# ============================================================

# doc2lora (Gemma stage-2 hypernet)
sbatch slurm/run_gpu.sbatch src/standard_rag/gen_from_retrieved.py \
  --retrieved_input "$RETRIEVED/hotpotQA_compact_neural_d2l_gemma.json" \
  --pipeline doc2lora \
  --model_path "$GEMMA_D2L_HOTPOT" \
  --answer_style short \
  --max_new_tokens 32 \
  --gen_output "$RETRIEVED/hotpotQA_compact_neural_d2l_gemma_topk_gen.json"

# regular LLM (gemma-2-2b-it)
sbatch slurm/run_gpu.sbatch src/standard_rag/gen_from_retrieved.py \
  --retrieved_input "$RETRIEVED/hotpotQA_compact_neural_regular_gemma.json" \
  --pipeline regular \
  --model_path "$GEMMA_BASE" \
  --answer_style short \
  --max_new_tokens 32 \
  --gen_output "$RETRIEVED/hotpotQA_compact_neural_regular_gemma_topk_gen.json"

# ============================================================
# ASQA — Neural retrieval, full/long answers
# ============================================================

# doc2lora (Gemma stage-2 hypernet)
sbatch slurm/run_gpu.sbatch src/standard_rag/gen_from_retrieved.py \
  --retrieved_input "$RETRIEVED/asqa_gold_neural_d2l_gemma.json" \
  --pipeline doc2lora \
  --model_path "$GEMMA_D2L_ASQA" \
  --answer_style full \
  --no_query_prefix \
  --max_new_tokens 128 \
  --gen_output "$RETRIEVED/asqa_gold_neural_d2l_gemma_topk_gen.json"

# regular LLM (gemma-2-2b-it)
sbatch slurm/run_gpu.sbatch src/standard_rag/gen_from_retrieved.py \
  --retrieved_input "$RETRIEVED/asqa_gold_neural_regular_gemma.json" \
  --pipeline regular \
  --model_path "$GEMMA_BASE" \
  --answer_style full \
  --max_new_tokens 128 \
  --gen_output "$RETRIEVED/asqa_gold_neural_regular_gemma_topk_gen.json"
