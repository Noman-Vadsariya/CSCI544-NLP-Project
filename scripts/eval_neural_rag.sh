#!/usr/bin/env bash
# Eval SPLADE + bridge-query neural RAG on hotpotQA_compact and ASQA gold subset.
# Submits both retrieval and generation in a single end-to-end job.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOTPOT_DATA="$REPO_ROOT/data/raw_datasets/hotpotQA_compact/test/ds.parquet"
ASQA_DATA="$REPO_ROOT/data/raw_datasets/asqa_compact/test/ds_gold_subset.parquet"
GEMMA_BASE="$REPO_ROOT/checkpoints/gemma-2-2b-it-bin/pytorch_model.bin"
GEMMA_D2L_HOTPOT="$REPO_ROOT/train_outputs/stage2_combined_gold_dataset_finetune/pytorch_model.bin"
OUT="$REPO_ROOT/data/retrieved"

export XFORMERS_DISABLED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================
# hotpotQA_compact
# ============================================================

DATA="$HOTPOT_DATA"

# doc2lora (Gemma stage-2 hypernet)
sbatch slurm/run_gpu.sbatch src/neural_retrieval_rag/neural_rag.py \
  --input "$DATA" \
  --pipeline doc2lora \
  --model_path "$GEMMA_D2L_HOTPOT" \
  --output         "$OUT/hotpotQA_compact_neural_d2l_gemma.json" \
  --gen_output     "$OUT/hotpotQA_compact_neural_d2l_gemma_gen.json" \
  --metrics_output "$OUT/hotpotQA_compact_neural_d2l_gemma_metrics.json"

# regular LLM (gemma-2-2b-it)
sbatch slurm/run_gpu.sbatch src/neural_retrieval_rag/neural_rag.py \
  --input "$DATA" \
  --pipeline regular \
  --model_path "$GEMMA_BASE" \
  --output         "$OUT/hotpotQA_compact_neural_regular_gemma.json" \
  --gen_output     "$OUT/hotpotQA_compact_neural_regular_gemma_gen.json" \
  --metrics_output "$OUT/hotpotQA_compact_neural_regular_gemma_metrics.json"

# ============================================================
# ASQA gold subset
# ============================================================

DATA="$ASQA_DATA"

# doc2lora (Gemma stage-2 hypernet)
sbatch slurm/run_gpu.sbatch src/neural_retrieval_rag/neural_rag.py \
  --input "$DATA" \
  --pipeline doc2lora \
  --answer_style full \
  --max_new_tokens 128 \
  --model_path "$GEMMA_D2L_HOTPOT" \
  --output         "$OUT/asqa_gold_neural_d2l_gemma.json" \
  --gen_output     "$OUT/asqa_gold_neural_d2l_gemma_gen.json" \
  --metrics_output "$OUT/asqa_gold_neural_d2l_gemma_metrics.json"

# regular LLM (gemma-2-2b-it)
sbatch slurm/run_gpu.sbatch src/neural_retrieval_rag/neural_rag.py \
  --input "$DATA" \
  --pipeline regular \
  --answer_style full \
  --max_new_tokens 128 \
  --model_path "$GEMMA_BASE" \
  --output         "$OUT/asqa_gold_neural_regular_gemma.json" \
  --gen_output     "$OUT/asqa_gold_neural_regular_gemma_gen.json" \
  --metrics_output "$OUT/asqa_gold_neural_regular_gemma_metrics.json"
