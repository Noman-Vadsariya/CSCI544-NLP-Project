#!/usr/bin/env bash
# Eval generation-only (gold context, no retrieval) on hotpotQA_gold_compact.
# Uses the same prompts as src/standard_rag/rag_colbert_reranker.py.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$REPO_ROOT/data/raw_datasets/hotpotQA_gold_compact/test/ds.parquet"
GEMMA_BASE="$REPO_ROOT/checkpoints/gemma-2-2b-it-bin/pytorch_model.bin"
GEMMA_D2L="$REPO_ROOT/train_outputs/stage2_combined_gold_dataset_finetune/pytorch_model.bin"
OUT="$REPO_ROOT/data/retrieved"

export XFORMERS_DISABLED=1

mkdir -p "$OUT"

# doc2lora (Gemma stage-2 hypernet)
sbatch slurm/run_gpu.sbatch src/evaluation/generate_gold_hotpot.py \
  --input "$DATA" \
  --mode hypernet \
  --model_path "$GEMMA_D2L" \
  --pred    "$OUT/hotpotQA_gold_hypernet_gemma_pred.json" \
  --gold    "$OUT/hotpotQA_gold_hypernet_gemma_gold.json" \
  --jsonl   "$OUT/hotpotQA_gold_hypernet_gemma_outputs.jsonl" \
  --summary "$OUT/hotpotQA_gold_hypernet_gemma_summary.json"

# regular LLM (gemma-2-2b-it)
sbatch slurm/run_gpu.sbatch src/evaluation/generate_gold_hotpot.py \
  --input "$DATA" \
  --mode standard \
  --model_path "$GEMMA_BASE" \
  --pred    "$OUT/hotpotQA_gold_regular_gemma_pred.json" \
  --gold    "$OUT/hotpotQA_gold_regular_gemma_gold.json" \
  --jsonl   "$OUT/hotpotQA_gold_regular_gemma_outputs.jsonl" \
  --summary "$OUT/hotpotQA_gold_regular_gemma_summary.json"
