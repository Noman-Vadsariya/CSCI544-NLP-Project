#!/usr/bin/env bash
# Eval SPLADE + bridge-query neural RAG on hotpotQA_niah (needle-in-a-haystack).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$REPO_ROOT/data/raw_datasets/hotpotQA_niah/test.parquet"
GEMMA="$REPO_ROOT/checkpoints/gemma-2-2b-it-bin/pytorch_model.bin"
GEMMA_D2L="$REPO_ROOT/train_outputs/stage2_hotpotQA_gold_compact_finetune/pytorch_model.bin"
OUT="$REPO_ROOT/data/retrieved"

export XFORMERS_DISABLED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
SBATCH_EXPORT="ALL,XFORMERS_DISABLED=1,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,HF_HUB_DISABLE_TELEMETRY=1"

# doc2lora (Gemma stage-2 hypernet)
sbatch --export="$SBATCH_EXPORT" slurm/run_gpu.sbatch src/neural_retrieval_rag/neural_rag.py \
  --input "$DATA" \
  --pipeline doc2lora \
  --model_path "$GEMMA_D2L" \
  --output         "$OUT/hotpotQA_niah_needles_neural_d2l_gemma.json" \
  --gen_output     "$OUT/hotpotQA_niah_needles_neural_d2l_gemma_gen.json" \
  --metrics_output "$OUT/hotpotQA_niah_needles_neural_d2l_gemma_metrics.json"

# regular LLM (gemma-2-2b-it)
sbatch --export="$SBATCH_EXPORT" slurm/run_gpu.sbatch src/neural_retrieval_rag/neural_rag.py \
  --input "$DATA" \
  --pipeline regular \
  --model_path "$GEMMA" \
  --output         "$OUT/hotpotQA_niah_needles_neural_regular_gemma.json" \
  --gen_output     "$OUT/hotpotQA_niah_needles_neural_regular_gemma_gen.json" \
  --metrics_output "$OUT/hotpotQA_niah_needles_neural_regular_gemma_metrics.json"
