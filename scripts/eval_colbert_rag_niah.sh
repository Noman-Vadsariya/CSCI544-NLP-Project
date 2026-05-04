#!/usr/bin/env bash
# Eval ColBERT-reranker RAG on hotpotQA_niah (needle-in-a-haystack).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$REPO_ROOT/data/raw_datasets/hotpotQA_niah/test.parquet"
EMB="$REPO_ROOT/checkpoints/bge-large-en-v1.5"
GEMMA_BASE="$REPO_ROOT/checkpoints/gemma-2-2b-it-bin/pytorch_model.bin"
GEMMA_D2L="$REPO_ROOT/train_outputs/stage2_hotpotQA_gold_compact_finetune/pytorch_model.bin"
PINECONE_NAMESPACE=hotpotqa_niah_needles_v1
OUT="$REPO_ROOT/data/retrieved"

export XFORMERS_DISABLED=1

# doc2lora (Gemma stage-2 hypernet)
sbatch slurm/run_gpu.sbatch src/standard_rag/rag_colbert_reranker.py \
  --input "$DATA" \
  --pipeline doc2lora \
  --model_path "$GEMMA_D2L" \
  --embedding_model "$EMB" \
  --pinecone_namespace "$PINECONE_NAMESPACE" \
  --retrieved_output "$OUT/hotpotQA_niah_needles_colbert_d2l_gemma.json" \
  --gen_output       "$OUT/hotpotQA_niah_needles_colbert_d2l_gemma_gen.json" \
  --metrics_output   "$OUT/hotpotQA_niah_needles_colbert_d2l_gemma_metrics.json"

# regular LLM (gemma-2-2b-it)
sbatch slurm/run_gpu.sbatch src/standard_rag/rag_colbert_reranker.py \
  --input "$DATA" \
  --pipeline regular \
  --model_path "$GEMMA_BASE" \
  --embedding_model "$EMB" \
  --pinecone_namespace "$PINECONE_NAMESPACE" \
  --retrieved_output "$OUT/hotpotQA_niah_needles_colbert_regular_gemma.json" \
  --gen_output       "$OUT/hotpotQA_niah_needles_colbert_regular_gemma_gen.json" \
  --metrics_output   "$OUT/hotpotQA_niah_needles_colbert_regular_gemma_metrics.json"
