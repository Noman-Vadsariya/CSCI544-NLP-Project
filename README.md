# CSCI544-NLP-Project

Doc2LoRA hypernetwork + RAG pipelines for HotpotQA, ASQA, and a synthetic
needle-in-a-haystack benchmark. Eval scripts in [scripts/](scripts/).
 
 ---

## Notes

This project used **4 RTX A6000 GPUs** for hypernetwork training and evaluation.

Most files in the `hypernetwork/` directory were repurposed from the [Doc2LoRA repository](https://github.com/SakanaAI/doc-to-lora). The primary files developed or modified for this project are:

- `train_stage1.py`
- `train_stage2.py`
- `inference.py`
- `measure_vram_usage.py`


Additionally, the `build_asqa` and `build_hotpotQA` scripts were adapted from the original Doc2LoRA data-building scripts to maintain compatibility with the Doc2LoRA pipeline.

---

## 1. Setup

Two install profiles. Pick the lighter one if you only want to reproduce
the RAG baselines and not retrain the hypernetwork.

### 1a. Retrieval-only (RAG baselines)

Everything you need to run [scripts/eval_colbert_*.sh](scripts/) and
[scripts/eval_neural_*.sh](scripts/) without retraining.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-base.txt
pip install -e .
```

### 1b. Retrieval + Hypernetwork (full)

Required for [scripts/train_*.sh](scripts/) and any
`--pipeline doc2lora` job.

Prerequisites: Python 3.10, [uv](https://docs.astral.sh/uv/),
CUDA 12.4 GPU.

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt
uv pip install -e .
```

`flash-attn` in `requirements.txt` is optional — drop it if your GPU
doesn't support it. [install.sh](install.sh) shows the full uv-based path
including the flash-attn / flashinfer wheels we used on the lab cluster.


---

## 2. Generate Data

All builders write under `data/raw_datasets/`. Run them from the repo
root. Order matters only where noted (NIAH and combined depend on
upstream builders).

### Source datasets

| Step | Script | Output | Notes |
|---|---|---|---|
| HotpotQA (compact) | [data/build_hotpotQA_compact.py](data/build_hotpotQA_compact.py) | `data/raw_datasets/hotpotQA_compact/{train,test}/ds.parquet` | Merges multiple QA pairs that share a context. |
| HotpotQA (gold) | [data/build_hotpotQA_golden_compact.py](data/build_hotpotQA_golden_compact.py) | `data/raw_datasets/hotpotQA_gold_compact/{train,test}/ds.parquet` | Keeps only supporting-fact paragraphs. Used for the gold-context eval. |
| ASQA (compact) | [data/build_asqa_compact.py](data/build_asqa_compact.py) | `data/raw_datasets/asqa_compact/...` | Fetches Wikipedia passages per question. |
| ASQA (gold) | [data/build_asqa_gold_subset.py](data/build_asqa_gold_subset.py) | `data/raw_datasets/asqa_compact/test/ds_gold_subset.parquet` | Extracts the gold context column used by the ASQA evaluation. |
| Combined gold (Used for final checkpoint results)| [data/build_combined_compact.py](data/build_combined_compact.py) | `data/raw_datasets/golden_rag_compact/{train,test}/ds.parquet` | HotpotQA gold + ASQA gold. Run the three above first. |

### Teacher-generated QA (for hypernetwork training)

Two scripts, one per teacher model, both wrapping
[data/self_generate_qa.py](data/self_generate_qa.py). Output directories
are namespaced by model, so the runs don't collide if you launch both.

```bash
bash scripts/gen_data_qwen.sh    # Qwen3-4B-Instruct-2507 teacher
bash scripts/gen_data_gemma.sh   # gemma-2-2b-it teacher
```

Each script submits two Slurm jobs covering the HotpotQA + ASQA compact
splits and the combined gold dataset.

### NIAH dataset

See [§4 NIAH](#niah-needle-in-a-haystack) for the full pipeline. The
short version:

```bash
# Builds data/raw_datasets/hotpotQA_niah/{test.parquet,test.jsonl,metadata.json}.
# needles.json must already exist at data/raw_datasets/hotpotQA_niah/needles.json.
python data/generate_hotpotQA_niah.py \
  --data-path data/raw_datasets/hotpotQA_compact/test/ds.parquet \
  --out-dir   data/raw_datasets/hotpotQA_niah \
  --needles-path data/raw_datasets/hotpotQA_niah/needles.json
```

To generate the needles we used ChatGPT to generate similar needles to the original setup from the Doc2LoRA work based on "The special magic number is {magic_number}". We generate 100 needles and evaluate the final RAG pipeline on finding and generating the correct response for these needles. An example of the json file is shown below:

```json
{
  "data": [
    {
      "needle": "The secret passphrase is secret_passphrase.",
      "question": "What is the secret passphrase? Reply with only the passphrase."
    },
    {
      "needle": "The hidden keyword is hidden_keyword.",
      "question": "What is the hidden keyword? Reply with only the keyword."
    }
  ]
}
```

---

## 3. Training the Hypernetwork

All training scripts assume Slurm and submit through `slurm/run_gpu*.sbatch`.
Outputs land in `train_outputs/`.

### Stage 1 — adapt the doc2lora hypernet to the dataset

Two starting points:

```bash
# Resume from an existing doc2lora checkpoint
bash scripts/train_stage1.sh

# Train from scratch on hotpotQA_gold_compact
bash scripts/train_stage1_scratch.sh
```

[scripts/train_stage1.sh](scripts/train_stage1.sh) writes to
`train_outputs/stage1_combined_gold_dataset_finetune/`.

### Stage 2 — chunked-context fine-tune

Picks up the stage-1 checkpoint and trains with random context chunking:

```bash
bash scripts/train_stage2.sh
```

Writes to `train_outputs/stage2_combined_gold_dataset_finetune/`. The
resulting `pytorch_model.bin` is the hypernet checkpoint that the
`--pipeline doc2lora` evals load.

---

## 4. RAG Pipelines

Every eval script submits two Slurm jobs per dataset: one for the
`doc2lora` Gemma hypernet and one for the regular `gemma-2-2b-it`
baseline. Outputs (retrieval JSONs, generation JSONs, metric summaries)
land in `data/retrieved/`. Metrics computed: EM, F1, ROUGE-L, and answer
containment. Only F1 and ROUGE-L are reported in the final results/report.

### Top-k coverage

Where each top-k value comes from:

| Top-k | How it's produced (HotpotQA / ASQA) |
|---|---|
| k = 2, 5, 10 | The generation-only scripts ([eval_colbert_gen.sh](scripts/eval_colbert_gen.sh), [eval_neural_gen.sh](scripts/eval_neural_gen.sh)) sweep all three values against cached retrieval. Default sweep is `TOP_KS = (2, 5, 10)` in [src/standard_rag/gen_from_retrieved.py](src/standard_rag/gen_from_retrieved.py); override with `--top_ks 2,5,10`. |
| k = 10 (ColBERT) | Baked into [src/standard_rag/rag_colbert_reranker.py](src/standard_rag/rag_colbert_reranker.py) (`K_VALUES = (10,)`); produced by [eval_colbert_rag.sh](scripts/eval_colbert_rag.sh) during the end-to-end run. |
| k = 10 (neural) | Baked into [src/neural_retrieval_rag/neural_rag.py](src/neural_retrieval_rag/neural_rag.py) (`K_VALUES = (10,)`); produced by [eval_neural_rag.sh](scripts/eval_neural_rag.sh) during the end-to-end run. |

So a complete reproduction = run the end-to-end script (gives k=10 for
ColBERT or k=20 for neural and caches the retrieval), then run the
matching generation-only script (gives k=2,5,10 over the same cached
retrieval).

For NIAH there is no separate gen-only step — both ColBERT and neural
NIAH scripts produce a single k per run, taken from the `K_VALUES`
constant in the corresponding source file:

| Pipeline | Source defining `K_VALUES` | NIAH script |
|---|---|---|
| ColBERT reranker | [src/standard_rag/rag_colbert_reranker.py](src/standard_rag/rag_colbert_reranker.py) | [eval_colbert_rag_niah.sh](scripts/eval_colbert_rag_niah.sh) |
| Neural (SPLADE) | [src/neural_retrieval_rag/neural_rag.py](src/neural_retrieval_rag/neural_rag.py) | [eval_neural_rag_niah.sh](scripts/eval_neural_rag_niah.sh) |

To report both k=10 and k=20 for NIAH, run each script twice — once
with `K_VALUES = (10,)` and once with `K_VALUES = (20,)`. Be sure to manually
edit the constant between runs and renaming the output JSONs so they
don't overwrite each other. There's no `--top_ks` flag plumbed through
the end-to-end RAG scripts. Make sure the rerank cutoff at the top of
`rag_colbert_reranker.py` stays ≥ `max(K_VALUES)` when bumping k.

### End-to-end retrieval + generation
This is what we used to run and generate the main results in the report. Be sure to run the generation for k=2 and 5 since by default k is 10 in the full pipeline.

| Dataset | ColBERT reranker | Neural (SPLADE) |
|---|---|---|
| HotpotQA compact + ASQA gold | [scripts/eval_colbert_rag.sh](scripts/eval_colbert_rag.sh) | [scripts/eval_neural_rag.sh](scripts/eval_neural_rag.sh) |

### Generation-only (cached retrieval)

Re-uses retrieval JSONs already in `data/retrieved/` so you can swap
generators without re-indexing.

| Retriever | Script |
|---|---|
| ColBERT | [scripts/eval_colbert_gen.sh](scripts/eval_colbert_gen.sh) |
| Neural (SPLADE) | [scripts/eval_neural_gen.sh](scripts/eval_neural_gen.sh) |

### Gold-context (no retrieval)

Upper-bound numbers using the dataset-provided gold context. These are just a sanity check and results are not reported in the final report:

```bash
bash scripts/eval_gold_hotpot.sh    # HotpotQA gold compact
bash scripts/eval_gold_asqa.sh   # ASQA gold
```

### NIAH (needle-in-a-haystack)

A separate evaluation track because the dataset and the Pinecone
indexing namespace are distinct from the standard HotpotQA / ASQA runs.

**Building the dataset.** Inject HotpotQA needles into
distractor-padded contexts at 10 depth bins:

```bash
python data/generate_hotpotQA_niah.py \
  --data-path data/raw_datasets/hotpotQA_compact/test/ds.parquet \
  --out-dir data/raw_datasets/hotpotQA_niah \
  --needles-path data/raw_datasets/hotpotQA_niah/needles.json \
  --n-distractors 32
```

Requires `hotpotQA_compact/test/ds.parquet` (built in §2) and a
`needles.json` defining the needle entries.

**Running the eval.** All NIAH eval jobs use a dedicated Pinecone
namespace (`hotpotqa_niah_needles_v1`) so they don't collide with the
non-NIAH index.

| Pipeline | Script |
|---|---|
| ColBERT reranker | [scripts/eval_colbert_rag_niah.sh](scripts/eval_colbert_rag_niah.sh) |
| Neural (SPLADE) | [scripts/eval_neural_rag_niah.sh](scripts/eval_neural_rag_niah.sh) |

NIAH metrics (underscore-aware normalization and retrieval-aware refusal
credit) are computed at generation time, so no separate rescore step is
required.

---

## Repo layout

```
data/                   Dataset builders + raw_datasets/, retrieved/ outputs
scripts/                Slurm submission scripts (see scripts/README.md)
slurm/                  Reusable sbatch wrappers (run_gpu.sbatch, run_gpu2.sbatch)
src/
  hypernetwork/         Stage-1 / stage-2 doc2lora training
  standard_rag/         ColBERT reranker + gen_from_retrieved
  neural_retrieval_rag/ SPLADE + bridge-query RAG
  evaluation/           Gold-context generators + retrieval-aware metrics
train_outputs/          Hypernet checkpoints land here
```
