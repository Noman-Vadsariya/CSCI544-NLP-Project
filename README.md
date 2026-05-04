# CSCI544-NLP-Project

Doc2LoRA hypernetwork + RAG pipelines for HotpotQA, ASQA, and a synthetic
needle-in-a-haystack benchmark. Eval scripts in [scripts/](scripts/).

## Table of contents

- [Notes](#notes)
- [1. Setup](#1-setup)
  - [1a. Retrieval / evaluation](#1a-retrieval--evaluation)
  - [1b. Retrieval + Hypernetwork (full)](#1b-retrieval--hypernetwork-full)
- [2. Generate Data](#2-generate-data)
  - [Source datasets](#source-datasets)
  - [Teacher-generated QA (for hypernetwork training)](#teacher-generated-qa-for-hypernetwork-training)
  - [NIAH dataset](#niah-dataset)
- [3. Training the Hypernetwork](#3-training-the-hypernetwork)
  - [Stage 1 — adapt the doc2lora hypernet to the dataset](#stage-1--adapt-the-doc2lora-hypernet-to-the-dataset)
  - [Stage 2 — chunked-context fine-tune](#stage-2--chunked-context-fine-tune)
- [4. RAG Pipelines](#4-rag-pipelines)
  - [Top-k coverage](#top-k-coverage)
  - [End-to-end retrieval + generation](#end-to-end-retrieval--generation)
  - [Generation-only (cached retrieval)](#generation-only-cached-retrieval)
  - [Gold-context (no retrieval)](#gold-context-no-retrieval)
  - [NIAH (needle-in-a-haystack)](#niah-needle-in-a-haystack)
- [Repo layout](#repo-layout)

---

## Notes

This project used **4 RTX A6000 GPUs** for hypernetwork training and evaluation.
The **p100s** from CARC were also used for development and testing of the retrieval pipeline.

Most files in the `hypernetwork/` directory were repurposed from the [Doc2LoRA repository](https://github.com/SakanaAI/doc-to-lora). The primary files developed or modified for this project are:

- `train_stage1.py`
- `train_stage2.py`
- `inference.py`
- `measure_vram_usage.py`


Additionally, the `build_asqa` and `build_hotpotQA` scripts were adapted from the original Doc2LoRA data-building scripts to maintain compatibility with the Doc2LoRA pipeline.

---

## 1. Setup

Two install profiles. Pick the lighter one if you only want to reproduce
the RAG baselines and not run anything related with the hypernetworks.

### 1a. Retrieval / evaluation

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

Prepared datasets used by the training and eval scripts live under
`data/raw_datasets/`. Run builders from the repo root. A few builders still
have hard-coded paths from the original Doc2LoRA code, so the table
below distinguishes the checked-in dataset location from the script's current
raw output.

The HotpotQA and ASQA builders pull from these HuggingFace datasets:

- HotpotQA: [hotpotqa/hotpot_qa](https://huggingface.co/datasets/hotpotqa/hotpot_qa)
- ASQA: [din0s/asqa](https://huggingface.co/datasets/din0s/asqa)

### Source datasets

| Step | Script | Current script output | Notes |
|---|---|---|---|
| HotpotQA compact | [data/build_hotpotQA_compact.py](data/build_hotpotQA_compact.py) | `raw_datasets/hotpotQA_compact/{train,test}/ds.parquet` | Merges multiple QA pairs that share a full HotpotQA context. Requires the raw source at `raw_datasets/raw_hotpotQA`. The prepared copy used by scripts is under `data/raw_datasets/hotpotQA_compact/`. |
| HotpotQA gold-style compact | [data/build_hotpotQA_golden_compact.py](data/build_hotpotQA_golden_compact.py) | `raw_datasets/hotpotQA_compact/{train,test}/ds.parquet` | Adds a `gold_context` column containing supporting-fact paragraphs. Requires the raw source at `raw_datasets/raw_hotpotQA`. The prepared gold dataset used by eval/training is `data/raw_datasets/hotpotQA_gold_compact/`. |
| ASQA compact | [data/build_asqa_compact.py](data/build_asqa_compact.py) | `asqa_final.jsonl` | Downloads ASQA from Hugging Face and fetches Wikipedia passages per question, so it needs network access. The prepared parquet copy used by scripts is `data/raw_datasets/asqa_compact/{train,test}/ds.parquet`. |
| ASQA gold | [data/build_asqa_gold_subset.py](data/build_asqa_gold_subset.py) | `data/raw_datasets/asqa_compact/test/ds_gold_subset.parquet` | Adds `gold_context` to the ASQA eval rows. Run with `--sample_frac 1.0` to build the full checked-in dataset. |
| Combined gold dataset | [data/build_combined_compact.py](data/build_combined_compact.py) | `data/raw_datasets/golden_rag_compact/{train,test}/ds.parquet` | Combines `hotpotQA_gold_compact`, `asqa_compact`, and `prontoQA_compact`. Run the builders above or provide their prepared parquet files first. |

### Teacher-generated QA (for hypernetwork training)

Two scripts, one per teacher model, both wrapping
[data/self_generate_qa.py](data/self_generate_qa.py). Output directories are
namespaced by model under `data/raw_datasets/self_gen/`, so the runs don't
collide if you launch both.

```bash
bash scripts/gen_data_qwen.sh    # Qwen3-4B-Instruct-2507 teacher
bash scripts/gen_data_gemma.sh   # gemma-2-2b-it teacher
```

Each script submits two Slurm jobs covering the HotpotQA + ASQA compact
splits and the combined gold dataset.

### NIAH dataset

See [4 NIAH](#niah-needle-in-a-haystack) for the full pipeline. The
short version:

```bash
# Builds data/raw_datasets/hotpotQA_niah/{test.parquet,test.jsonl,metadata.json}.
# needles.json must already exist at data/raw_datasets/hotpotQA_niah/needles.json.
python data/generate_hotpotQA_niah.py \
  --data-path data/raw_datasets/hotpotQA_compact/test/ds.parquet \
  --out-dir   data/raw_datasets/hotpotQA_niah \
  --needles-path data/raw_datasets/hotpotQA_niah/needles.json \
  --n-samples 101 \
  --n-distractors 25
```

To generate the needles we used ChatGPT to create entries similar to the
original Doc2LoRA setup based on "The special magic number is {magic_number}". The checked-in `needles.json` has 101 entries and each generated sample uses one unique needle/question pair and assigns it to one of 10 depth bins. An example of the JSON file is shown below:

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

So a complete reproduction, run the end-to-end script (gives k=10 metrics and
generation for both retrievers, while neural also stores 20 retrieved passages
by default), then run the matching generation-only script (gives k=2,5,10 over
the same cached retrieval).

For NIAH there is no separate gen-only step. Both ColBERT and neural
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

**Building the dataset.** Inject generated needles into HotpotQA
distractor-padded contexts at 10 depth bins. The checked-in dataset was built
with 101 samples and 25 distractor blocks per sample:

```bash
python data/generate_hotpotQA_niah.py \
  --data-path data/raw_datasets/hotpotQA_compact/test/ds.parquet \
  --out-dir data/raw_datasets/hotpotQA_niah \
  --needles-path data/raw_datasets/hotpotQA_niah/needles.json \
  --n-samples 101 \
  --n-distractors 25
```

Requires `hotpotQA_compact/test/ds.parquet` (built in 2) and a
`needles.json` defining the needle entries. If you omit `--n-samples`, the
script defaults to a 20-sample smoke-test dataset.

**Running the eval.** The ColBERT NIAH eval uses a dedicated Pinecone namespace
(`hotpotqa_niah_needles_v1`) so it doesn't collide with the non-NIAH index.
The neural SPLADE NIAH eval is local and does not use Pinecone.

| Pipeline | Script |
|---|---|
| ColBERT reranker | [scripts/eval_colbert_rag_niah.sh](scripts/eval_colbert_rag_niah.sh) |
| Neural (SPLADE) | [scripts/eval_neural_rag_niah.sh](scripts/eval_neural_rag_niah.sh) |

The eval scripts will output json files with the retrieved and generated results and the evaluation metrics.

---

## Repo layout

```
data/                   Dataset builders plus raw_datasets/ inputs and retrieved/ outputs
requirements-base.txt   Retrieval/evaluation runtime dependencies
requirements.txt        Full hypernetwork training + generation dependencies
scripts/                Slurm submission scripts
slurm/                  Reusable sbatch wrappers (run_gpu.sbatch, run_gpu2.sbatch)
src/
  hypernetwork/         Stage-1 / stage-2 doc2lora training
  standard_rag/         ColBERT reranker + gen_from_retrieved
  neural_retrieval_rag/ SPLADE + bridge-query RAG
  evaluation/           Gold-context generators + retrieval-aware metrics
train_outputs/          Hypernet checkpoints go here
```
