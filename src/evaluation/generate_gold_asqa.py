"""Generate ASQA predictions with gold context only (no retrieval).

Evaluates ONLY the generation phase using the pre-extracted gold context
already present in the dataset (column: gold_context).

Input format: compact parquet with columns context, prompts, responses,
gold_context (produced by data/build_asqa_gold_subset.py).
    data/raw_datasets/asqa_compact/test/ds_gold_subset.parquet

Supported pipelines:
  - hypernet (doc2lora): internalize gold context, then query with
    HYPERNET_QUERY_PREFIX (same as rag_colbert_reranker.py).
  - standard (baseline LLM): pass gold context as "Passages:" via run_baseline.

Outputs:
  - pred.json:    {qid: prediction}
  - gold.json:    list of {_id, answer}
  - outputs.jsonl: per-sample log with prediction + metrics
  - summary.json:  aggregate EM / F1 / ROUGE-1 / ROUGE-L / containment / latency
"""
import argparse
import json
import os
import re
import string
import time
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm
from datasets import load_dataset

from src.hypernetwork.inference import (
    load_baseline,
    load_hypernet,
    run_baseline,
    run_hypernet,
)

DEFAULT_DATA = "data/raw_datasets/asqa_compact/test/ds_gold_subset.parquet"

# Used for doc2lora (hypernet) only — matches rag_colbert_reranker.py / neural_rag.py.
# For the standard (baseline) pipeline, run_baseline is called with answer_style="full"
# which uses the long-form ASQA prompt template defined in src/hypernetwork/inference.py.
HYPERNET_QUERY_PREFIX = (
    "Answer the question in as few words as possible. "
    "Only output the answer itself, no explanation or extra text.\n\n"
)


# ------------------------------------------------------------------
# Metric helpers
# ------------------------------------------------------------------

def normalize_answer(s):
    exclude = set(string.punctuation)
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def compute_em(prediction, ground_truth):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_containment(prediction, ground_truth):
    gold_norm = normalize_answer(ground_truth)
    if not gold_norm:
        return 0.0
    return float(gold_norm in normalize_answer(prediction))


def compute_rouge(prediction, ground_truth):
    """Unigram ROUGE-1 and ROUGE-L (LCS-based), both F1."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0

    # ROUGE-1
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    p1 = num_same / len(pred_tokens)
    r1 = num_same / len(gold_tokens)
    rouge1 = (2 * p1 * r1 / (p1 + r1)) if (p1 + r1) > 0 else 0.0

    # ROUGE-L (LCS length via DP)
    m, n = len(gold_tokens), len(pred_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gold_tokens[i - 1] == pred_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    pl = lcs / len(pred_tokens)
    rl = lcs / len(gold_tokens)
    rougeL = (2 * pl * rl / (pl + rl)) if (pl + rl) > 0 else 0.0

    return rouge1, rougeL


# ------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["hypernet", "standard"],
        default="hypernet",
        help="hypernet: doc2lora pipeline. standard: baseline LLM.",
    )
    parser.add_argument(
        "--input",
        dest="input",
        default=DEFAULT_DATA,
        help="Parquet file with context/prompts/responses/gold_context columns.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens to generate (ASQA answers are long-form).")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="For hypernet: hypernet .bin. For standard: HF model dir (or .bin; its dir is used).",
    )
    parser.add_argument("--jsonl", type=Path, default=Path("src/evaluation/asqa_outputs.jsonl"))
    parser.add_argument("--pred", type=Path, default=Path("src/evaluation/asqa_pred.json"))
    parser.add_argument("--gold", type=Path, default=Path("src/evaluation/asqa_gold.json"))
    parser.add_argument("--summary", type=Path, default=Path("src/evaluation/asqa_summary.json"))
    return parser.parse_args()


def load_generator(mode, model_path):
    if mode == "hypernet":
        return load_hypernet(model_path) if model_path else load_hypernet()
    resolved = model_path
    if resolved and os.path.isfile(resolved):
        resolved = os.path.dirname(resolved)
    return load_baseline(resolved) if resolved else load_baseline()


def main():
    args = parse_args()

    ds = load_dataset("parquet", data_files=args.input)[args.split]
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"Loaded {len(ds)} samples from {args.input}")

    if "gold_context" not in ds.column_names:
        raise ValueError(
            "Dataset is missing 'gold_context' column. "
            "Run data/build_asqa_gold_subset.py first."
        )

    model, tokenizer = load_generator(args.mode, args.model_path)

    pred_answers = {}
    gold_records = []
    em_total = f1_total = contain_total = rouge1_total = rougeL_total = 0.0
    latencies = []
    peak_mems = []

    device = torch.cuda.current_device() if torch.cuda.is_available() else None

    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.jsonl.open("w") as fout:
        for i, sample in enumerate(tqdm(ds, desc=f"Generating [{args.mode}] (gold-context)")):
            qid = str(i)
            question = sample["prompts"][0] if isinstance(sample["prompts"], list) else sample["prompts"]
            gold_answer = sample["responses"][0] if isinstance(sample["responses"], list) else sample["responses"]
            gold_context = sample["gold_context"]

            example = {
                "context": gold_context,
                "prompts": [question],
                "responses": [gold_answer],
            }

            if device is not None:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)

            t0 = time.time()
            if args.mode == "hypernet":
                outputs = run_hypernet(model, tokenizer, example, max_new_tokens=args.max_new_tokens,
                                       answer_style="full")
            else:
                outputs = run_baseline(model, tokenizer, example, max_new_tokens=args.max_new_tokens,
                                       answer_style="full")
            if device is not None:
                torch.cuda.synchronize(device)
            latency = time.time() - t0
            peak_mem_mb = (
                torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                if device is not None else 0.0
            )

            prediction = outputs[0].strip()

            em = compute_em(prediction, gold_answer)
            f1 = compute_f1(prediction, gold_answer)
            contain = compute_containment(prediction, gold_answer)
            rouge1, rougeL = compute_rouge(prediction, gold_answer)

            em_total += em
            f1_total += f1
            contain_total += contain
            rouge1_total += rouge1
            rougeL_total += rougeL
            latencies.append(latency)
            peak_mems.append(peak_mem_mb)

            pred_answers[qid] = prediction
            gold_records.append({"_id": qid, "answer": gold_answer})

            fout.write(json.dumps({
                "_id": qid,
                "question": question,
                "gold_context": gold_context,
                "gold_answer": gold_answer,
                "prediction": prediction,
                "em": em,
                "f1": f1,
                "contain": contain,
                "rouge1": rouge1,
                "rougeL": rougeL,
                "latency": latency,
                "peak_mem_mb": peak_mem_mb,
            }) + "\n")
            fout.flush()

    n = len(pred_answers)
    summary = {
        "mode": args.mode,
        "input": args.input,
        "model_path": args.model_path,
        "num_samples": n,
        "answer_em":       em_total      / n if n else 0.0,
        "answer_f1":       f1_total      / n if n else 0.0,
        "answer_contain":  contain_total / n if n else 0.0,
        "rouge1":          rouge1_total  / n if n else 0.0,
        "rougeL":          rougeL_total  / n if n else 0.0,
        "avg_latency_sec":  sum(latencies) / len(latencies) if latencies else 0.0,
        "avg_peak_mem_mb":  sum(peak_mems) / len(peak_mems) if peak_mems else 0.0,
        "max_peak_mem_mb":  max(peak_mems) if peak_mems else 0.0,
    }

    for p in (args.pred, args.gold, args.summary):
        p.parent.mkdir(parents=True, exist_ok=True)

    args.pred.write_text(json.dumps(pred_answers, indent=2))
    args.gold.write_text(json.dumps(gold_records, indent=2))
    args.summary.write_text(json.dumps(summary, indent=2))

    print(f"Wrote {n} predictions to {args.pred}")
    print(f"Wrote {len(gold_records)} gold records to {args.gold}")
    print("\n===== GENERATION SUMMARY (gold context) =====")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
