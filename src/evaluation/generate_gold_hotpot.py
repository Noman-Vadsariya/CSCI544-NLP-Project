"""Generate HotpotQA predictions with gold context only (no retrieval).

This script evaluates ONLY the generation phase, assuming the gold context
is already provided. Retrieval is skipped.

Supported input formats (auto-detected):
  - Compact parquet (columns: context, prompts, responses), where context
    is already the gold context. Example:
        data/raw_datasets/hotpotQA_gold_compact/test/ds.parquet
  - Raw HotpotQA HF dataset (with _id, question, answer, supporting_facts,
    context). Gold context is reconstructed from the supporting facts.

Supported pipelines:
  - hypernet (doc2lora): internalize gold context, then query with the same
    HYPERNET_QUERY_PREFIX used in src/standard_rag/rag_colbert_reranker.py.
  - standard (baseline LLM): pass gold context as "Passages:" through
    run_baseline's prompt template.

Outputs:
  - pred.json: {"answer": {qid: pred}, "sp": {}}   HotpotQA evaluator format
  - gold.json: list of {_id, answer, [supporting_facts]}
  - outputs.jsonl: per-sample log with prediction + metrics
  - summary.json: aggregate EM / F1 / containment / latency / peak memory

Evaluate answer quality with:
    python src/evaluation/evaluate_hotpot.py pred.json gold.json --answer-only
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

RAW_HOTPOT_PATH = "data/raw_datasets/raw_hotpotQA"
COMPACT_COLUMNS = {"context", "prompts", "responses"}

HYPERNET_QUERY_PREFIX = (
    "Answer the following question based on the provided context.\n\n"
)


def build_gold_context(sample):
    """Concat gold supporting-fact sentences (matches hotpotQA_gold_compact)."""
    gold_titles = set(sample["supporting_facts"]["title"])
    sentences = []
    for title, sents in zip(sample["context"]["title"], sample["context"]["sentences"]):
        if title in gold_titles:
            sentences.extend(sents)
    return "\n".join(sentences)


def supporting_facts_to_list(sf):
    return [[t, int(i)] for t, i in zip(sf["title"], sf["sent_id"])]


# ------------------------------------------------------------------
# Metric helpers (mirror src/standard_rag/rag_colbert_reranker.py)
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


def extract_answer_span(text, gold):
    """Post-process free-form LLM output into a short answer span."""
    if not text:
        return ""

    gold_norm = normalize_answer(gold)

    first_word = re.split(r"[\s,.!?:;]+", text.strip().lower(), maxsplit=1)[0]
    if gold_norm in {"yes", "no"} and first_word in {"yes", "no"}:
        return first_word

    t = text.strip()
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
    t = t.replace("**", "").replace("*", "").replace("`", "")

    for line in t.split("\n"):
        line = re.sub(r"^[\-\*#\d\.\)]+\s*", "", line.strip())
        if line:
            t = line
            break

    m = re.match(r"^(.+?[.!?])(?:\s|$)", t)
    if m:
        t = m.group(1).rstrip(".!?")

    return t.strip()


# ------------------------------------------------------------------
# Dataset loading (format auto-detection)
# ------------------------------------------------------------------

def load_samples(path, split):
    """Return (format, dataset). Format is 'compact' or 'raw'."""
    if path.endswith(".parquet") or (os.path.isfile(path) and path.endswith(".parquet")):
        ds = load_dataset("parquet", data_files=path)["train"]
    else:
        ds = load_dataset(path, split=split)
    cols = set(ds.column_names)
    fmt = "compact" if COMPACT_COLUMNS.issubset(cols) else "raw"
    return fmt, ds


def normalize_sample(sample, fmt, index):
    """Return dict with _id, question, answer, gold_context, supporting_facts."""
    if fmt == "compact":
        return {
            "_id": str(index),
            "question": sample["prompts"][0],
            "answer": sample["responses"][0],
            "gold_context": sample["context"],
            "supporting_facts": None,
        }
    return {
        "_id": sample["_id"],
        "question": sample["question"],
        "answer": sample["answer"],
        "gold_context": build_gold_context(sample),
        "supporting_facts": supporting_facts_to_list(sample["supporting_facts"]),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["hypernet", "standard"],
        default="hypernet",
        help="hypernet: doc2lora pipeline. standard: baseline LLM. Both use gold context only.",
    )
    parser.add_argument(
        "--input",
        "--dataset",
        dest="input",
        default=RAW_HOTPOT_PATH,
        help="Parquet file (compact format) or HF dataset directory (raw format).",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="For hypernet: hypernet .bin. For standard: HF model dir (or .bin; its dir is used).",
    )
    parser.add_argument(
        "--answer_style",
        choices=["short", "full"],
        default="short",
        help="short: terse answer + span extraction (HotpotQA default). "
             "full: long-form answer, no span extraction (use for ASQA-style evaluation).",
    )
    parser.add_argument("--jsonl", type=Path, default=Path("src/evaluation/hotpot_outputs.jsonl"))
    parser.add_argument("--pred", type=Path, default=Path("src/evaluation/hotpot_pred.json"))
    parser.add_argument("--gold", type=Path, default=Path("src/evaluation/hotpot_gold.json"))
    parser.add_argument("--summary", type=Path, default=Path("src/evaluation/hotpot_summary.json"))
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

    fmt, ds = load_samples(args.input, args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"Loaded {len(ds)} samples from {args.input} (format: {fmt})")

    model, tokenizer = load_generator(args.mode, args.model_path)

    pred_answers = {}
    gold_records = []
    em_total = 0.0
    f1_total = 0.0
    contain_total = 0.0
    latencies = []
    peak_mems = []

    device = torch.cuda.current_device() if torch.cuda.is_available() else None

    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.jsonl.open("w") as f:
        for i, sample in enumerate(tqdm(ds, desc=f"Generating [{args.mode}] (gold-context)")):
            s = normalize_sample(sample, fmt, i)
            qid = s["_id"]
            question = s["question"]
            gold_answer = s["answer"]
            gold_context = s["gold_context"]

            # Match the prompt used in src/standard_rag/rag_colbert_reranker.py:
            # - doc2lora prepends HYPERNET_QUERY_PREFIX to the raw query
            # - regular/baseline passes the raw query (run_baseline wraps it)
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
                                       answer_style=args.answer_style)
            else:
                outputs = run_baseline(model, tokenizer, example, max_new_tokens=args.max_new_tokens,
                                       answer_style=args.answer_style)
            if device is not None:
                torch.cuda.synchronize(device)
            latency = time.time() - t0
            peak_mem_mb = (
                torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                if device is not None
                else 0.0
            )

            raw_pred = outputs[0].strip()
            prediction = raw_pred if args.answer_style == "full" else extract_answer_span(raw_pred, gold_answer)

            em = compute_em(prediction, gold_answer)
            f1 = compute_f1(prediction, gold_answer)
            contain = compute_containment(raw_pred, gold_answer)

            em_total += em
            f1_total += f1
            contain_total += contain
            latencies.append(latency)
            peak_mems.append(peak_mem_mb)

            pred_answers[qid] = prediction
            gold_record = {"_id": qid, "answer": gold_answer}
            if s["supporting_facts"] is not None:
                gold_record["supporting_facts"] = s["supporting_facts"]
            gold_records.append(gold_record)

            f.write(json.dumps({
                "_id": qid,
                "question": question,
                "context": gold_context,
                "gold_answer": gold_answer,
                "supporting_facts": s["supporting_facts"],
                "raw_prediction": raw_pred,
                "prediction": prediction,
                "em": em,
                "f1": f1,
                "contain": contain,
                "latency": latency,
                "peak_mem_mb": peak_mem_mb,
            }) + "\n")
            f.flush()

    n = len(pred_answers)
    summary = {
        "mode": args.mode,
        "answer_style": args.answer_style,
        "input": args.input,
        "model_path": args.model_path,
        "num_samples": n,
        "answer_em": em_total / n if n else 0.0,
        "answer_f1": f1_total / n if n else 0.0,
        "answer_contain": contain_total / n if n else 0.0,
        "avg_latency_sec": sum(latencies) / len(latencies) if latencies else 0.0,
        "avg_peak_mem_mb": sum(peak_mems) / len(peak_mems) if peak_mems else 0.0,
        "max_peak_mem_mb": max(peak_mems) if peak_mems else 0.0,
    }

    args.pred.parent.mkdir(parents=True, exist_ok=True)
    args.gold.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.pred.write_text(json.dumps({"answer": pred_answers, "sp": {}}, indent=2))
    args.gold.write_text(json.dumps(gold_records, indent=2))
    args.summary.write_text(json.dumps(summary, indent=2))

    print(f"Wrote {n} predictions to {args.pred}")
    print(f"Wrote {len(gold_records)} gold records to {args.gold}")
    print("\n===== GENERATION SUMMARY (gold context) =====")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
