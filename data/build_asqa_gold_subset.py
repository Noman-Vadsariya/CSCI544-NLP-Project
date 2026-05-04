#!/usr/bin/env python3
"""
build_asqa_gold_subset.py

Takes an existing ASQA compact parquet (context / prompts / responses),
samples 20 % of rows, and adds a `gold_context` column containing the
most answer-relevant passage(s) extracted from the context field.

Gold context extraction:
  1. Split context into paragraphs.
  2. Score each paragraph by keyword overlap with both the question and
     the answer (stop-words removed).
  3. Take the top-scoring paragraph(s) whose combined length covers at
     least MIN_GOLD_CHARS characters, capped at MAX_GOLD_PARAGRAPHS.

Usage:
    python data/build_asqa_gold_subset.py \
        --input  data/raw_datasets/asqa_compact/test/ds.parquet \
        --output data/raw_datasets/asqa_compact/test/ds_gold_subset.parquet
"""

import argparse
import random
import re
import string
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from datasets import load_dataset

# --------------------------------------------------------------------------
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "on", "at", "by", "for", "with", "about",
    "against", "between", "through", "during", "before", "after", "above",
    "below", "from", "up", "down", "out", "off", "over", "under", "into",
    "and", "but", "or", "nor", "so", "yet", "both", "either", "neither",
    "not", "no", "nor", "this", "that", "these", "those", "what", "which",
    "who", "whom", "whose", "when", "where", "why", "how", "i", "you",
    "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
}

MIN_GOLD_CHARS = 200
MAX_GOLD_PARAGRAPHS = 3
# --------------------------------------------------------------------------


def keywords(text: str) -> List[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return [w for w in text.split() if w and w not in STOPWORDS]


def score_paragraph(para: str, question_kws: List[str], answer_kws: List[str]) -> float:
    para_lower = para.lower()
    q_hits = sum(1 for w in question_kws if w in para_lower)
    a_hits = sum(1 for w in answer_kws if w in para_lower)
    # Weight answer keywords more heavily — they anchor the gold content
    return q_hits + 2.0 * a_hits


def extract_gold_context(
    context: str,
    question: str,
    answer: str,
    min_chars: int = MIN_GOLD_CHARS,
    max_paragraphs: int = MAX_GOLD_PARAGRAPHS,
) -> str:
    # Split on blank lines first; fall back to single newlines if too few blocks.
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", context) if p.strip()]
    if len(paragraphs) < 3:
        paragraphs = [p.strip() for p in context.split("\n") if p.strip()]

    if not paragraphs:
        return context

    q_kws = keywords(question)
    a_kws = keywords(answer)

    scored: List[Tuple[float, int, str]] = []
    for idx, para in enumerate(paragraphs):
        sc = score_paragraph(para, q_kws, a_kws)
        scored.append((sc, idx, para))

    # Sort by score descending, preserve original order among ties via idx.
    scored.sort(key=lambda x: (-x[0], x[1]))

    selected_paras = []
    total_chars = 0
    for sc, idx, para in scored:
        if len(selected_paras) >= max_paragraphs and total_chars >= min_chars:
            break
        selected_paras.append((idx, para))
        total_chars += len(para)

    # Re-sort selected paragraphs into original document order.
    selected_paras.sort(key=lambda x: x[0])
    gold = "\n\n".join(p for _, p in selected_paras)

    # Fallback: if nothing scored at all, return first paragraph(s).
    if not gold.strip():
        gold = "\n\n".join(paragraphs[:max_paragraphs])

    return gold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw_datasets/asqa_compact/test/ds.parquet",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw_datasets/asqa_compact/test/ds_gold_subset.parquet",
    )
    parser.add_argument("--sample_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading: {args.input}")
    ds = load_dataset("parquet", data_files=args.input)["train"]
    print(f"Total rows: {len(ds)}")

    random.seed(args.seed)
    n_sample = max(1, int(len(ds) * args.sample_frac))
    indices = sorted(random.sample(range(len(ds)), n_sample))
    ds_subset = ds.select(indices)
    print(f"Sampled {len(ds_subset)} rows ({args.sample_frac*100:.0f}%)")

    records = []
    for row in ds_subset:
        context = row["context"]
        question = row["prompts"][0] if isinstance(row["prompts"], list) else row["prompts"]
        answer = row["responses"][0] if isinstance(row["responses"], list) else row["responses"]

        gold_context = extract_gold_context(context, question, answer)

        records.append({
            "context": context,
            "prompts": row["prompts"],
            "responses": row["responses"],
            "gold_context": gold_context,
        })

    df = pd.DataFrame(records)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")

    # Spot-check a few examples
    print("\n--- Spot check (3 examples) ---")
    for i in range(min(3, len(records))):
        r = records[i]
        q = r["prompts"][0] if isinstance(r["prompts"], list) else r["prompts"]
        a = r["responses"][0] if isinstance(r["responses"], list) else r["responses"]
        gc = r["gold_context"]
        print(f"\nQ: {q}")
        print(f"A: {a[:150]}")
        print(f"Gold context ({len(gc)} chars): {gc[:300]}...")


if __name__ == "__main__":
    main()
