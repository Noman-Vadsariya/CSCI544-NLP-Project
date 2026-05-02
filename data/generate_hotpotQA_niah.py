from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer


RNG_SEED = 42
SPECIAL_TPL = "The special magic number is {magic_number}."
PROMPT = "What is the special magic number? Reply with only the number."
SEP = "\n"

NOISE_BLOCK = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."

@dataclass
class NIAHSample:
    sample_id: int
    source_example_idx: int
    question: str
    answer: str
    context: str
    prompts: list[str]
    responses: list[str]
    needle_text: str
    needle_index: int
    depth_bin: int
    total_blocks: int
    distractor_type: str


def normalize_answer(s: str) -> str:
    if s is None:
        return ""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return " ".join(s.split())


def save_jsonl(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


def digits4() -> str:
    return f"{random.randint(0, 9999):04d}"


def chunk_text(text: str, tokenizer, chunk_size: int, overlap: int) -> List[str]:
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=512)
    if not tokens:
        return []
    out = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i : i + chunk_size]
        chunk = tokenizer.decode(chunk_tokens, skip_special_tokens=True).strip()
        if chunk:
            out.append(chunk)
    return out


def choose_position(total_blocks: int, depth_bin: int) -> int:
    """Choose an insertion index so the needle falls in the requested decile."""
    total_blocks = max(1, total_blocks)
    depth_bin = max(0, min(9, depth_bin))
    start = math.floor(total_blocks * (depth_bin / 10))
    end = math.ceil(total_blocks * ((depth_bin + 1) / 10)) - 1
    start = max(0, min(start, total_blocks - 1))
    end = max(start, min(end, total_blocks - 1))
    return random.randint(start, end)


def build_context_from_blocks(blocks: List[str], needle_text: str, depth_bin: int) -> tuple[str, int]:
    """Insert needle into a list of blocks and return the joined context + needle index."""
    total_blocks = max(1, len(blocks) + 1)
    insert_at = choose_position(total_blocks, depth_bin)

    final_blocks = []
    noise_idx = 0
    for idx in range(total_blocks):
        if idx == insert_at:
            final_blocks.append(needle_text)
        else:
            if noise_idx < len(blocks):
                final_blocks.append(blocks[noise_idx])
                noise_idx += 1
            else:
                # If the source blocks are exhausted, pad with the paper's fixed noise block.
                final_blocks.append(NOISE_BLOCK)

    return SEP.join(final_blocks), insert_at


def make_hotpotqa_distractor_pool(contexts: list[str], tokenizer, chunk_size: int, overlap: int, max_chunks_per_example: int = 4) -> list[str]:
    pool: list[str] = []
    for ctx in contexts:
        chunks = chunk_text(ctx, tokenizer, chunk_size=chunk_size, overlap=overlap)
        pool.extend(chunks[:max_chunks_per_example])
    return pool


def sample_hotpotqa_blocks(
    rng: random.Random,
    pool: list[str],
    source_answer: str,
    n_distractors: int,
) -> list[str]:
    """Sample distractor chunks and avoid accidental answer leakage."""
    out: list[str] = []
    ans_norm = normalize_answer(source_answer)

    attempts = 0
    max_attempts = max(50, n_distractors * 50)
    while len(out) < n_distractors and attempts < max_attempts:
        attempts += 1
        cand = rng.choice(pool)
        if ans_norm and ans_norm in normalize_answer(cand):
            continue
        out.append(cand)

    if len(out) < n_distractors:
        # Fallback to the paper's fixed noise block if the pool is too small.
        out.extend([NOISE_BLOCK] * (n_distractors - len(out)))

    return out


def build_examples_from_hotpotqa(
    questions: list[str],
    answers: list[str],
    context_pool: list[str],
    n_samples: int,
    n_distractors: int,
    seed: int,
    use_fixed_noise_only: bool,
) -> list[NIAHSample]:
    rng = random.Random(seed)
    candidate_indices = [i for i, a in enumerate(answers) if normalize_answer(a)]
    rng.shuffle(candidate_indices)

    samples: list[NIAHSample] = []
    used = 0

    for ex_idx in candidate_indices:
        if used >= n_samples:
            break

        question = questions[ex_idx]
        answer = answers[ex_idx]
        if not normalize_answer(answer):
            continue

        magic = digits4()
        needle_text = SPECIAL_TPL.format(magic_number=magic)

        # Depth is always assigned across 10 bins, as in the reference generator.
        depth_bin = used % 10

        if use_fixed_noise_only:
            # Exact paper-style distractor blocks.
            blocks = [NOISE_BLOCK for _ in range(n_distractors)]
        else:
            # HotpotQA-adapted distractors.
            blocks = sample_hotpotqa_blocks(rng, context_pool, answer, n_distractors=n_distractors)

        context, needle_index = build_context_from_blocks(blocks, needle_text, depth_bin)

        samples.append(
            NIAHSample(
                sample_id=used,
                source_example_idx=ex_idx,
                question=question,
                answer=answer,
                context=context,
                prompts=[PROMPT],
                responses=[magic],
                needle_text=needle_text,
                needle_index=needle_index,
                depth_bin=depth_bin,
                total_blocks=len(blocks) + 1,
                distractor_type=("fixed_noise" if use_fixed_noise_only else "hotpotqa_chunks"),
            )
        )
        used += 1

    return samples


def load_hotpotqa_examples(data_path: str):
    ds = load_dataset("parquet", data_files=data_path)["train"]
    questions = [x[0] for x in ds["prompts"]]
    answers = [x[0] for x in ds["responses"]]
    contexts = ds["context"]
    return questions, answers, contexts


def split_train_val_test(rows: list[NIAHSample], train_ratio: float, val_ratio: float, seed: int):
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)

    n = len(rows)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(n_train, n)
    n_val = min(n_val, max(0, n - n_train))
    n_test = max(0, n - n_train - n_val)

    train = rows[:n_train]
    val = rows[n_train : n_train + n_val]
    test = rows[n_train + n_val : n_train + n_val + n_test]
    return train, val, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a HotpotQA-based synthetic NIAH dataset.")
    parser.add_argument("--data-path", type=str, default="raw_datasets/hotpotQA_compact/test/ds.parquet")
    parser.add_argument("--out-dir", type=str, default="raw_datasets/hotpotQA_niah")
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--tokenizer-name", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--chunk-size", type=int, default=40, help="Heuristic tokens per block, matching the repo's TOKENS_PER_BLOCK.")
    parser.add_argument("--chunk-overlap", type=int, default=8)
    parser.add_argument("--n-samples", type=int, default=20, help="Small test size first.")
    parser.add_argument("--n-distractors", type=int, default=25, help="Number of distractor blocks per sample.")
    parser.add_argument("--max-chunks-per-example", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.0)
    parser.add_argument("--val-ratio", type=float, default=0.5)
    parser.add_argument("--fixed-noise-only", action="store_true", help="Use the paper's exact fixed noise block instead of HotpotQA chunks.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    questions, answers, contexts = load_hotpotqa_examples(args.data_path)

    # Build a distractor pool from HotpotQA chunks.
    context_pool = make_hotpotqa_distractor_pool(
        contexts,
        tokenizer,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
        max_chunks_per_example=args.max_chunks_per_example,
    )

    if not context_pool:
        raise RuntimeError("HotpotQA distractor pool is empty.")

    samples = build_examples_from_hotpotqa(
        questions=questions,
        answers=answers,
        context_pool=context_pool,
        n_samples=args.n_samples,
        n_distractors=args.n_distractors,
        seed=args.seed,
        use_fixed_noise_only=args.fixed_noise_only,
    )

    if not samples:
        raise RuntimeError("No samples were generated. Check the input dataset.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(json.dumps(asdict(samples[0]), indent=2, ensure_ascii=False))
        print(f"Generated {len(samples)} samples total")
        return

    save_jsonl((asdict(x) for x in samples), out_dir / "test.jsonl")

    meta = {
        "seed": args.seed,
        "data_path": args.data_path,
        "tokenizer_name": args.tokenizer_name,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "n_samples": args.n_samples,
        "n_distractors": args.n_distractors,
        "max_chunks_per_example": args.max_chunks_per_example,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "fixed_noise_only": args.fixed_noise_only,
        "note": (
            "Needle / prompt / 4-digit answer follow the Doc-to-LoRA NIAH recipe; "
            "distractors are HotpotQA chunks unless --fixed-noise-only is set."
        ),
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(samples)} samples total at {out_dir}")

    # Replace with your actual file paths
    jsonl_path = "raw_datasets/hotpotQA_niah/test.jsonl"
    parquet_path = "raw_datasets/hotpotQA_niah/test.parquet"

    # Read JSONL
    df = pd.read_json(jsonl_path, lines=True)

    # Save as Parquet
    df.to_parquet(parquet_path)
    # print(f"train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    main()
