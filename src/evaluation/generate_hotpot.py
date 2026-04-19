"""Generate HotpotQA predictions using either the hypernet or a baseline LLM.

Outputs:
  - pred.json: {"answer": {qid: pred}, "sp": {}}   HotpotQA evaluator format
  - gold.json: list of {_id, answer, supporting_facts}
  - outputs.jsonl: per-sample log with question/context/pred for inspection

Evaluate with:
    python src/evaluation/evaluate_hotpot.py pred.json gold.json --answer-only

Supporting-fact prediction is not produced (hypernet/baseline answer only),
so evaluation must use --answer-only.
"""
import argparse
import json
from pathlib import Path

from tqdm import tqdm
from datasets import load_dataset

from src.hypernetwork.inference import (
    load_baseline,
    load_hypernet,
    run_baseline,
    run_hypernet,
)

RAW_HOTPOT_PATH = "data/raw_datasets/raw_hotpotQA"


def build_gold_context(sample):
    """Concat gold supporting-fact sentences (matches hotpotQA_gold_compact)."""
    gold_titles = set(sample["supporting_facts"]["title"])
    sentences = []
    for title, sents in zip(sample["context"]["title"], sample["context"]["sentences"]):
        if title in gold_titles:
            sentences.extend(sents)
    return "\n".join(sentences)


def build_full_context(sample):
    """Standard HotpotQA distractor setting: all 10 paragraphs, title-prefixed."""
    parts = []
    for title, sents in zip(sample["context"]["title"], sample["context"]["sentences"]):
        parts.append(f"{title}: {''.join(sents)}")
    return "\n\n".join(parts)


def supporting_facts_to_list(sf):
    return [[t, int(i)] for t, i in zip(sf["title"], sf["sent_id"])]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["hypernet", "standard"],
        default="hypernet",
        help="hypernet: internalize gold context + ask question. "
        "standard: full 10-paragraph context in prompt (HotpotQA distractor setting).",
    )
    parser.add_argument("--dataset", default=RAW_HOTPOT_PATH)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=Path("src/evaluation/hotpot_outputs.jsonl"),
    )
    parser.add_argument(
        "--pred",
        type=Path,
        default=Path("src/evaluation/hotpot_pred.json"),
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("src/evaluation/hotpot_gold.json"),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    if args.limit:
        ds = ds.take(args.limit)

    if args.mode == "hypernet":
        model, tokenizer = load_hypernet()
        run_fn = run_hypernet
        build_ctx = build_gold_context
    else:
        model, tokenizer = load_baseline()
        run_fn = run_baseline
        build_ctx = build_full_context

    pred_answers = {}
    gold_records = []

    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.jsonl.open("w") as f:
        for sample in tqdm(ds, desc=f"Generating [{args.mode}]"):
            qid = sample["_id"]
            question = sample["question"]
            gold_answer = sample["answer"]
            sf = supporting_facts_to_list(sample["supporting_facts"])
            context = build_ctx(sample)

            example = {
                "context": context,
                "prompts": [question],
                "responses": [gold_answer],
            }
            outputs = run_fn(model, tokenizer, example, max_new_tokens=args.max_new_tokens)
            pred = outputs[0].strip()

            pred_answers[qid] = pred
            gold_records.append({
                "_id": qid,
                "answer": gold_answer,
                "supporting_facts": sf,
            })

            f.write(json.dumps({
                "_id": qid,
                "question": question,
                "context": context,
                "gold_answer": gold_answer,
                "supporting_facts": sf,
                "prediction": pred,
            }) + "\n")
            f.flush()

    args.pred.parent.mkdir(parents=True, exist_ok=True)
    args.gold.parent.mkdir(parents=True, exist_ok=True)
    args.pred.write_text(json.dumps({"answer": pred_answers, "sp": {}}, indent=2))
    args.gold.write_text(json.dumps(gold_records, indent=2))
    print(f"Wrote {len(pred_answers)} predictions to {args.pred}")
    print(f"Wrote {len(gold_records)} gold records to {args.gold}")


if __name__ == "__main__":
    main()
