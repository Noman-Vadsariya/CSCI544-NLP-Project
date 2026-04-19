#!/usr/bin/env python3
"""
evaluate_hotpotqa.py

Standalone HotpotQA evaluator.

Expected prediction format:
{
  "answer": {
    "<qid>": "<predicted answer>"
  },
  "sp": {
    "<qid>": [["<title>", <sent_id>], ...]
  }
}

Expected gold format:
A HotpotQA dev/train JSON file where each example contains at least:
- "_id"
- "answer"
- "supporting_facts"

Usage:
    python evaluate_hotpotqa.py pred.json gold.json

Optional:
    python evaluate_hotpotqa.py pred.json gold.json --answer-only
"""

import argparse
import json
import re
import string
from collections import Counter
from typing import Dict, List, Tuple, Any


def normalize_answer(s: str) -> str:
    """
    Official-style normalization:
    - lowercase
    - remove punctuation
    - remove articles
    - fix whitespace
    """
    def lower(text: str) -> str:
        return text.lower()

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))


def f1_score(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    """
    Token-level F1/precision/recall for answers.
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    pred_tokens = normalized_prediction.split()
    gold_tokens = normalized_ground_truth.split()

    # Handle empty answers safely
    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0, 0.0, 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0, 0.0, 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def sp_score(
    prediction: List[List[Any]],
    ground_truth: List[List[Any]],
) -> Tuple[float, float, float, float]:
    """
    Supporting-fact EM/F1/precision/recall.

    Each supporting fact is [title, sent_id].
    Comparison is done as a set of tuples.
    """
    pred_set = set()
    gold_set = set()

    for item in prediction:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            pred_set.add((str(item[0]), int(item[1])))

    for item in ground_truth:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            gold_set.add((str(item[0]), int(item[1])))

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    em = float(pred_set == gold_set)

    return em, f1, precision, recall


def evaluate(prediction: Dict[str, Any], gold_data: List[Dict[str, Any]], answer_only: bool = False) -> Dict[str, float]:
    """
    Evaluate predictions against HotpotQA gold examples.
    """
    pred_answers = prediction.get("answer", {})
    pred_sp = prediction.get("sp", {})

    metrics = {
        "answer_em": 0.0,
        "answer_f1": 0.0,
        "answer_precision": 0.0,
        "answer_recall": 0.0,
        "sp_em": 0.0,
        "sp_f1": 0.0,
        "sp_precision": 0.0,
        "sp_recall": 0.0,
        "joint_em": 0.0,
        "joint_f1": 0.0,
        "joint_precision": 0.0,
        "joint_recall": 0.0,
    }

    total = 0
    missing_answer = 0
    missing_sp = 0

    for ex in gold_data:
        qid = ex["_id"]
        gold_answer = ex["answer"]
        gold_supporting_facts = ex["supporting_facts"]

        total += 1

        if qid not in pred_answers:
            missing_answer += 1
            pred_answer = ""
        else:
            pred_answer = pred_answers[qid]

        ans_em = exact_match_score(pred_answer, gold_answer)
        ans_f1, ans_prec, ans_rec = f1_score(pred_answer, gold_answer)

        metrics["answer_em"] += ans_em
        metrics["answer_f1"] += ans_f1
        metrics["answer_precision"] += ans_prec
        metrics["answer_recall"] += ans_rec

        if not answer_only:
            if qid not in pred_sp:
                missing_sp += 1
                pred_supporting_facts = []
            else:
                pred_supporting_facts = pred_sp[qid]

            sp_em, sp_f1, sp_prec, sp_rec = sp_score(pred_supporting_facts, gold_supporting_facts)

            metrics["sp_em"] += sp_em
            metrics["sp_f1"] += sp_f1
            metrics["sp_precision"] += sp_prec
            metrics["sp_recall"] += sp_rec

            # Official HotpotQA-style joint metric:
            # joint precision = answer_precision * sp_precision
            # joint recall    = answer_recall * sp_recall
            # joint EM        = answer_em * sp_em
            joint_prec = ans_prec * sp_prec
            joint_rec = ans_rec * sp_rec
            joint_em = ans_em * sp_em
            joint_f1 = (
                2 * joint_prec * joint_rec / (joint_prec + joint_rec)
                if (joint_prec + joint_rec) > 0
                else 0.0
            )

            metrics["joint_em"] += joint_em
            metrics["joint_f1"] += joint_f1
            metrics["joint_precision"] += joint_prec
            metrics["joint_recall"] += joint_rec

    if total == 0:
        raise ValueError("Gold file is empty or invalid.")

    for key in metrics:
        metrics[key] /= total

    metrics["count"] = total
    metrics["missing_answer"] = missing_answer
    if not answer_only:
        metrics["missing_sp"] = missing_sp

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate HotpotQA predictions.")
    parser.add_argument("prediction_file", type=str, help="Path to prediction JSON")
    parser.add_argument("gold_file", type=str, help="Path to HotpotQA gold JSON")
    parser.add_argument(
        "--answer-only",
        action="store_true",
        help="Only compute answer metrics, ignore supporting facts and joint metrics",
    )
    args = parser.parse_args()

    with open(args.prediction_file, "r", encoding="utf-8") as f:
        prediction = json.load(f)

    with open(args.gold_file, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    metrics = evaluate(prediction, gold_data, answer_only=args.answer_only)

    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()