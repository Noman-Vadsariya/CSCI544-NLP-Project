#!/usr/bin/env python3
"""
question_to_FOL.py

Description:
    Uses AMR to generate an FOL representation of each input
    query in the validation json file, and iteratively prints:
    - question
    - generated FOL logic representation

Dependencies:
    - First, run download_data.py to get this file created:
        hotpotqa_json/validation.json
    - Pip install the below libs
    - It may work on CPU or may require GPU...idk yet

Run this script like:
    python question_to_FOL.py hotpotqa_json/validation.json --limit 3

Authored by:
    Ayush Saha
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import amrlib
from amr_logic_converter import AmrLogicConverter


QUESTION_KEYS = ("question", "query", "text", "sentence", "utterance")


def load_records(path: str) -> List[Dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8").strip()

    # Normal JSON
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # Common dataset shapes: {"data": [...]}, {"examples": [...]}, etc.
            for key in ("data", "examples", "questions", "items", "records"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]
    except json.JSONDecodeError:
        pass

    # JSONL / concatenated JSON objects fallback
    records: List[Dict[str, Any]] = []
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        obj, end = decoder.raw_decode(text, idx)
        if isinstance(obj, dict):
            records.append(obj)
        idx = end

    return records


def extract_question(rec: Dict[str, Any]) -> str:
    for key in QUESTION_KEYS:
        val = rec.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    raise KeyError(f"No question field found. Tried: {QUESTION_KEYS}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", help="Path to validation JSON file")
    parser.add_argument("--limit", type=int, default=None, help="Print only first N examples")
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Optional AMRLib StoG model directory. If omitted, amrlib uses its default model_stog path.",
    )
    parser.add_argument(
        "--quantify",
        action="store_true",
        help="Wrap AMR instances in existential quantifiers in the output logic.",
    )
    args = parser.parse_args()

    # AMR parsing model
    stog = amrlib.load_stog_model(model_dir=args.model_dir)

    # AMR -> FOL converter
    converter = AmrLogicConverter(
        existentially_quantify_instances=args.quantify
    )

    records = load_records(args.json_path)
    if args.limit is not None:
        records = records[: args.limit]

    for i, rec in enumerate(records, start=1):
        try:
            question = extract_question(rec)
        except KeyError as e:
            print(f"\n=== Example {i} ===")
            print(f"Skipping record: {e}")
            continue

        # Parse question -> AMR
        amr_graphs = stog.parse_sents([question], add_metadata=False)
        amr = amr_graphs[0]

        # AMR -> FOL
        fol = converter.convert(amr)

        print(f"\n=== Example {i} ===")
        print("Question:")
        print(question)
        print("\nFOL logic representation:")
        print(fol)


if __name__ == "__main__":
    main()
    