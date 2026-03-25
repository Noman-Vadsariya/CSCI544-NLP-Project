#!/usr/bin/env python3
"""
Download HotpotQA dataset (train and val splits), and
save them into hotpotqa_json/train.json
                hotpotqa_json/validation.json
"""

import argparse
import json
from pathlib import Path
from datasets import load_dataset


def save_split_to_json(dataset_split, output_path):
    """Save a Hugging Face dataset split to a JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for example in dataset_split:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")  # JSONL format


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="distractor",
        choices=["distractor", "fullwiki"],
        help="HotpotQA config"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./hotpotqa_json",
        help="Output directory"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading HotpotQA ({args.config})...")
    dataset = load_dataset("hotpotqa/hotpot_qa", args.config)

    for split_name, split_data in dataset.items():
        output_file = out_dir / f"{split_name}.json"
        print(f"Saving {split_name} -> {output_file}")

        save_split_to_json(split_data, output_file)

    print("All splits saved as JSON (JSONL format).")


if __name__ == "__main__":
    main()
    