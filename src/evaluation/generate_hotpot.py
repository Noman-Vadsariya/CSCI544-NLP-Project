import json
from pathlib import Path

from tqdm import tqdm
from src.hypernetwork.inference import run_hypernet, load_hypernet
from datasets import load_dataset

OUTPUT_PATH = Path("src/evaluation/hotpot_hypernet_outputs.jsonl")

ds = load_dataset('parquet', data_files='data/raw_datasets/hotpotQA_gold_compact/test/ds.parquet')['train']
hypernet_model, hypernet_tokenizer = load_hypernet()

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT_PATH.open("w") as f:
    for i, sample in enumerate(tqdm(ds, desc="Generating")):
        hypernet_output = run_hypernet(hypernet_model, hypernet_tokenizer, sample)
        record = {
            "idx": i,
            "context": sample["context"],
            "prompts": sample["prompts"],
            "gold_responses": sample["responses"],
            "hypernet_outputs": hypernet_output,
        }
        f.write(json.dumps(record) + "\n")
        f.flush()
        print(f"[{i}] prompts={sample['prompts']}")
        print(f"outputs={hypernet_output}")

print(f"Wrote {len(ds)} records to {OUTPUT_PATH}")
