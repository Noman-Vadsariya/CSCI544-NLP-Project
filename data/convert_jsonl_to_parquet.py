import pandas as pd

# Replace with your actual file paths
jsonl_path = "raw_datasets/hotpotQA_niah/test.jsonl"
parquet_path = "raw_datasets/hotpotQA_niah/test.parquet"

# Read JSONL
df = pd.read_json(jsonl_path, lines=True)

# Save as Parquet
df.to_parquet(parquet_path)