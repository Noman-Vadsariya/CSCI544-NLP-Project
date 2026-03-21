import pandas as pd
from datasets import Dataset

def load_parquet_dataset(path):
    """
    Load parquet dataset containing:
    context | query | response
    """

    df = pd.read_parquet(path)

    required_cols = ["context", "prompts", "responses"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    dataset = Dataset.from_pandas(df)

    return dataset