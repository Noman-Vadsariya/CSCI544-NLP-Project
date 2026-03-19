from src.dataset_loader import load_parquet_dataset

DATA_PATH = "data/raw_datasets/combined_compact/train/ds.parquet"

dataset = load_parquet_dataset(DATA_PATH)

print("Dataset loaded successfully")
print("Number of samples:", len(dataset))

sample = dataset[0]
print("\nSample Example:")
print("Context:", sample["context"][:200])
print("Prompt:", sample["prompts"])
print("Response:", sample["responses"])