from src.dataset_loader import load_parquet_dataset
from src.tokenizer_pipeline import TokenizerPipeline

dataset = load_parquet_dataset("data/raw_datasets/combined_compact/train/ds.parquet")

tokenizer = TokenizerPipeline()

sample = dataset[0]

context = sample["context"]

chunks, masks = tokenizer.chunk_context(context, chunk_size=256)

print("Number of chunks:", len(chunks))

print("Chunk shape:", chunks[0].shape)
print("Mask shape:", masks[0].shape)

print("\nFirst 10 tokens:")
print(chunks[0][:10])