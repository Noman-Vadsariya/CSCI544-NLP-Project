import gc
from collections import Counter

from datasets import Dataset, load_dataset
import pandas as pd
from tqdm import tqdm

def merge_parquet_file():
    # Replace with your actual file paths
    file1 = 'raw_datasets/raw_hotpotQA/train-0.parquet'
    file2 = 'raw_datasets/raw_hotpotQA/train-1.parquet'
    output_file = 'raw_datasets/raw_hotpotQA/train.parquet'

    # Read both Parquet files
    df1 = pd.read_parquet(file1)
    df2 = pd.read_parquet(file2)

    # Concatenate the DataFrames
    merged_df = pd.concat([df1, df2], ignore_index=True)

    # Save the merged DataFrame to a new Parquet file
    merged_df.to_parquet(output_file)

if __name__ == "__main__":
    ds_name = "raw_datasets/raw_hotpotQA"
    # merge_parquet_file()
    # for split in ["train", "test"]:
    for split in ["train", "test"]:
        ctx_qa_dict = dict()
        ds = load_dataset(ds_name, split=split)
        for i, sample in tqdm(enumerate(ds)):
            gold_titles = Counter(sample["supporting_facts"]["title"])
            sentences = []
            gold_sentences = []
            for title, sent in zip(sample["context"]["title"], sample["context"]["sentences"]):
                if title in gold_titles:
                    gold_sentences.extend(sent)
                sentences.extend(sent)

            response = sample["answer"]
            ctx = "\n".join(sentences)
            gold_ctx = "\n".join(gold_sentences)
            q = sample["question"]
            if ctx not in ctx_qa_dict:
                ctx_qa_dict[ctx] = {"prompts": [], "responses": [], "gold_context": ""}
            ctx_qa_dict[ctx]["prompts"].append(q)
            ctx_qa_dict[ctx]["responses"].append(response)
            ctx_qa_dict[ctx]["gold_context"] = gold_ctx

        print(f"Unique contexts: {len(ctx_qa_dict)}")
        # convert ctx_qa_dict to a list of dictionaries
        samples = [
            {
                "context": ctx,
                "prompts": ctx_qa_dict[ctx]["prompts"],
                "responses": ctx_qa_dict[ctx]["responses"],
                "gold_context": ctx_qa_dict[ctx]["gold_context"],
            }
            for ctx in ctx_qa_dict
        ]
        print(f"Sampled data: {samples[0]}")
        # breakpoint()
        # save to a new dataset
        ds = Dataset.from_list(samples)

        save_path = f"raw_datasets/hotpotQA_compact/{split}/ds.parquet"
        print(f"Saving dataset to {save_path}")
        ds.to_parquet(save_path)
        print("=" * 80)
        del ds, samples, ctx_qa_dict
        gc.collect()