import gc

from datasets import Dataset, load_dataset
from tqdm import tqdm

def split_and_save_dataset(dataset, train_ratio, train_save_path, test_save_path):
    """
    Splits the dataset into train and test sets and saves them to the specified paths.

    Args:
        dataset (Dataset): The dataset to split.
        train_ratio (float): The ratio of the dataset to use for training.
        train_save_path (str): Path to save the train dataset.
        test_save_path (str): Path to save the test dataset.
    """
    # Shuffle the dataset
    dataset = dataset.shuffle(seed=42)
    
    # Calculate split sizes
    train_size = int(len(dataset) * train_ratio)
    
    # Split the dataset
    train_dataset = dataset.select(range(train_size))
    test_dataset = dataset.select(range(train_size, len(dataset)))
    
    # Save the train dataset
    print(f"Saving train dataset to {train_save_path}")
    train_dataset.to_parquet(train_save_path)
    
    # Save the test dataset
    print(f"Saving test dataset to {test_save_path}")
    test_dataset.to_parquet(test_save_path)


if __name__ == "__main__":
    ds_name = "raw_datasets/prontoQA_compact"

    for split in ["train"]:
        ctx_qa_dict = dict()
        ds = load_dataset(ds_name, split=split)
        print(f"Original size: {len(ds)}")
        for i, sample in tqdm(enumerate(ds)):
            ctx = sample["context"]
            if ctx not in ctx_qa_dict:
                ctx_qa_dict[ctx] = {"prompts": [], "responses": []}
            # question = closed_qa_prompting(sample["prompt"])
            question = sample["question"]
            answer = "True" if sample["answer"] == 'A' else "False"
            ctx_qa_dict[ctx]["prompts"].append(question)
            ctx_qa_dict[ctx]["responses"].append(answer)

        print(f"Unique contexts: {len(ctx_qa_dict)}")
        # convert ctx_qa_dict to a list of dictionaries
        samples = [
            {
                "context": ctx,
                "prompts": ctx_qa_dict[ctx]["prompts"],
                "responses": ctx_qa_dict[ctx]["responses"],
            }
            for ctx in ctx_qa_dict
        ]
        print(f"Sampled data: {samples[0]}")
        # breakpoint()
        # save to a new dataset
        ds = Dataset.from_list(samples)

        
        # Split the dataset into train and test sets and save them
        train_save_path = "./raw_datasets/prontoQA_compact/train/ds.parquet"
        test_save_path = "./raw_datasets/prontoQA_compact/test/ds.parquet"
        split_and_save_dataset(ds, train_ratio=0.7, train_save_path=train_save_path, test_save_path=test_save_path)

        del ds, samples, ctx_qa_dict
        gc.collect()