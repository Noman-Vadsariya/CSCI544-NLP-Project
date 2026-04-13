import os
from datasets import Dataset, concatenate_datasets, Value


def combine_and_shuffle_parquet_files(
    dataset_folders, output_train_path, output_test_path
):
    """
    Combines all train and test .parquet files in the specified dataset folders,
    shuffles them, and saves the combined datasets.

    Args:
        dataset_folders (list): List of dataset folder paths to combine.
        output_train_path (str): Path to save the combined train dataset.
        output_test_path (str): Path to save the combined test dataset.
    """
    train_datasets = []
    test_datasets = []

    # Process each specified dataset folder
    for folder in dataset_folders:
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith(".parquet"):
                    file_path = os.path.join(root, file)
                    dataset = Dataset.from_parquet(file_path)
                    # Cast 'context' column to string if it exists and is not already string
                    if "context" in dataset.column_names:
                        # Save original context lengths for truncation check
                        orig_lengths = [
                            len(str(x)) if x is not None else 0
                            for x in dataset["context"]
                        ]
                        dataset = dataset.cast_column("context", Value("string"))
                        # Check for truncation
                        new_lengths = [
                            len(str(x)) if x is not None else 0
                            for x in dataset["context"]
                        ]
                        for i, (orig, new) in enumerate(zip(orig_lengths, new_lengths)):
                            if orig != new:
                                print(
                                    f"WARNING: Possible truncation in 'context' at row {i} in {file_path}"
                                )
                                break
                    num_samples = len(dataset)
                    if (
                        "train" in root.lower()
                    ):  # Check if the file is in a train directory
                        print(
                            f"Adding train file: {file_path} with {num_samples} samples"
                        )
                        train_datasets.append(dataset)
                    elif (
                        "test" in root.lower()
                    ):  # Check if the file is in a test directory
                        print(
                            f"Adding test file: {file_path} with {num_samples} samples"
                        )
                        test_datasets.append(dataset)

    # Combine all train datasets and shuffle
    if train_datasets:
        combined_train = concatenate_datasets(train_datasets).shuffle(seed=42)
        print(f"Saving combined train dataset to {output_train_path}")
        combined_train.to_parquet(output_train_path)
        print(f"Combined train dataset size: {len(combined_train)}\n")
    else:
        print("No train files found to combine.")

    # Combine all test datasets and shuffle
    if test_datasets:
        combined_test = concatenate_datasets(test_datasets).shuffle(seed=42)
        print(f"Saving combined test dataset to {output_test_path}")
        combined_test.to_parquet(output_test_path)
        print(f"Combined test dataset size: {len(combined_test)}\n")
    else:
        print("No test files found to combine.")


if __name__ == "__main__":
    # Specify the dataset folders to combine
    dataset_folders = [
        "./raw_datasets/hotpotQA_compact",
        # "./raw_datasets/hotpotQA_gold_compact",
        "./raw_datasets/asqa_compact",
        "./raw_datasets/prontoQA_compact",
    ]

    # output_train_file = "./raw_datasets/combined_compact/train/ds.parquet"
    # output_test_file = "./raw_datasets/combined_compact/test/ds.parquet"

    output_train_file = "./raw_datasets/combined_noisy_datasets/train/ds.parquet"
    output_test_file = "./raw_datasets/combined_noisy_datasets/test/ds.parquet"

    combine_and_shuffle_parquet_files(
        dataset_folders, output_train_file, output_test_file
    )
