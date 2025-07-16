# src/download_data.py

import os
from datasets import load_dataset


def download_raw_data():
    """
    Downloads the AG News dataset and saves the full training set
    as a single raw CSV file.
    """
    print("Starting raw data download...")

    # Define the raw data directory and ensure it exists
    raw_data_dir = "data/raw"
    os.makedirs(raw_data_dir, exist_ok=True)
    raw_file_path = os.path.join(raw_data_dir, "ag_news_raw.csv")

    # Load the dataset from Hugging Face
    # We use the 'train' split which contains 120,000 samples.
    dataset = load_dataset("ag_news", split="train")

    # Convert to a pandas DataFrame
    df = dataset.to_pandas()

    # Save to CSV in the raw data directory
    df.to_csv(raw_file_path, index=False)

    print(f"Raw data downloaded and saved to {raw_file_path}")

    return raw_file_path


if __name__ == "__main__":
    download_raw_data()
