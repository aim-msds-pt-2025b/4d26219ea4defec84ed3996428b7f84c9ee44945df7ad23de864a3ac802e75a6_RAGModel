# src/download_data.py

import logging
from datasets import load_dataset
from src.config import config
from src.utils import handle_errors


@handle_errors
def download_raw_data():
    """
    Downloads the AG News dataset and saves the full training set
    as a single raw CSV file.

    This function loads the AG News dataset from Hugging Face's datasets
    library and saves it as a CSV file for further processing.

    Returns:
        str: Path to the downloaded raw data file.

    Raises:
        Exception: If data download or saving fails.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting raw data download...")

    # Ensure raw data directory exists
    config.__post_init__()

    # Load the dataset from Hugging Face
    # We use the 'train' split which contains 120,000 samples.
    logger.info("Loading AG News dataset from Hugging Face...")
    dataset = load_dataset("ag_news", split="train")

    # Convert to a pandas DataFrame
    df = dataset.to_pandas()  # type: ignore

    # Validate the dataset
    logger.info(
        "Dataset loaded with %d samples and %d columns", len(df), len(df.columns)
    )  # type: ignore

    # Save to CSV in the raw data directory
    df.to_csv(config.raw_data_path, index=False)

    logger.info(f"Raw data downloaded and saved to {config.raw_data_path}")
    return config.raw_data_path


if __name__ == "__main__":
    download_raw_data()
