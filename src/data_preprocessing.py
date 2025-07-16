# src/data_preprocessing.py

import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from src.config import config
from src.utils import (
    handle_errors,
    validate_file_exists,
    validate_dataframe,
    validate_text_data,
)


@handle_errors
def preprocess_data(raw_data_path: str):
    """
    Loads the raw dataset from a CSV file, splits it into training
    and testing sets, and saves them as processed CSV files.

    This function performs data preprocessing including data validation,
    label mapping, and train-test splitting with stratification.

    Args:
        raw_data_path (str): The file path for the raw data CSV.

    Returns:
        tuple: A tuple containing (train_path, test_path) - paths to the
               saved training and testing CSV files.

    Raises:
        FileNotFoundError: If the raw data file doesn't exist.
        ValueError: If the data doesn't meet validation requirements.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting data preprocessing...")

    # Validate input file exists
    validate_file_exists(raw_data_path)

    # Load dataset from the local raw file
    df = pd.read_csv(raw_data_path)
    logger.info(f"Dataset loaded with {len(df)} records.")

    # Validate the dataset
    validate_dataframe(df, required_columns=["text", "label"], min_rows=100)
    validate_text_data(df, text_column="text")

    # Map labels to human-readable names
    label_names = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    df["label_name"] = df["label"].map(label_names)

    # Split data into training and testing sets
    X = df["text"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    # Ensure processed directory exists
    config.__post_init__()

    # Save the split data
    train_df = pd.DataFrame({"text": X_train, "label": y_train})
    test_df = pd.DataFrame({"text": X_test, "label": y_test})

    train_df.to_csv(config.train_path, index=False)
    test_df.to_csv(config.test_path, index=False)

    logger.info(
        f"Data preprocessing complete. Train set: {len(train_df)} samples, Test set: {len(test_df)} samples"
    )
    logger.info(
        f"Train and test sets saved to {config.train_path} and {config.test_path}"
    )

    return config.train_path, config.test_path


if __name__ == "__main__":
    # Example usage for testing
    raw_path = config.raw_data_path
    train_path, test_path = preprocess_data(raw_path)
    print(f"Preprocessing complete. Files saved: {train_path}, {test_path}")
