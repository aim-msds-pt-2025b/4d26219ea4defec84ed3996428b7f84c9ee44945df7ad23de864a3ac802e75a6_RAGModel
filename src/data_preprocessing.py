# src/data_preprocessing.py

import pandas as pd
import logging
from sklearn.model_selection import train_test_split
import numpy as np
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
            tuple: (train_path, test_path)

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

    # Create drifted versions of train and test per HW3 requirements
    # Text task: simulate drift by injecting simple noise into text and labels proportionally
    rng = np.random.default_rng(config.RANDOM_STATE)

    def drift_text(series: pd.Series) -> pd.Series:
        # Append random punctuation or duplicate words to simulate style drift
        punct = ["!", ".", "?", ",", ";"]
        choices = rng.choice(punct, size=len(series))
        # randomly duplicate last word 10-15%
        mask = rng.random(len(series)) < rng.uniform(0.10, 0.15)

        def mutate(s, ch, dupe):
            if not isinstance(s, str) or not s:
                return s
            out = s + ch
            if dupe:
                parts = s.strip().split()
                if parts:
                    out = s + " " + parts[-1]
            return out

        return pd.Series(
            [
                mutate(s, ch, bool(m))
                for s, ch, m in zip(series.tolist(), choices, mask)
            ],
            index=series.index,
        )

    train_df_drifted = train_df.copy()
    test_df_drifted = test_df.copy()
    train_df_drifted["text"] = drift_text(train_df_drifted["text"])
    test_df_drifted["text"] = drift_text(test_df_drifted["text"])

    # Randomly flip 10-15% of labels uniformly
    def flip_labels(labels: pd.Series) -> pd.Series:
        unique_labels = labels.unique().tolist()
        p = rng.uniform(0.10, 0.15)
        mask = rng.random(len(labels)) < p
        flipped = labels.copy()
        for idx in labels[mask].index:
            # choose a label different from current
            cur = labels.loc[idx]
            other_labels = [
                candidate for candidate in unique_labels if candidate != cur
            ]
            if other_labels:
                flipped.loc[idx] = rng.choice(other_labels)
        return flipped

    train_df_drifted["label"] = flip_labels(train_df_drifted["label"])
    test_df_drifted["label"] = flip_labels(test_df_drifted["label"])

    # Save drifted datasets
    drifted_train_path = "data/drifted_train.csv"
    drifted_test_path = "data/drifted_test.csv"
    train_df_drifted.to_csv(drifted_train_path, index=False)
    test_df_drifted.to_csv(drifted_test_path, index=False)

    logger.info(
        f"Data preprocessing complete. Train set: {len(train_df)} samples, Test set: {len(test_df)} samples"
    )
    logger.info(
        f"Train and test sets saved to {config.train_path} and {config.test_path}"
    )
    logger.info(
        f"Drifted datasets saved to {drifted_train_path} and {drifted_test_path}"
    )

    return config.train_path, config.test_path


def preprocess_data_with_drift(raw_data_path: str):
    """
    Extended preprocessing that also returns in-memory splits and drifted variants.

    Returns:
        tuple: (
            X_train, X_test, y_train, y_test,
            X_train_drifted, y_train_drifted, X_test_drifted, y_test_drifted
        )
    """
    # Re-run core preprocessing to ensure files are created and validated
    train_path, test_path = preprocess_data(raw_data_path)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df["text"]
    y_train = train_df["label"]
    X_test = test_df["text"]
    y_test = test_df["label"]

    # Generate drift following the same approach as in preprocess_data
    rng = np.random.default_rng(config.RANDOM_STATE)

    def drift_text(series: pd.Series) -> pd.Series:
        punct = ["!", ".", "?", ",", ";"]
        choices = rng.choice(punct, size=len(series))
        mask = rng.random(len(series)) < rng.uniform(0.10, 0.15)

        def mutate(s, ch, dupe):
            if not isinstance(s, str) or not s:
                return s
            out = s + ch
            if dupe:
                parts = s.strip().split()
                if parts:
                    out = s + " " + parts[-1]
            return out

        return pd.Series(
            [
                mutate(s, ch, bool(m))
                for s, ch, m in zip(series.tolist(), choices, mask)
            ],
            index=series.index,
        )

    def flip_labels(labels: pd.Series) -> pd.Series:
        unique_labels = labels.unique().tolist()
        p = rng.uniform(0.10, 0.15)
        mask = rng.random(len(labels)) < p
        flipped = labels.copy()
        for idx in labels[mask].index:
            cur = labels.loc[idx]
            other_labels = [
                candidate for candidate in unique_labels if candidate != cur
            ]
            if other_labels:
                flipped.loc[idx] = rng.choice(other_labels)
        return flipped

    X_train_drifted = drift_text(X_train)
    y_train_drifted = flip_labels(y_train)
    X_test_drifted = drift_text(X_test)
    y_test_drifted = flip_labels(y_test)

    # Persist drifted datasets (overwrite)
    pd.DataFrame({"text": X_train_drifted, "label": y_train_drifted}).to_csv(
        "data/drifted_train.csv", index=False
    )
    pd.DataFrame({"text": X_test_drifted, "label": y_test_drifted}).to_csv(
        "data/drifted_test.csv", index=False
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_drifted,
        y_train_drifted,
        X_test_drifted,
        y_test_drifted,
    )


if __name__ == "__main__":
    # Example usage for testing
    raw_path = config.raw_data_path
    train_path, test_path = preprocess_data(raw_path)
    print(f"Preprocessing complete. Files saved: {train_path}, {test_path}")
