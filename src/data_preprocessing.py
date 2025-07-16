# src/data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
import os


def preprocess_data(raw_data_path: str):  # ADDED: Function now takes a path as input
    """
    Loads the raw dataset from a CSV file, splits it into training
    and testing sets, and saves them as processed CSV files.

    Args:
        raw_data_path (str): The file path for the raw data CSV.
    """
    print("Starting data preprocessing...")

    # Load dataset from the local raw file
    df = pd.read_csv(raw_data_path)  # MODIFIED: Reads from CSV instead of Hugging Face

    # Map labels to human-readable names
    label_names = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    df["label_name"] = df["label"].map(label_names)

    print(f"Dataset loaded with {len(df)} records.")

    # Split data into training and testing sets
    X = df["text"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create processed directory if it doesn't exist
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    # Define processed file paths
    train_path = os.path.join(processed_dir, "train.csv")
    test_path = os.path.join(processed_dir, "test.csv")

    # Save the split data
    train_df = pd.DataFrame({"text": X_train, "label": y_train})
    test_df = pd.DataFrame({"text": X_test, "label": y_test})

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Data preprocessing complete. Train and test sets saved.")

    return train_path, test_path


# REMOVED: The if __name__ == '__main__' block is no longer needed
# as this script is primarily a module in the pipeline.
