# tests/test_data.py
import os
import pandas as pd
from src.data_preprocessing import preprocess_data


def test_preprocess_data_creates_files():
    """
    Tests if the preprocess_data function successfully creates
    the train and test CSV files.
    """
    # Clean up previous runs if files exist
    if os.path.exists("data/processed/train.csv"):
        os.remove("data/processed/train.csv")
    if os.path.exists("data/processed/test.csv"):
        os.remove("data/processed/test.csv")

    # Run the function to be tested
    train_path, test_path = preprocess_data()

    # Assert that the files now exist
    assert os.path.exists(train_path)
    assert os.path.exists(test_path)

    # Assert that the files are not empty
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    assert not train_df.empty
    assert not test_df.empty
