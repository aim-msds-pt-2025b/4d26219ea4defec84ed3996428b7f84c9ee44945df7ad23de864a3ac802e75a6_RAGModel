# tests/test_data.py

import os
import pandas as pd
import pytest
from src.data_preprocessing import preprocess_data
from src.config import config


def test_preprocess_data_creates_files():
    """
    Tests if the preprocess_data function successfully creates
    the train and test CSV files.
    """
    # Create sample raw data with more rows to meet validation requirements
    sample_data = pd.DataFrame(
        {
            "text": [
                "Business news about stocks and markets",
                "Sports news about football game",
                "Technology news about AI development",
                "World news about politics",
                "Business report on company earnings",
                "Sports update on basketball",
                "Tech article about machine learning",
                "World update on international affairs",
                "Business analysis of market trends",
                "Sports coverage of tennis match",
            ]
            * 15,  # Repeat 15 times to get 150 rows
            "label": [2, 1, 3, 0, 2, 1, 3, 0, 2, 1] * 15,
        }
    )

    # Create raw data directory and save sample data
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    sample_data.to_csv(config.raw_data_path, index=False)

    # Clean up previous test files if they exist
    if os.path.exists(config.train_path):
        os.remove(config.train_path)
    if os.path.exists(config.test_path):
        os.remove(config.test_path)

    # Run the function to be tested
    train_path, test_path = preprocess_data(config.raw_data_path)

    # Assert that the files now exist
    assert os.path.exists(train_path)
    assert os.path.exists(test_path)

    # Assert that the files are not empty
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    assert not train_df.empty
    assert not test_df.empty

    # Assert that the split is reasonable
    assert len(train_df) > len(test_df)  # Train should be larger than test
    assert len(train_df) + len(test_df) == len(sample_data)  # Total should match

    # Assert that required columns exist
    assert "text" in train_df.columns
    assert "label" in train_df.columns
    assert "text" in test_df.columns
    assert "label" in test_df.columns

    # Clean up test files
    os.remove(config.raw_data_path)
    os.remove(train_path)
    os.remove(test_path)


def test_preprocess_data_with_invalid_file():
    """
    Tests that preprocess_data raises appropriate error for non-existent file.
    """
    non_existent_path = "data/raw/non_existent_file.csv"

    with pytest.raises(FileNotFoundError):
        preprocess_data(non_existent_path)


def test_preprocess_data_with_invalid_columns():
    """
    Tests that preprocess_data raises appropriate error for invalid data structure.
    """
    # Create sample data with missing required columns
    sample_data = pd.DataFrame(
        {
            "content": ["Some text"],  # Wrong column name
            "category": [0],  # Wrong column name
        }
    )

    # Create raw data directory and save sample data
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    invalid_data_path = os.path.join(config.RAW_DATA_DIR, "invalid_data.csv")
    sample_data.to_csv(invalid_data_path, index=False)

    with pytest.raises(ValueError):
        preprocess_data(invalid_data_path)

    # Clean up
    os.remove(invalid_data_path)
