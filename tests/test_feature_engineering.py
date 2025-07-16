# tests/test_feature_engineering.py

import os
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from src.feature_engineering import feature_engineering
from src.config import config


@pytest.fixture
def sample_train_test_data():
    """Create sample training and test data for testing."""
    train_data = pd.DataFrame(
        {
            "text": [
                "business news about stocks and markets",
                "sports news about football game results",
                "technology news about artificial intelligence",
                "world news about international politics",
                "business report on company earnings",
                "sports update on basketball tournament",
                "tech news about quantum computing",
                "world update on climate change",
                "business analysis of market trends",
                "sports coverage of tennis match",
                "technology review of latest gadgets",
                "world report on political developments",
            ],
            "label": [2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3, 0],
        }
    )

    test_data = pd.DataFrame(
        {
            "text": [
                "business analysis of market trends",
                "sports coverage of tennis match",
                "tech article about cybersecurity",
                "world news about humanitarian crisis",
                "business update on company stocks",
                "sports news about tournament results",
            ],
            "label": [2, 1, 3, 0, 2, 1],
        }
    )

    # Ensure directories exist
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

    # Save test data
    train_path = os.path.join(config.PROCESSED_DATA_DIR, "test_train.csv")
    test_path = os.path.join(config.PROCESSED_DATA_DIR, "test_test.csv")

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    yield train_path, test_path

    # Cleanup
    for path in [train_path, test_path]:
        if os.path.exists(path):
            os.remove(path)


def test_feature_engineering_creates_vectorizer(sample_train_test_data):
    """Test that feature engineering creates and saves the vectorizer."""
    train_path, test_path = sample_train_test_data

    # Run feature engineering
    X_train, X_test, y_train, y_test = feature_engineering(train_path, test_path)

    # Check that vectorizer was saved
    assert os.path.exists(config.vectorizer_path)

    # Check output shapes and types
    assert X_train.shape[0] == 12  # 12 training samples
    assert X_test.shape[0] == 6  # 6 test samples
    assert X_train.shape[1] == X_test.shape[1]  # Same number of features
    assert len(y_train) == 12
    assert len(y_test) == 6

    # Check that the vectorizer can be loaded
    import joblib

    loaded_vectorizer = joblib.load(config.vectorizer_path)
    assert isinstance(loaded_vectorizer, TfidfVectorizer)

    # Cleanup
    if os.path.exists(config.vectorizer_path):
        os.remove(config.vectorizer_path)


def test_feature_engineering_with_missing_files():
    """Test that feature engineering raises error for missing files."""
    non_existent_train = "data/processed/non_existent_train.csv"
    non_existent_test = "data/processed/non_existent_test.csv"

    with pytest.raises(FileNotFoundError):
        feature_engineering(non_existent_train, non_existent_test)


def test_feature_engineering_with_invalid_columns():
    """Test that feature engineering raises error for invalid data structure."""
    # Create sample data with missing required columns
    invalid_data = pd.DataFrame(
        {
            "content": ["Some text"],  # Wrong column name
            "category": [0],  # Wrong column name
        }
    )

    # Create temporary files
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    invalid_train_path = os.path.join(config.PROCESSED_DATA_DIR, "invalid_train.csv")
    invalid_test_path = os.path.join(config.PROCESSED_DATA_DIR, "invalid_test.csv")

    invalid_data.to_csv(invalid_train_path, index=False)
    invalid_data.to_csv(invalid_test_path, index=False)

    with pytest.raises(ValueError):
        feature_engineering(invalid_train_path, invalid_test_path)

    # Cleanup
    for path in [invalid_train_path, invalid_test_path]:
        if os.path.exists(path):
            os.remove(path)
