# src/feature_engineering.py

import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from src.config import config
from src.utils import handle_errors, validate_file_exists, validate_dataframe


@handle_errors
def feature_engineering(train_path, test_path):
    """
    Applies TF-IDF vectorization to the text data and saves the
    vectorizer and transformed data.

    This function loads the training and test data, applies TF-IDF
    vectorization using scikit-learn, and saves the fitted vectorizer
    for future use.

    Args:
        train_path (str): Path to the training CSV file.
        test_path (str): Path to the test CSV file.

    Returns:
        tuple: A tuple containing (X_train_tfidf, X_test_tfidf, y_train, y_test)
               - TF-IDF transformed features and labels.

    Raises:
        FileNotFoundError: If input files don't exist.
        ValueError: If data doesn't meet validation requirements.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting feature engineering...")

    # Validate input files exist
    validate_file_exists(train_path)
    validate_file_exists(test_path)

    # Load the data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Validate the datasets
    required_columns = ["text", "label"]
    validate_dataframe(train_df, required_columns, min_rows=10)
    validate_dataframe(test_df, required_columns, min_rows=5)

    # Initialize and fit the vectorizer with config parameters
    vectorizer = TfidfVectorizer(
        stop_words=config.TFIDF_STOP_WORDS, max_features=config.TFIDF_MAX_FEATURES
    )

    # Transform the text data
    X_train_tfidf = vectorizer.fit_transform(train_df["text"])
    X_test_tfidf = vectorizer.transform(test_df["text"])

    y_train = train_df["label"]
    y_test = test_df["label"]

    logger.info("TF-IDF transformation complete.")
    logger.info("Training features shape: %s", X_train_tfidf.shape)
    logger.info("Test features shape: %s", X_test_tfidf.shape)

    # Ensure models directory exists
    config.__post_init__()

    # Save the vectorizer
    joblib.dump(vectorizer, config.vectorizer_path)
    logger.info("TF-IDF vectorizer saved to %s", config.vectorizer_path)

    return X_train_tfidf, X_test_tfidf, y_train, y_test


if __name__ == "__main__":
    # Example usage for testing
    train_path = config.train_path
    test_path = config.test_path
    X_train, X_test, y_train, y_test = feature_engineering(train_path, test_path)
    print(
        f"Feature engineering complete. Training shape: {X_train.shape}, Test shape: {X_test.shape}"
    )
