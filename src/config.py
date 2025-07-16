# src/config.py

import os
from dataclasses import dataclass


@dataclass
class Config:
    """
    Configuration settings for the ML pipeline.

    This class centralizes all configuration parameters for the machine learning
    pipeline, including file paths, model parameters, and other settings.
    """

    # Data paths
    RAW_DATA_DIR: str = "data/raw"
    PROCESSED_DATA_DIR: str = "data/processed"
    MODELS_DIR: str = "models"
    REPORTS_DIR: str = "reports"

    # Model parameters
    TFIDF_MAX_FEATURES: int = 5000
    TFIDF_STOP_WORDS: str = "english"
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42

    # File names
    RAW_DATA_FILE: str = "ag_news_raw.csv"
    TRAIN_FILE: str = "train.csv"
    TEST_FILE: str = "test.csv"
    MODEL_FILE: str = "model.pkl"
    VECTORIZER_FILE: str = "tfidf_vectorizer.pkl"
    METRICS_FILE: str = "metrics.txt"

    def __post_init__(self):
        """Create directories if they don't exist."""
        for directory in [
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.MODELS_DIR,
            self.REPORTS_DIR,
        ]:
            os.makedirs(directory, exist_ok=True)

    @property
    def raw_data_path(self) -> str:
        """Get the full path to the raw data file."""
        return os.path.join(self.RAW_DATA_DIR, self.RAW_DATA_FILE)

    @property
    def train_path(self) -> str:
        """Get the full path to the training data file."""
        return os.path.join(self.PROCESSED_DATA_DIR, self.TRAIN_FILE)

    @property
    def test_path(self) -> str:
        """Get the full path to the test data file."""
        return os.path.join(self.PROCESSED_DATA_DIR, self.TEST_FILE)

    @property
    def model_path(self) -> str:
        """Get the full path to the model file."""
        return os.path.join(self.MODELS_DIR, self.MODEL_FILE)

    @property
    def vectorizer_path(self) -> str:
        """Get the full path to the vectorizer file."""
        return os.path.join(self.MODELS_DIR, self.VECTORIZER_FILE)

    @property
    def metrics_path(self) -> str:
        """Get the full path to the metrics file."""
        return os.path.join(self.REPORTS_DIR, self.METRICS_FILE)


# Global config instance
config = Config()
