# src/model_training.py

import logging
from sklearn.linear_model import LogisticRegression
import joblib
from src.config import config
from src.utils import handle_errors


@handle_errors
def train_model(X_train, y_train):
    """
    Trains a Logistic Regression model and saves it.

    This function trains a logistic regression classifier on the provided
    training data and saves the trained model to disk.

    Args:
        X_train: Training features (typically TF-IDF transformed text).
        y_train: Training labels.

    Returns:
        sklearn.linear_model.LogisticRegression: The trained model.

    Raises:
        Exception: If model training or saving fails.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting model training...")

    # Initialize and train the model
    model = LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)
    model.fit(X_train, y_train)

    logger.info("Model training complete.")
    logger.info(
        "Model trained on %d samples with %d features",
        X_train.shape[0],
        X_train.shape[1],
    )

    # Ensure models directory exists
    config.__post_init__()

    # Save the model
    joblib.dump(model, config.model_path)
    logger.info("Model saved to %s", config.model_path)

    return model


if __name__ == "__main__":
    # Example usage for testing
    print("This module should be run as part of the full pipeline.")
    print("Use run_pipeline.py to execute the complete ML pipeline.")
