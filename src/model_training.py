# src/model_training.py

import logging
import os
from sklearn.linear_model import LogisticRegression
import joblib
from src.config import config
from src.utils import handle_errors
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlflow  # type: ignore
    import mlflow.pyfunc  # type: ignore
mlflow = None
try:  # runtime optional import
    import mlflow as _mlflow  # type: ignore

    mlflow = _mlflow
except Exception:
    pass


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
    random_state = config.RANDOM_STATE
    model = LogisticRegression(max_iter=1000, random_state=random_state)

    # MLflow tracking
    if mlflow is not None:
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        )
        with mlflow.start_run(run_name="train_logreg"):
            # Log 3 hyperparameters (adapted for logistic regression)
            mlflow.log_param("max_iter", 1000)
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("penalty", "l2")

            model.fit(X_train, y_train)
    else:
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
    # Log artifact if mlflow available
    if mlflow is not None:
        mlflow.log_artifact(config.model_path, artifact_path="model_artifacts")

    return model


if __name__ == "__main__":
    # Example usage for testing
    print("This module should be run as part of the full pipeline.")
    print("Use run_pipeline.py to execute the complete ML pipeline.")
