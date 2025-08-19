# src/evaluation.py

import logging
import json
import os
from sklearn.metrics import accuracy_score, classification_report
from src.config import config
from src.utils import handle_errors
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlflow  # type: ignore
mlflow = None
try:
    import mlflow as _mlflow  # type: ignore

    mlflow = _mlflow
except Exception:
    pass


@handle_errors
def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the test set and saves the metrics.

    This function makes predictions on the test set, calculates accuracy
    and classification metrics, and saves the results to a file.

    Args:
        model: The trained model to evaluate.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        float: The accuracy score of the model.

    Raises:
        Exception: If model evaluation or metric saving fails.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting model evaluation...")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    logger.info("Model Accuracy: %.4f", accuracy)
    logger.info("Classification Report:")
    logger.info("\n%s", report)

    # Ensure reports directory exists
    config.__post_init__()

    # Save metrics to a file and JSON
    with open(config.metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(str(report))

    json_path = os.path.join("reports", "evaluation_results.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump({"accuracy": float(accuracy)}, jf, indent=2)
    # Log 2 metrics to MLflow (accuracy and f1 macro if available via report would require parsing; keep accuracy)
    if mlflow is not None:
        try:
            mlflow.log_metric("accuracy", float(accuracy))
        except Exception:
            pass

    logger.info("Metrics saved to %s", config.metrics_path)
    return accuracy


if __name__ == "__main__":
    # Example usage for testing
    print("This module should be run as part of the full pipeline.")
    print("Use run_pipeline.py to execute the complete ML pipeline.")
