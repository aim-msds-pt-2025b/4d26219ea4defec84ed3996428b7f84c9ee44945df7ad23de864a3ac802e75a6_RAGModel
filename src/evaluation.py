# src/evaluation.py

import logging
from sklearn.metrics import accuracy_score, classification_report
from src.config import config
from src.utils import handle_errors


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

    # Save metrics to a file
    with open(config.metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(str(report))

    logger.info("Metrics saved to %s", config.metrics_path)
    return accuracy


if __name__ == "__main__":
    # Example usage for testing
    print("This module should be run as part of the full pipeline.")
    print("Use run_pipeline.py to execute the complete ML pipeline.")
