# src/run_pipeline.py

from src.download_data import download_raw_data
from src.data_preprocessing import preprocess_data
from src.feature_engineering import feature_engineering
from src.model_training import train_model
from src.evaluation import evaluate_model
from src.utils import setup_logging, handle_errors
from src.drift_detection import detect_drift
import logging
import os

# MLflow import with conditional handling
try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MLFLOW_AVAILABLE = False


@handle_errors
def main():
    """
    Main function to run the entire ML pipeline.

    This function orchestrates the complete machine learning pipeline from
    data download to model evaluation. It includes proper logging and error
    handling throughout the process.

    Pipeline steps:
    1. Download raw data from external source
    2. Preprocess data (clean, split into train/test)
    3. Feature engineering (TF-IDF vectorization)
    4. Model training (Logistic Regression)
    5. Model evaluation (accuracy, classification report)

    Raises:
        Exception: If any step in the pipeline fails.
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("--- ML Pipeline Started ---")

    try:
        # Set MLflow tracking (local default or docker-compose mlflow)
        if MLFLOW_AVAILABLE and mlflow is not None:
            mlflow.set_tracking_uri(
                os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
            )

        # Step 1: Download Raw Data
        logger.info("Step 1: Downloading raw data...")
        raw_data_path = download_raw_data()

        # Step 2: Data Preprocessing
        logger.info("Step 2: Preprocessing data...")
        train_path, test_path = preprocess_data(raw_data_path)

        # Step 3: Feature Engineering
        logger.info("Step 3: Feature engineering...")
        X_train_tfidf, X_test_tfidf, y_train, y_test = feature_engineering(
            train_path, test_path
        )

        # Step 4: Model Training
        logger.info("Step 4: Training model...")
        model = train_model(X_train_tfidf, y_train)

        # Step 5: Model Evaluation
        logger.info("Step 5: Evaluating model...")
        evaluate_model(model, X_test_tfidf, y_test)

        # Drift detection using saved CSVs (drifted created during preprocessing)
        logger.info("Step 6: Drift detection...")
        # Note: Drifted data was already created in Step 2, no need to regenerate

        # Run drift detection on test set vs drifted test set
        test_drift_results = detect_drift(
            "data/processed/test.csv", "data/drifted_test.csv"
        )

        # Log drift results to MLflow if available
        if mlflow is not None:
            try:
                if MLFLOW_AVAILABLE and mlflow is not None:
                    mlflow.log_param(
                        "test_drift_detected",
                        test_drift_results.get(
                            "dataset_drift",
                            test_drift_results.get("drift_detected", False),
                        ),
                    )
                    mlflow.log_param(
                        "test_overall_drift_score",
                        test_drift_results["overall_drift_score"],
                    )
            except Exception:
                pass

        logger.info(f"Test drift results: {test_drift_results}")

        # Raise error if drift detected (as required by HW3)
        # Support both old and new result format for compatibility
        drift_detected = test_drift_results.get(
            "dataset_drift", test_drift_results.get("drift_detected", False)
        )
        if drift_detected:
            raise ValueError(
                "Data drift detected in test set! Model retraining required."
            )

        # Optional: Also check train vs drifted train
        train_drift_results = detect_drift(
            "data/processed/train.csv", "data/drifted_train.csv"
        )
        logger.info(f"Train drift results: {train_drift_results}")

        logger.info("--- ML Pipeline Finished Successfully ---")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
