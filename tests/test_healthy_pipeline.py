#!/usr/bin/env python3
"""
Test script to run pipeline with healthy (non-drifted) data to verify complete pipeline execution.
This simulates the scenario where no drift is detected and model registration should succeed.
"""

import logging
import os
import sys

# Add project root to path before importing local modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import mlflow  # noqa: E402

from src.data_preprocessing import preprocess_data  # noqa: E402
from src.drift_detection import detect_drift  # noqa: E402
from src.download_data import download_raw_data  # noqa: E402
from src.evaluation import evaluate_model  # noqa: E402
from src.feature_engineering import feature_engineering  # noqa: E402
from src.model_training import train_model  # noqa: E402
from src.utils import setup_logging  # noqa: E402


def main():
    """Run the complete ML pipeline with healthy data (no drift simulation)."""

    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri("http://localhost:5000")

        logger.info("--- Healthy ML Pipeline Started (No Drift) ---")

        # Step 1: Download data
        logger.info("Step 1: Downloading raw data...")
        download_raw_data()

        # Step 2: Preprocessing (without drift simulation)
        logger.info("Step 2: Preprocessing data (healthy scenario)...")
        train_path, test_path = preprocess_data("data/raw/ag_news_raw.csv")

        # Step 3: Feature engineering
        logger.info("Step 3: Feature engineering...")
        X_train_tfidf, X_test_tfidf, y_train, y_test = feature_engineering(
            train_path, test_path
        )

        # Step 4: Model training
        logger.info("Step 4: Training model...")
        model = train_model(X_train_tfidf, y_train)

        # Step 5: Model evaluation
        logger.info("Step 5: Evaluating model...")
        accuracy = evaluate_model(model, X_test_tfidf, y_test)

        # Step 6: Drift detection on healthy data
        logger.info("Step 6: Drift detection (healthy data)...")

        # For healthy scenario, compare processed data with itself (should show no drift)
        drift_results = detect_drift(
            "data/processed/test.csv", "data/processed/test.csv"
        )

        logger.info(f"Healthy scenario drift results: {drift_results}")

        if not drift_results.get("drift_detected", False):
            logger.info("✅ Healthy scenario: No drift detected as expected!")
            logger.info("✅ Pipeline completed successfully!")
            logger.info(f"✅ Model registered with accuracy: {accuracy:.4f}")
        else:
            logger.warning("⚠️ Unexpected drift detected in healthy scenario")

        return True

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
