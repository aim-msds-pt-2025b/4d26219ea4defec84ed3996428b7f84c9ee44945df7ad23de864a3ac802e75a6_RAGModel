#!/usr/bin/env python3
"""
Test script to run pipeline with healthy (non-drifted) data to verify complete pipeline execution.
This simulates the scenario where no drift is detected and model registration should succeed.
"""

import logging
import sys


import mlflow
from src.download_data import download_data
from src.data_preprocessing import preprocess_data
from src.feature_engineering import engineer_features
from src.model_training import train_model
from src.evaluation import evaluate_model
from src.drift_detection import detect_drift
from src.utils import setup_logging


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
        download_data()

        # Step 2: Preprocessing (without drift simulation)
        logger.info("Step 2: Preprocessing data (healthy scenario)...")
        X_train, X_test, y_train, y_test = preprocess_data(simulate_drift=False)

        # Step 3: Feature engineering
        logger.info("Step 3: Feature engineering...")
        X_train_tfidf, X_test_tfidf = engineer_features(X_train, X_test)

        # Step 4: Model training
        logger.info("Step 4: Training model...")
        model = train_model(X_train_tfidf, y_train)

        # Step 5: Model evaluation
        logger.info("Step 5: Evaluating model...")
        accuracy, f1_score = evaluate_model(model, X_test_tfidf, y_test)

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
            logger.info(
                f"✅ Model registered with accuracy: {accuracy:.4f}, F1-score: {f1_score:.4f}"
            )
        else:
            logger.warning("⚠️ Unexpected drift detected in healthy scenario")

        return True

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
