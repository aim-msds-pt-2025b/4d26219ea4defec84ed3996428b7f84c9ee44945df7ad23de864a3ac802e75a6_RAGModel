# src/run_pipeline.py

from src.download_data import download_raw_data
from src.data_preprocessing import preprocess_data
from src.feature_engineering import feature_engineering
from src.model_training import train_model
from src.evaluation import evaluate_model
from src.utils import setup_logging, handle_errors
import logging


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

        logger.info("--- ML Pipeline Finished Successfully ---")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
