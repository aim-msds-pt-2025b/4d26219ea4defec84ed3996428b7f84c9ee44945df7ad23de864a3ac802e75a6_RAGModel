# src/run_pipeline.py

from download_data import download_raw_data  # ADDED
from data_preprocessing import preprocess_data
from feature_engineering import feature_engineering
from model_training import train_model
from evaluation import evaluate_model


def main():
    """
    Main function to run the entire ML pipeline.
    """
    print("--- ML Pipeline Started ---")

    # Step 1: Download Raw Data
    raw_data_path = download_raw_data()

    # Step 2: Data Preprocessing (uses the raw data path)
    train_path, test_path = preprocess_data(raw_data_path)

    # Step 3: Feature Engineering
    X_train_tfidf, X_test_tfidf, y_train, y_test = feature_engineering(
        train_path, test_path
    )

    # Step 4: Model Training
    model = train_model(X_train_tfidf, y_train)

    # Step 5: Model Evaluation
    evaluate_model(model, X_test_tfidf, y_test)

    print("--- ML Pipeline Finished Successfully ---")


if __name__ == "__main__":
    main()
