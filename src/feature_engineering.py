# src/feature_engineering.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os


def feature_engineering(train_path, test_path):
    """
    Applies TF-IDF vectorization to the text data and saves the
    vectorizer and transformed data.
    """
    print("Starting feature engineering...")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Initialize and fit the vectorizer on the training data
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

    X_train_tfidf = vectorizer.fit_transform(train_df["text"])
    X_test_tfidf = vectorizer.transform(test_df["text"])

    y_train = train_df["label"]
    y_test = test_df["label"]

    print("TF-IDF transformation complete.")

    # Save the vectorizer
    os.makedirs("models", exist_ok=True)
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    print("TF-IDF vectorizer saved to models/tfidf_vectorizer.pkl")

    return X_train_tfidf, X_test_tfidf, y_train, y_test
