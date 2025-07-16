# src/model_training.py

from sklearn.linear_model import LogisticRegression
import joblib
import os


def train_model(X_train, y_train):
    """
    Trains a Logistic Regression model and saves it.
    """
    print("Starting model training...")

    # Initialize and train the model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    print("Model training complete.")

    # Save the model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    print("Model saved to models/model.pkl")
    return model
