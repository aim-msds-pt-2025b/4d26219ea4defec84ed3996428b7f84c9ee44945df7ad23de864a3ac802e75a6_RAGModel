# src/evaluation.py

from sklearn.metrics import accuracy_score, classification_report
import os


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the test set and saves the metrics.
    """
    print("Starting model evaluation...")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Model Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    # Save metrics to a file
    os.makedirs("reports", exist_ok=True)
    with open("reports/metrics.txt", "w") as f:
        f.write(f"Accuracy: {accuracy}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    print("Metrics saved to reports/metrics.txt")
    return accuracy
