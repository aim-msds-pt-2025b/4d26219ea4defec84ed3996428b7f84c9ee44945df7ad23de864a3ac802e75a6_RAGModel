# tests/test_evaluation.py

import os
import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.evaluation import evaluate_model
from src.config import config


@pytest.fixture
def sample_model_and_data():
    """Create a sample model and test data for evaluation."""
    np.random.seed(42)
    X_train = np.random.rand(100, 50)
    y_train = np.random.randint(0, 4, 100)

    # Train a simple model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    # Create test data
    X_test = np.random.rand(20, 50)
    y_test = np.random.randint(0, 4, 20)

    return model, X_test, y_test


@pytest.mark.integration
@pytest.mark.slow
def test_evaluate_model_saves_metrics(sample_model_and_data):
    """Test that model evaluation saves metrics correctly."""
    model, X_test, y_test = sample_model_and_data

    # Clean up any existing metrics file
    if os.path.exists(config.metrics_path):
        os.remove(config.metrics_path)

    # Evaluate the model
    accuracy = evaluate_model(model, X_test, y_test)

    # Check that metrics file was created
    assert os.path.exists(config.metrics_path)

    # Check that accuracy is reasonable
    assert 0.0 <= accuracy <= 1.0
    assert isinstance(accuracy, float)

    # Check that metrics file contains expected content
    with open(config.metrics_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert "Accuracy:" in content
        assert "Classification Report:" in content
        assert f"{accuracy:.4f}" in content

    # Cleanup
    if os.path.exists(config.metrics_path):
        os.remove(config.metrics_path)


@pytest.mark.unit
@pytest.mark.fast
def test_evaluate_model_returns_correct_accuracy(sample_model_and_data):
    """Test that evaluate_model returns the correct accuracy."""
    model, X_test, y_test = sample_model_and_data

    # Clean up any existing metrics file
    if os.path.exists(config.metrics_path):
        os.remove(config.metrics_path)

    # Get predictions manually to compare
    y_pred = model.predict(X_test)
    expected_accuracy = np.mean(y_pred == y_test)

    # Evaluate using the function
    returned_accuracy = evaluate_model(model, X_test, y_test)

    # Check that accuracies match
    assert abs(returned_accuracy - expected_accuracy) < 1e-6

    # Cleanup
    if os.path.exists(config.metrics_path):
        os.remove(config.metrics_path)


@pytest.mark.unit
@pytest.mark.fast
def test_evaluate_model_with_perfect_predictions():
    """Test evaluation with perfect predictions."""
    np.random.seed(42)
    X_test = np.random.rand(10, 5)
    y_test = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])

    # Create a mock model that always predicts correctly
    class PerfectModel:
        def predict(self, X):
            return y_test

    model = PerfectModel()

    # Clean up any existing metrics file
    if os.path.exists(config.metrics_path):
        os.remove(config.metrics_path)

    # Evaluate the model
    accuracy = evaluate_model(model, X_test, y_test)

    # Check that accuracy is perfect
    assert accuracy == 1.0

    # Cleanup
    if os.path.exists(config.metrics_path):
        os.remove(config.metrics_path)
