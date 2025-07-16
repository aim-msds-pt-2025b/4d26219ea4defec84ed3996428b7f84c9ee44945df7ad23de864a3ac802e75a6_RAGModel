# tests/test_model_training.py

import os
import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.model_training import train_model
from src.config import config


@pytest.fixture
def sample_training_data():
    """Create sample training data for testing."""
    np.random.seed(42)
    X_train = np.random.rand(100, 50)  # 100 samples, 50 features
    y_train = np.random.randint(0, 4, 100)  # 4 classes
    return X_train, y_train


def test_train_model_saves_model(sample_training_data):
    """Test that model training saves the model correctly."""
    X_train, y_train = sample_training_data

    # Clean up any existing model
    if os.path.exists(config.model_path):
        os.remove(config.model_path)

    # Train the model
    model = train_model(X_train, y_train)

    # Check that model was saved
    assert os.path.exists(config.model_path)

    # Check that returned model is correct type
    assert isinstance(model, LogisticRegression)

    # Check that model can make predictions
    predictions = model.predict(X_train[:5])
    assert len(predictions) == 5
    assert all(pred in [0, 1, 2, 3] for pred in predictions)

    # Cleanup
    if os.path.exists(config.model_path):
        os.remove(config.model_path)


def test_train_model_loads_correctly(sample_training_data):
    """Test that the saved model can be loaded and used."""
    X_train, y_train = sample_training_data

    # Clean up any existing model
    if os.path.exists(config.model_path):
        os.remove(config.model_path)

    # Train and save the model
    original_model = train_model(X_train, y_train)

    # Load the saved model
    import joblib

    loaded_model = joblib.load(config.model_path)

    # Check that loaded model makes same predictions
    test_sample = X_train[:5]
    original_predictions = original_model.predict(test_sample)
    loaded_predictions = loaded_model.predict(test_sample)

    assert np.array_equal(original_predictions, loaded_predictions)

    # Cleanup
    if os.path.exists(config.model_path):
        os.remove(config.model_path)


def test_train_model_with_invalid_data():
    """Test that model training handles invalid data appropriately."""
    # Test with empty data
    X_empty = np.array([]).reshape(0, 10)
    y_empty = np.array([])

    with pytest.raises(ValueError):
        train_model(X_empty, y_empty)

    # Test with mismatched X and y shapes
    X_train = np.random.rand(10, 5)
    y_train = np.random.randint(0, 4, 15)  # Different number of samples

    with pytest.raises(ValueError):
        train_model(X_train, y_train)
