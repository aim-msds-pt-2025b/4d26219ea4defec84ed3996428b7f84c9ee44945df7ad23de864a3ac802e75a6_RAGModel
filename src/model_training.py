import logging
import os
from sklearn.linear_model import LogisticRegression
import joblib
from src.config import config
from src.utils import handle_errors
import pandas as pd
import numpy as np

# MLflow optional import
mlflow = None
try:
    import mlflow
    import mlflow.pyfunc
    import mlflow.sklearn
except ImportError:
    mlflow = None


class CustomMLModel:
    """
    Custom MLflow PyFunc model wrapper for your trained model.
    Encapsulates preprocessing and prediction logic.
    """

    def __init__(self):
        self.model = None
        self.preprocessor = None  # scaler, encoder, etc.
        self.feature_names = None

    def load_context(self, context):
        """Load model artifacts from MLflow context."""
        self.model = joblib.load(context.artifacts["model"])
        # Load preprocessor if exists
        if "preprocessor" in context.artifacts:
            self.preprocessor = joblib.load(context.artifacts["preprocessor"])
        # Load feature names
        if "feature_names" in context.artifacts:
            with open(context.artifacts["feature_names"], "r") as f:
                self.feature_names = [line.strip() for line in f.readlines()]

    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_context first.")

        # Apply preprocessing if available
        if self.preprocessor:
            processed_input = self.preprocessor.transform(model_input)
        else:
            processed_input = (
                model_input.values if hasattr(model_input, "values") else model_input
            )

        # Make predictions
        predictions = self.model.predict(processed_input)
        return predictions


# Create the proper MLflow PyFunc model if available
if mlflow is not None:
    try:
        from typing import Any
        import mlflow.pyfunc

        # This creates a proper MLflow PyFunc model class
        class CustomMLModelPyFunc(mlflow.pyfunc.PythonModel):
            """MLflow-compatible custom model wrapper"""

            def __init__(self):
                self.model = None
                self.preprocessor = None
                self.feature_names = None

            def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
                """Load model artifacts from MLflow context."""
                self.model = joblib.load(context.artifacts["model"])
                if "preprocessor" in context.artifacts:
                    self.preprocessor = joblib.load(context.artifacts["preprocessor"])
                if "feature_names" in context.artifacts:
                    with open(context.artifacts["feature_names"], "r") as f:
                        self.feature_names = [line.strip() for line in f.readlines()]

            def predict(
                self,
                context: mlflow.pyfunc.PythonModelContext,
                model_input: Any,
                params: Any = None,
            ) -> Any:
                """Make predictions using the trained model."""
                if self.model is None:
                    raise ValueError("Model not loaded. Call load_context first.")

                if self.preprocessor:
                    processed_input = self.preprocessor.transform(model_input)
                else:
                    processed_input = (
                        model_input.values
                        if hasattr(model_input, "values")
                        else model_input
                    )

                predictions = self.model.predict(processed_input)
                return predictions

        # Use the MLflow-compatible version
        CustomMLModel = CustomMLModelPyFunc  # type: ignore
    except (AttributeError, ImportError):
        # Fall back to basic version if MLflow PythonModel not available
        pass


@handle_errors
def train_model(X_train, y_train):
    """
    Trains a Logistic Regression model and saves it.

    This function trains a logistic regression classifier on the provided
    training data and saves the trained model to disk.

    Args:
        X_train: Training features (typically TF-IDF transformed text).
        y_train: Training labels.

    Returns:
        sklearn.linear_model.LogisticRegression: The trained model.

    Raises:
        Exception: If model training or saving fails.
    """
    global mlflow  # Access the global mlflow variable

    logger = logging.getLogger(__name__)
    logger.info("Starting model training...")

    # Initialize and train the model
    random_state = config.RANDOM_STATE
    model = LogisticRegression(max_iter=1000, random_state=random_state)

    # MLflow tracking
    if mlflow is not None:
        try:
            mlflow.set_tracking_uri(
                os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
            )

            # Create or get experiment
            experiment_name = "news_classification"
            try:
                mlflow.create_experiment(experiment_name)
            except Exception:
                # Experiment already exists or other MLflow exception
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if not experiment:
                    # Fallback to default experiment
                    experiment_name = "Default"

            mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name="train_logreg"):
                # Log 3 hyperparameters (adapted for logistic regression)
                mlflow.log_param("max_iter", 1000)
                mlflow.log_param("random_state", random_state)
                mlflow.log_param("penalty", "l2")

                model.fit(X_train, y_train)

                # Save model locally first
                joblib.dump(model, config.model_path)

                # Log custom PyFunc model with artifacts
                artifacts = {"model": config.model_path}

                # Create custom model instance
                custom_model = CustomMLModelPyFunc()

                # Create a sample input for model signature
                import pandas as pd

                sample_input = pd.DataFrame(
                    X_train[:1].toarray()
                    if hasattr(X_train, "toarray")
                    else X_train[:1]
                )

                # Log the model using custom PyFunc wrapper (as required by HW3)
                try:
                    mlflow.pyfunc.log_model(
                        name="model",  # Use 'name' instead of deprecated 'artifact_path'
                        python_model=custom_model,
                        artifacts=artifacts,
                        input_example=sample_input,  # Add input example for signature
                    )
                except Exception as e:
                    logger.warning(f"Custom PyFunc model logging failed: {e}")
                    # Fallback: log as sklearn model
                    try:
                        import mlflow.sklearn  # type: ignore

                        mlflow.sklearn.log_model(  # type: ignore
                            model,
                            name="model_sklearn",  # Use 'name' instead of 'artifact_path'
                            registered_model_name=None,
                            input_example=sample_input,
                        )
                    except (AttributeError, ImportError):
                        # Last fallback: just log as artifact
                        mlflow.log_artifact(
                            config.model_path, artifact_path="model_artifacts"
                        )

                # Log model file as artifact for backup
                mlflow.log_artifact(config.model_path, artifact_path="model_artifacts")
        except Exception as e:
            logger.warning(
                f"MLflow logging failed: {e}. Continuing without MLflow tracking."
            )
            model.fit(X_train, y_train)
            # Save the model
            joblib.dump(model, config.model_path)
    else:
        model.fit(X_train, y_train)
        # Save the model
        joblib.dump(model, config.model_path)

    logger.info("Model training complete.")
    logger.info(
        "Model trained on %d samples with %d features",
        X_train.shape[0],
        X_train.shape[1],
    )

    # Ensure models directory exists
    config.__post_init__()

    if mlflow is None:
        # Save the model if not already saved in MLflow block
        joblib.dump(model, config.model_path)

    logger.info("Model saved to %s", config.model_path)

    return model


@handle_errors
def register_model_if_threshold_met(accuracy: float, run_id: str | None = None):
    """
    Register model if performance threshold is met.

    Args:
        accuracy: Model accuracy score
        run_id: MLflow run ID (if None, uses active run)

    Returns:
        bool: True if model was registered, False otherwise
    """
    global mlflow  # Access the global mlflow variable

    logger = logging.getLogger(__name__)

    # Performance threshold for classification
    ACCURACY_THRESHOLD = 0.8

    if accuracy > ACCURACY_THRESHOLD:
        if mlflow is not None:
            try:
                if run_id is None and mlflow.active_run():
                    active_run = mlflow.active_run()
                    if active_run and active_run.info:
                        run_id = active_run.info.run_id

                if run_id:
                    model_uri = f"runs:/{run_id}/model"
                    model_name = "news_classification_model"

                    registered_model = mlflow.register_model(
                        model_uri=model_uri, name=model_name
                    )

                    logger.info(
                        f"Model registered: {model_name} (version {registered_model.version})"
                    )
                    return True
                else:
                    logger.warning("No active MLflow run found for model registration")
            except Exception as e:
                logger.error(f"Model registration failed: {str(e)}")

    else:
        logger.info(
            f"Model accuracy {accuracy:.4f} below threshold {ACCURACY_THRESHOLD}, not registering"
        )

    return False


if __name__ == "__main__":
    # Example usage for testing
    print("This module should be run as part of the full pipeline.")
    print("Use run_pipeline.py to execute the complete ML pipeline.")
