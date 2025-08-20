"""
ML Pipeline DAG for News Topic Classification

This DAG orchestrates the complete machine learning pipeline including:
1. Data download from Hugging Face datasets
2. Data preprocessing and train/test splitting
3. Feature engineering with TF-IDF vectorization
4. Model training using Logistic Regression
5. Model evaluation and metrics generation
6. Drift detection and branching logic for retraining

Each task is designed to be idempotent and includes proper error handling
and logging for production reliability.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import sys
import logging
import os
import json

# Add the project root to Python path for imports
sys.path.insert(0, os.environ.get("AIRFLOW_PROJECT_ROOT", "/opt/airflow"))

# Import ML pipeline modules
from src.download_data import download_raw_data
from src.data_preprocessing import preprocess_data
from src.feature_engineering import feature_engineering
from src.model_training import train_model
from src.evaluation import evaluate_model
from src.drift_detection import detect_drift
from src.config import config

# MLflow setup
try:
    import mlflow

    mlflow.set_tracking_uri("http://mlflow:5000")
except ImportError:
    mlflow = None


# Import ML pipeline modules

# Configure logging for the DAG
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default arguments for all tasks in this DAG
default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,  # Retry failed tasks twice
    "retry_delay": timedelta(minutes=5),
    "catchup": False,  # Don't run historical DAG instances
}

# Define the DAG
dag = DAG(
    "ml_pipeline_news_classification",
    default_args=default_args,
    description="End-to-end ML pipeline for news topic classification",
    schedule_interval=None,  # Manual trigger only for now
    max_active_runs=1,  # Prevent concurrent runs
    tags=["ml", "news-classification", "nlp"],
)


def preprocess_data_task(**context):
    """
    Task to download data and preprocess raw data to create train/test splits.
    Combined download + preprocessing for HW3 structure.
    """
    logger.info("Starting data download and preprocessing task")
    try:
        # Step 1: Download data
        raw_data_path = download_raw_data()
        logger.info(f"Data downloaded to: {raw_data_path}")

        # Step 2: Preprocess data (includes drift generation)
        train_path, test_path = preprocess_data(raw_data_path)
        logger.info(f"Preprocessing completed: train={train_path}, test={test_path}")

        # Push results to XCom
        context["task_instance"].xcom_push(key="train_path", value=train_path)
        context["task_instance"].xcom_push(key="test_path", value=test_path)
        context["task_instance"].xcom_push(key="raw_data_path", value=raw_data_path)

        return {
            "train_path": train_path,
            "test_path": test_path,
            "raw_data_path": raw_data_path,
        }
    except Exception as e:
        logger.error(f"Data preprocessing failed: {str(e)}")
        raise


def feature_engineering_task(**context):
    """
    Task to perform feature engineering with TF-IDF vectorization.

    Creates TF-IDF features from preprocessed text data and saves
    the fitted vectorizer for consistent transform during inference.
    """
    logger.info("Starting feature engineering task")
    try:
        # Get train/test paths from upstream task
        train_path = context["task_instance"].xcom_pull(
            task_ids="preprocess_data", key="train_path"
        )
        test_path = context["task_instance"].xcom_pull(
            task_ids="preprocess_data", key="test_path"
        )

        if not train_path or not test_path:
            train_path = config.train_path
            test_path = config.test_path

        logger.info("Feature engineering on: train=%s, test=%s", train_path, test_path)

        # Call feature engineering function with paths
        X_train_tfidf, X_test_tfidf, y_train, y_test = feature_engineering(
            train_path, test_path
        )
        logger.info("Feature engineering completed")

        # Push paths to XCom for downstream tasks
        context["task_instance"].xcom_push(
            key="vectorizer_path", value=config.vectorizer_path
        )
        context["task_instance"].xcom_push(
            key="X_train_shape", value=X_train_tfidf.shape
        )
        context["task_instance"].xcom_push(key="X_test_shape", value=X_test_tfidf.shape)

        return {
            "vectorizer_path": config.vectorizer_path,
            "train_shape": X_train_tfidf.shape,
            "test_shape": X_test_tfidf.shape,
        }
    except Exception as e:
        logger.error("Feature engineering failed: %s", str(e))
        raise


def train_model_task(**context):
    """
    Task to train the machine learning model.

    Loads TF-IDF features and trains a Logistic Regression classifier,
    saving the trained model for evaluation and inference.
    """
    import joblib
    import pandas as pd

    logger.info("Starting model training task")
    try:
        # Get paths from upstream tasks
        train_path = context["task_instance"].xcom_pull(
            task_ids="preprocess_data", key="train_path"
        )
        vectorizer_path = context["task_instance"].xcom_pull(
            task_ids="feature_engineering", key="vectorizer_path"
        )

        if not train_path:
            train_path = config.train_path
        if not vectorizer_path:
            vectorizer_path = config.vectorizer_path

        logger.info("Training model with data from: %s", train_path)

        # Load training data
        train_df = pd.read_csv(train_path)

        # Load the fitted vectorizer
        vectorizer = joblib.load(vectorizer_path)

        # Transform training text to TF-IDF features
        X_train = vectorizer.transform(train_df["text"])
        y_train = train_df["label"]

        # Call model training function
        train_model(X_train, y_train)
        logger.info("Model training completed successfully")

        # Push model path to XCom
        context["task_instance"].xcom_push(key="model_path", value=config.model_path)

        return config.model_path
    except Exception as e:
        logger.error("Model training failed: %s", str(e))
        raise


def evaluate_model_task(**context):
    """
    Task to evaluate the trained model and generate metrics.

    Loads the trained model and vectorizer to evaluate performance
    on test data, generating classification metrics and reports.
    """
    import joblib
    import pandas as pd

    logger.info("Starting model evaluation task")
    try:
        # Get paths from upstream tasks
        test_path = context["task_instance"].xcom_pull(
            task_ids="preprocess_data", key="test_path"
        )
        model_path = context["task_instance"].xcom_pull(
            task_ids="train_model", key="model_path"
        )
        vectorizer_path = context["task_instance"].xcom_pull(
            task_ids="feature_engineering", key="vectorizer_path"
        )

        if not test_path:
            test_path = config.test_path
        if not model_path:
            model_path = config.model_path
        if not vectorizer_path:
            vectorizer_path = config.vectorizer_path

        logger.info("Evaluating model: %s", model_path)

        # Load test data
        test_df = pd.read_csv(test_path)

        # Load the trained model and vectorizer
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)

        # Transform test text to TF-IDF features
        X_test = vectorizer.transform(test_df["text"])
        y_test = test_df["label"]

        # Call evaluation function
        accuracy = evaluate_model(model, X_test, y_test)
        logger.info("Model evaluation completed with accuracy: %s", accuracy)

        # Push evaluation results to XCom
        context["task_instance"].xcom_push(key="accuracy", value=accuracy)
        context["task_instance"].xcom_push(
            key="metrics_path", value=config.metrics_path
        )

        return accuracy
    except Exception as e:
        logger.error("Model evaluation failed: %s", str(e))
        raise


def drift_detection_task(**context):
    """
    Task to detect data drift between reference and current datasets.
    """
    logger.info("Starting drift detection task")
    try:
        # Run drift detection
        drift_results = detect_drift("data/processed/test.csv", "data/drifted_test.csv")

        # Save results to file for branching task to read
        drift_report_path = "reports/drift_report.json"
        with open(drift_report_path, "w", encoding="utf-8") as f:
            json.dump(drift_results, f, indent=2)

        logger.info(f"Drift detection completed: {drift_results}")

        # Push to XCom as well
        context["task_instance"].xcom_push(
            key="drift_detected", value=drift_results["drift_detected"]
        )
        context["task_instance"].xcom_push(
            key="overall_drift_score", value=drift_results["overall_drift_score"]
        )

        return drift_results
    except Exception as e:
        logger.error(f"Drift detection failed: {str(e)}")
        raise


def branch_on_drift(**context):
    """
    Branch task that decides whether to retrain model or complete pipeline.
    """
    logger.info("Starting branching decision")
    try:
        # Read drift results from file
        drift_report_path = "reports/drift_report.json"
        with open(drift_report_path, "r", encoding="utf-8") as f:
            drift_results = json.load(f)

        drift_detected = drift_results.get("drift_detected", False)

        if drift_detected:
            logger.info("Drift detected - branching to retrain_model")
            return "retrain_model"
        else:
            logger.info("No drift detected - branching to pipeline_complete")
            return "pipeline_complete"
    except Exception as e:
        logger.error(f"Branching decision failed: {str(e)}")
        # Default to completion if can't read drift results
        return "pipeline_complete"


def retrain_model_task(**context):
    """
    Task to retrain model with original (non-drifted) data.
    """
    import joblib
    import pandas as pd

    logger.info("Starting model retraining due to drift detection")
    try:
        # Use original non-drifted data for retraining
        train_path = config.train_path
        vectorizer_path = config.vectorizer_path

        # Load training data
        train_df = pd.read_csv(train_path)

        # Load the fitted vectorizer
        vectorizer = joblib.load(vectorizer_path)

        # Transform training text to TF-IDF features
        X_train = vectorizer.transform(train_df["text"])
        y_train = train_df["label"]

        # Retrain the model
        train_model(X_train, y_train)

        logger.info("Model retraining completed successfully")
        return "retraining_complete"
    except Exception as e:
        logger.error(f"Model retraining failed: {str(e)}")
        raise


def pipeline_complete_task(**context):
    """
    Task to handle successful pipeline completion without retraining.
    """
    logger.info("ML Pipeline completed successfully without retraining!")

    # Get results from upstream tasks
    accuracy = context["task_instance"].xcom_pull(
        task_ids="evaluate_model", key="accuracy"
    )
    drift_detected = context["task_instance"].xcom_pull(
        task_ids="drift_detection", key="drift_detected"
    )

    logger.info(f"Final model accuracy: {accuracy}")
    logger.info(f"Drift detected: {drift_detected}")

    return {
        "status": "success",
        "accuracy": accuracy,
        "drift_detected": drift_detected,
        "timestamp": datetime.now().isoformat(),
    }
    """
    Task to handle successful pipeline completion.

    Logs final results and could send notifications or trigger
    downstream processes like model deployment.
    """
    logger.info("ML Pipeline completed successfully!")

    # Get final results from upstream tasks
    accuracy = context["task_instance"].xcom_pull(
        task_ids="evaluate_model", key="accuracy"
    )

    metrics_path = context["task_instance"].xcom_pull(
        task_ids="evaluate_model", key="metrics_path"
    )

    logger.info("Final model accuracy: %s", accuracy)
    logger.info("Metrics saved to: %s", metrics_path)

    return {
        "status": "success",
        "accuracy": accuracy,
        "metrics_path": metrics_path,
        "timestamp": datetime.now().isoformat(),
    }


# Define tasks using PythonOperator
# 5 primary tasks as required by HW3

preprocess_task = PythonOperator(
    task_id="preprocess_data",
    python_callable=preprocess_data_task,
    dag=dag,
)

feature_engineering_task_op = PythonOperator(
    task_id="feature_engineering",
    python_callable=feature_engineering_task,
    dag=dag,
)

train_task = PythonOperator(
    task_id="train_model",
    python_callable=train_model_task,
    dag=dag,
)

evaluate_task = PythonOperator(
    task_id="evaluate_model",
    python_callable=evaluate_model_task,
    dag=dag,
)

drift_detection_task_op = PythonOperator(
    task_id="drift_detection",
    python_callable=drift_detection_task,
    dag=dag,
)

# Branching task
branch_task = BranchPythonOperator(
    task_id="branch_on_drift",
    python_callable=branch_on_drift,
    dag=dag,
)

# End tasks
retrain_task = PythonOperator(
    task_id="retrain_model",
    python_callable=retrain_model_task,
    dag=dag,
)

complete_task = PythonOperator(
    task_id="pipeline_complete",
    python_callable=pipeline_complete_task,
    dag=dag,
)

# Optional: Add a task to validate the environment setup
validate_environment = BashOperator(
    task_id="validate_environment",
    bash_command="""
    echo "Validating ML pipeline environment..."
    python -c "
import sys
import os
print(f'Python version: {sys.version}')
import src.config
print(f'Config loaded successfully')
print(f'Working directory: {os.getcwd()}')
print('Environment validation completed')
    "
    """,
    dag=dag,
)

# Define task dependencies for HW3 branching structure
# preprocess_data >> feature_engineering >> train_model >> evaluate_model >> drift_detection >> branch_on_drift >> [retrain_model, pipeline_complete]

task_dependencies = (
    preprocess_task
    >> feature_engineering_task_op
    >> train_task
    >> evaluate_task
    >> drift_detection_task_op
    >> branch_task
)

# Branching dependencies
branch_task >> [retrain_task, complete_task]

# Alternative dependency definition using set_upstream/set_downstream:
# download_task.set_upstream(validate_environment)
# preprocess_task.set_upstream(download_task)
# feature_engineering_task_op.set_upstream(preprocess_task)
# train_task.set_upstream(feature_engineering_task_op)
# evaluate_task.set_upstream(train_task)
# success_task.set_upstream(evaluate_task)

# DAG documentation for Airflow UI
dag.doc_md = """
## ML Pipeline DAG for News Topic Classification with Drift Detection

This DAG implements a complete machine learning pipeline for classifying news articles
into four categories: World, Sports, Business, and Sci/Tech, with data drift detection and branching logic.

### Pipeline Steps:
1. **Data Preprocessing**: Clean data and create train/test splits (generates drifted datasets)
2. **Feature Engineering**: Create TF-IDF features from text data
3. **Model Training**: Train Logistic Regression classifier with MLflow tracking
4. **Model Evaluation**: Generate performance metrics and model registration
5. **Drift Detection**: Analyze data drift using Evidently
6. **Branching**: Decide whether to retrain or complete based on drift detection
7. **Retrain Model**: Retrain with original data if drift detected
8. **Pipeline Complete**: Completion task if no drift detected

### Key Features:
- **MLflow Integration**: Experiment tracking and model registration
- **Drift Detection**: Automated data drift analysis with Evidently
- **Branching Logic**: Conditional retraining based on drift detection
- **Error Handling**: Comprehensive error handling with retries
- **XCom Communication**: Tasks pass data through Airflow's XCom system

### Monitoring:
- MLflow UI available at http://mlflow:5000
- Check reports/drift_report.json for drift analysis results
- Monitor task logs for detailed execution information
"""
