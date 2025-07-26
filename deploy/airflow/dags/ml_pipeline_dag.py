"""
ML Pipeline DAG for News Topic Classification

This DAG orchestrates the complete machine learning pipeline including:
1. Data download from Hugging Face datasets
2. Data preprocessing and train/test splitting
3. Feature engineering with TF-IDF vectorization
4. Model training using Logistic Regression
5. Model evaluation and metrics generation

Each task is designed to be idempotent and includes proper error handling
and logging for production reliability.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import sys
import logging
import os

# Add the project root to Python path for imports
sys.path.insert(0, os.environ.get("AIRFLOW_PROJECT_ROOT", "/opt/airflow"))


# Import ML pipeline modules
from src.download_data import download_raw_data
from src.data_preprocessing import preprocess_data
from src.feature_engineering import feature_engineering
from src.model_training import train_model
from src.evaluation import evaluate_model
from src.config import config

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


def download_data_task(**context):
    """
    Task to download AG News dataset from Hugging Face.

    Uses PythonOperator with provide_context=True to access task_instance
    for logging and XCom communication between tasks.
    """
    logger.info("Starting data download task")
    try:
        # Call the download_raw_data function from src module
        result = download_raw_data()
        logger.info("Data download completed successfully: %s", result)

        # Push result to XCom for downstream tasks
        context["task_instance"].xcom_push(
            key="raw_data_path", value=config.raw_data_path
        )
        return result
    except Exception as e:
        logger.error("Data download failed: %s", str(e))
        raise


def preprocess_data_task(**context):
    """
    Task to preprocess raw data and create train/test splits.

    Retrieves raw data path from upstream task via XCom and outputs
    paths to processed train/test files for downstream consumption.
    """
    logger.info("Starting data preprocessing task")
    try:
        # Get raw data path from upstream task
        raw_data_path = context["task_instance"].xcom_pull(
            task_ids="download_data", key="raw_data_path"
        )

        if not raw_data_path:
            raw_data_path = config.raw_data_path

        logger.info(f"Processing data from: {raw_data_path}")

        # Call preprocessing function
        train_path, test_path = preprocess_data(raw_data_path)
        logger.info(f"Preprocessing completed: train={train_path}, test={test_path}")

        # Push results to XCom
        context["task_instance"].xcom_push(key="train_path", value=train_path)
        context["task_instance"].xcom_push(key="test_path", value=test_path)

        return {"train_path": train_path, "test_path": test_path}
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


def pipeline_success_notification(**context):
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
# Each task wraps a function from the ML pipeline modules

download_task = PythonOperator(
    task_id="download_data",
    python_callable=download_data_task,
    dag=dag,
)

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

success_task = PythonOperator(
    task_id="pipeline_success",
    python_callable=pipeline_success_notification,
    provide_context=True,
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

# Define task dependencies using bit-shift operators
# This creates a linear pipeline with proper data flow
task_dependencies = (
    validate_environment
    >> download_task
    >> preprocess_task
    >> feature_engineering_task_op
    >> train_task
    >> evaluate_task
    >> success_task
)

# Alternative dependency definition using set_upstream/set_downstream:
# download_task.set_upstream(validate_environment)
# preprocess_task.set_upstream(download_task)
# feature_engineering_task_op.set_upstream(preprocess_task)
# train_task.set_upstream(feature_engineering_task_op)
# evaluate_task.set_upstream(train_task)
# success_task.set_upstream(evaluate_task)

# DAG documentation for Airflow UI
dag.doc_md = """
## ML Pipeline DAG for News Topic Classification

This DAG implements a complete machine learning pipeline for classifying news articles
into four categories: World, Sports, Business, and Sci/Tech.

### Pipeline Steps:
1. **Environment Validation**: Verify Python environment and dependencies
2. **Data Download**: Fetch AG News dataset from Hugging Face
3. **Data Preprocessing**: Clean data and create train/test splits
4. **Feature Engineering**: Create TF-IDF features from text data
5. **Model Training**: Train Logistic Regression classifier
6. **Model Evaluation**: Generate performance metrics and reports
7. **Pipeline Success**: Log results and handle completion

### Key Features:
- **Idempotent Tasks**: Each task can be safely re-run
- **Error Handling**: Comprehensive error handling with retries
- **XCom Communication**: Tasks pass data through Airflow's XCom system
- **Logging**: Detailed logging for debugging and monitoring
- **Manual Trigger**: Pipeline runs on-demand rather than scheduled

### Monitoring:
- Check task logs in Airflow UI for detailed execution information
- Monitor XCom values to track data flow between tasks
- Use Gantt chart view to analyze task execution times
"""
