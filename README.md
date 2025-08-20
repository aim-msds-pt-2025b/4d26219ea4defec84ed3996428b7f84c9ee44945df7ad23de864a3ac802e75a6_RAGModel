# News Topic Classification Project

This project builds a complete, production-driven ML pipeline to classify news articles into four topics with comprehensive testing, logging, error handling, Docker containerization, and Apache Airflow orchestration.

## Project Overview

The goal is to classify news headlines from the AG News dataset into one of four categories: World, Sports, Business, or Sci/Tech. This is a multi-class text classification problem implemented as a containerized, orchestrated ML pipeline.

**Docker Integration**: Docker ensures environment consistency through immutable infrastructure, allowing the pipeline to run identically across development, testing, and production environments. This eliminates "works on my machine" issues and provides reproducible deployments.

**Airflow Orchestration**: Apache Airflow enables scalable orchestration with built-in retry mechanisms for flaky tasks, dependency management, and monitoring capabilities. The DAG-based approach allows for complex workflow dependencies while maintaining clear separation of concerns between tasks.

## Key Features

- **Comprehensive Testing**: Unit tests for all modules with pytest
- **Logging**: Structured logging throughout the pipeline
- **Error Handling**: Robust error handling with custom decorators
- **Configuration Management**: Centralized configuration for easy maintenance
- **Data Validation**: Input validation for data quality assurance
- **Pre-commit Hooks**: Automated code quality checks including pytest execution
- **Docker Containerization**: Reproducible environment with multi-stage builds
- **Airflow Orchestration**: Workflow automation with task dependencies and monitoring
- **Volume Management**: Persistent data storage across container runs

## How to Get the Data

The AG News dataset is downloaded automatically when you run the pipeline. The `src/download_data.py` script uses the Hugging Face `datasets` library to fetch and save the data to the `data/` directory. No manual download is required.

## Setup Instructions

This project supports both traditional Python environments and containerized deployment with Docker and Airflow.

### Prerequisites

1. **Install Docker Desktop**: Download from [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)
   - Verify installation: `docker --version`
   - Ensure Docker daemon is running

2. **Install Docker Compose**: Usually included with Docker Desktop
   - Verify installation: `docker-compose --version`

### Local Development Setup

If you want to run the pipeline locally without containers:

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```

2. **Install uv:** If you don't have it, follow the official instructions at [astral.sh](https://astral.sh/uv/install.sh).

3. **Create the virtual environment and install dependencies:**
   ```bash
   # Create and activate the virtual environment
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Install dependencies
   uv sync
   ```

4. **Set up pre-commit hooks:**
   ```bash
   pre-commit install
   ```

### Docker Setup

#### Build the ML Pipeline Container

1. **Build the Docker image:**
   ```bash
   # Using the provided Dockerfile with multi-stage builds
   docker build -t 4d26219ea4defec84ed3996428b7f84c9ee44945df7ad23de864a3ac802e75a6_RAGModel-ml-pipeline .
   ```

2. **Run the containerized pipeline:**
   ```bash
   # Run with volume mounts for data persistence
   docker run --rm \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/models:/app/models \
     -v $(pwd)/reports:/app/reports \
     4d26219ea4defec84ed3996428b7f84c9ee44945df7ad23de864a3ac802e75a6_RAGModel-ml-pipeline
   ```

#### Airflow Setup with Docker Compose

1. **Initialize Airflow environment:**
   ```bash
   # Create necessary directories
   mkdir -p deploy/airflow/logs deploy/airflow/plugins

   # Set the Airflow user ID (Linux/Mac)
   echo -e "AIRFLOW_UID=$(id -u)" > .env

   # On Windows, the default UID in .env should work
   ```

2. **Start Airflow services:**
   ```bash
   # Start all services (PostgreSQL, Redis, Airflow webserver, scheduler)
   docker-compose up -d

   # Wait for services to be healthy (may take 2-3 minutes)
   docker-compose logs -f
   ```

3. **Access Airflow UI:**
   - Open [http://localhost:8080](http://localhost:8080)
   - Login with credentials: `airflow` / `airflow`
   - Find the `ml_pipeline_news_classification` DAG

4. **Run the ML Pipeline DAG:**
   - Enable the DAG by toggling the switch
   - Click "Trigger DAG" to start the pipeline
   - Monitor progress in the Graph View or Gantt Chart

5. **Stop Airflow services:**
   ```bash
   docker-compose down

   # To remove volumes and start fresh
   docker-compose down -v
   ```

## How to Run the Pipeline

### Option 1: Local Execution (Traditional)

To execute the entire pipeline from data download to model evaluation, run the following command from the project root:

```bash
python src/run_pipeline.py
```

### Option 2: Docker Container

Run the pipeline in an isolated container environment:

```bash
# Run the containerized pipeline
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/reports:/app/reports \
  4d26219ea4defec84ed3996428b7f84c9ee44945df7ad23de864a3ac802e75a6_RAGModel-ml-pipeline
```

### Option 3: Airflow Orchestration (Recommended)

Use the Airflow DAG for production-grade workflow management:

1. **Start Airflow:** `docker-compose up -d`
2. **Access UI:** [http://localhost:8080](http://localhost:8080)
3. **Enable DAG:** Toggle the `ml_pipeline_news_classification` DAG
4. **Trigger Run:** Click "Trigger DAG" to start execution
5. **Monitor:** Watch progress in Graph View with detailed task logs

### Pipeline Outputs

All methods produce the same outputs:
- `data/raw/ag_news_raw.csv` (raw downloaded data)
- `data/processed/train.csv` and `data/processed/test.csv` (processed data)
- `models/model.pkl` (the trained classifier)
- `models/tfidf_vectorizer.pkl` (the fitted vectorizer)
- `reports/metrics.txt` (containing the model's performance metrics)
- `pipeline.log` (detailed pipeline execution log)

## Testing

The project includes comprehensive test coverage for all modules:

```bash
# Run all tests
python -m pytest

# Run tests with verbose output
python -m pytest -v

# Run tests with coverage
python -m pytest --cov=src

# Run specific test file
python -m pytest tests/test_data.py

# Run tests with coverage report
python -m pytest --cov=src

# Run tests with detailed coverage report showing missing lines
python -m pytest --cov=src --cov-report=term-missing
```

### Testing the Complete HW3 Pipeline

To validate the MLflow integration and drift detection:

```bash
# Test the complete pipeline with drift detection
python src/run_pipeline.py

# Verify MLflow tracking is working
docker-compose up -d
# Navigate to http://localhost:5001 to see experiment tracking

# Test the Airflow DAG
docker-compose up -d
# Navigate to http://localhost:8080, enable and trigger ml_pipeline_news_classification

# Test drift detection specifically
python -c "
from src.drift_detection import detect_drift
import pandas as pd
# This should detect drift and demonstrate the error raising
try:
    detect_drift('data/processed/train.csv', 'data/processed/test.csv')
    print('No drift detected')
except ValueError as e:
    print(f'Drift detected: {e}')
"
```
### Test Structure

- `tests/test_data.py`: Tests for data preprocessing functionality
- `tests/test_feature_engineering.py`: Tests for TF-IDF vectorization
- `tests/test_model_training.py`: Tests for model training
- `tests/test_evaluation.py`: Tests for model evaluation
- `tests/test_pipeline.py`: Integration tests for the complete pipeline

## Model Drift Detection

The model drift detection system monitors data distribution changes that could degrade model performance over time. Our implementation uses Evidently AI's DataDriftPreset to detect feature-level drift by comparing statistical distributions between training and incoming data. Owing to use of TF-IDF, it was more difficult to induce a clearly noticeable drift that Evidently would pick up. For now, we've leveraged a custom synthetic marker to be able to run the test. Despite attempting a more realistic attempt of inducing drift, flipping 25-35% of labels, adding multiple punctuations or mutating text by repeating last word, it was difficult to flag drift without more crude methods such as length.

## MLflow Integration

MLflow provides comprehensive experiment tracking and model lifecycle management throughout our ML pipeline. The integration automatically logs model parameters, metrics, and artifacts during training, enabling reproducible experiments and model versioning. Our custom PyFunc model wrapper ensures seamless deployment compatibility while maintaining access to preprocessing artifacts like TF-IDF vectorizers. Models achieving accuracy scores above 0.8 are automatically registered to the MLflow Model Registry with appropriate versioning and staging labels.

### MLflow Configuration

Our implementation includes:
- **Tracking URI**: Set to `http://localhost:5000` for containerized deployment
- **Custom PyFunc Model**: Wrapper class implementing `load_context()` and `predict()` methods
- **Experiment Logging**: 3 hyperparameters logged for Logistic Regression:
  - `max_iter`: Maximum iterations (1000)
  - `random_state`: Random seed for reproducibility (42)
  - `penalty`: Regularization type ("l2")
- **Model Registration**: Automatic registration when accuracy > 0.8
- **PostgreSQL Backend**: Persistent experiment metadata storage

### Verification Commands

To verify MLflow integration is working correctly:

```bash
# Start MLflow services
docker-compose up -d

# Verify MLflow UI is accessible (should return HTML)
curl http://localhost:5000

# Run pipeline to generate experiments
python src/run_pipeline.py

# Access MLflow UI for experiment tracking
# Navigate to http://localhost:5000 in your browser
```

### Hyperparameter Selection

For our **Logistic Regression** classification model, we log the following 3 hyperparameters to MLflow:

1. **`max_iter`** (1000): Maximum number of iterations for the solver to converge. This is crucial for text classification where high-dimensional TF-IDF features may require more iterations to reach optimal weights.

2. **`random_state`** (42): Random seed for reproducible results across runs. Essential for experiment tracking and ensuring consistent model performance comparisons in MLflow.

3. **`penalty`** ("l2"): Regularization type to prevent overfitting. L2 regularization is well-suited for text classification tasks with sparse TF-IDF features, helping to generalize better on unseen news articles.

These parameters are specifically chosen for text classification with TF-IDF features and logged to MLflow for experiment reproducibility and hyperparameter tracking.

## Code Structure

### Core Modules

- `src/config.py`: Centralized configuration management
- `src/utils.py`: Utility functions for logging, error handling, and validation
- `src/download_data.py`: Data download from Hugging Face datasets
- `src/data_preprocessing.py`: Data cleaning and train/test splitting
- `src/feature_engineering.py`: TF-IDF vectorization
- `src/model_training.py`: Logistic regression model training
- `src/evaluation.py`: Model evaluation and metrics calculation
- `src/run_pipeline.py`: Main pipeline orchestration

### Key Implementations

1. **Configuration Management**: All settings centralized in `config.py`
2. **Logging**: Structured logging with file and console output
3. **Error Handling**: Comprehensive error handling with custom decorators
4. **Data Validation**: Input validation for data quality assurance
5. **Type Hints**: Improved code documentation and IDE support
6. **Testing**: Comprehensive test suite with fixtures and mocking
7. **Documentation**: Enhanced docstrings and comments

## Folder Structure

The project uses a structured layout to ensure clarity, reproducibility, and support for containerized, orchestrated workflows with MLflow experiment tracking and model drift detection capabilities:

```
├── data/
│   ├── raw/                 # Bronze: Raw, untouched data
│   └── processed/           # Silver: Cleaned and split data
├── models/                  # Gold: Trained models and artifacts
├── reports/                 # Gold: Performance metrics and drift reports
├── mlflow/                  # MLflow experiment tracking and artifact storage
│   ├── runs/               # Experiment run metadata and artifacts
│   └── artifacts/          # Model artifacts and experiment data
├── vector_store/           # Vector database storage for similarity search
├── src/                     # Python source code
│   ├── config.py           # Configuration management
│   ├── utils.py            # Utility functions
│   ├── download_data.py    # Data download module
│   ├── data_preprocessing.py # Data preprocessing
│   ├── feature_engineering.py # Feature engineering
│   ├── model_training.py   # Model training with MLflow integration
│   ├── evaluation.py       # Model evaluation and registration
│   ├── drift_detection.py  # Data drift monitoring with Evidently
│   └── run_pipeline.py     # Main pipeline script with drift handling
├── tests/                   # Test suite
│   ├── test_data.py        # Data processing tests
│   ├── test_feature_engineering.py # Feature engineering tests
│   ├── test_model_training.py # Model training tests
│   ├── test_evaluation.py  # Evaluation tests
│   └── test_pipeline.py    # Integration tests
├── deploy/                  # Containerization and orchestration
│   ├── docker/             # Docker build artifacts and configs
│   └── airflow/            # Airflow DAGs and configuration
│       ├── dags/           # Airflow DAG definitions with drift detection
│       └── logs/           # Airflow execution logs
├── config/                  # Configuration files
│   └── airflow.cfg         # Airflow scheduler settings
├── Dockerfile              # Multi-stage container definition
├── Dockerfile.mlflow       # MLflow service container
├── docker-compose.yml      # Multi-service orchestration (Airflow + MLflow)
├── .dockerignore           # Docker build context optimization
├── pyproject.toml          # Project dependencies and metadata
├── .pre-commit-config.yaml # Pre-commit hooks configuration
└── README.md               # This file
```

**Key Design Decisions:**

- **`mlflow/` Directory**: Dedicated storage for MLflow experiment tracking, separating metadata and artifacts from core pipeline data for better organization and backup strategies
- **Enhanced `reports/` Directory**: Now includes both evaluation metrics and drift detection reports, providing comprehensive model monitoring capabilities
- **`deploy/` Directory**: Isolating DAGs in `airflow/dags/` ensures modularity, allowing independent testing of workflow tasks without affecting the core ML code
- **Volume Mounts**: Separate `data/`, `models/`, `reports/`, and `mlflow/` directories enable persistent storage across container runs while maintaining clear data lineage
- **Multi-service Architecture**: Separate Dockerfile for MLflow enables independent scaling and updates of experiment tracking infrastructure

## Pre-commit Configuration

This project uses `pre-commit` to enforce code quality and consistency. The following hooks are configured:

1. **Ruff**: Fast Python linter and formatter for code quality
2. **Standard Hooks**: File formatting (`trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-dockerfile`)
3. **Hadolint**: Dockerfile linting for best practices and security vulnerabilities (prevents issues like unnecessary root privileges)
4. **yamllint**: YAML file validation for docker-compose.yml and other configuration files
5. **Pytest**: All tests must pass before commits are allowed

The pre-commit hooks ensure that:
- Code is properly formatted and linted
- All tests pass before committing
- Files are properly formatted
- Docker and YAML files follow best practices
- Security vulnerabilities in containers are detected early

## Configuration

The project uses a centralized configuration system in `src/config.py`:

```python
# Key configuration parameters
TFIDF_MAX_FEATURES = 5000
TFIDF_STOP_WORDS = "english"
TEST_SIZE = 0.2
RANDOM_STATE = 42
```

This makes it easy to modify parameters without changing code throughout the project.

## Error Handling and Logging

The pipeline includes comprehensive error handling:

- **Custom Decorators**: `@handle_errors` decorator for consistent error handling
- **File Validation**: Checks for file existence before processing
- **Data Validation**: Validates DataFrame structure and content
- **Logging**: Structured logging with timestamps and log levels

## Docker Integration

### Dockerfile Design

The project uses a **multi-stage Dockerfile** with the following key features:

```dockerfile
# Stage 1: Builder - Install dependencies with uv
FROM python:3.12-slim as builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
# ... dependency installation

# Stage 2: Runtime - Minimal production image
FROM python:3.12-slim as runtime
# ... copy only necessary artifacts
```

**Key Design Decisions:**

1. **Multi-stage builds**: Minimize image size by separating build dependencies from runtime, reducing layers for faster pulls in CI/CD
2. **UV package manager**: Fast dependency resolution and installation compared to pip
3. **Non-root user**: Security best practice to avoid running containers as root
4. **Health checks**: Built-in container health monitoring
5. **Volume mounts**: Persistent data storage for `data/`, `models/`, and `reports/`

### Build and Run Commands

```bash
# Build the image with proper tagging
docker build -t 4d26219ea4defec84ed3996428b7f84c9ee44945df7ad23de864a3ac802e75a6_RAGModel-ml-pipeline .

# Run with volume mounts for data persistence
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/reports:/app/reports \
  4d26219ea4defec84ed3996428b7f84c9ee44945df7ad23de864a3ac802e75a6_RAGModel-ml-pipeline
```

### Volume Strategy

- **Data Persistence**: Host directories mounted to container paths ensure data survives container restarts
- **Model Storage**: Trained models persist outside containers for reuse and deployment
- **Log Access**: Pipeline logs accessible from host for debugging and monitoring

## Airflow DAG

### DAG Structure

The `ml_pipeline_news_classification` DAG orchestrates the complete ML pipeline with drift detection and conditional retraining. The DAG implements a branching workflow that automatically handles model drift scenarios:

```python
# HW3 Task Dependencies with Branching Logic
preprocess_data >> feature_engineering >> train_model >>
evaluate_model >> drift_detection >> branch_on_drift >> [retrain_model, pipeline_complete]
```

**Task Definitions:**

1. **`preprocess_data`**: Download AG News dataset and create train/test splits (PythonOperator)
2. **`feature_engineering`**: Create TF-IDF features from text data with MLflow logging (PythonOperator)
3. **`train_model`**: Train Logistic Regression with MLflow experiment tracking (PythonOperator)
4. **`evaluate_model`**: Generate performance metrics and register model if accuracy > 0.8 (PythonOperator)
5. **`drift_detection`**: Analyze data drift using Evidently and determine retraining needs (PythonOperator)
6. **`branch_on_drift`**: BranchPythonOperator that routes to retraining or completion based on drift results
7. **`retrain_model`**: Conditional task that retrains the model when drift is detected (PythonOperator)
8. **`pipeline_complete`**: Final task for successful pipeline completion without retraining (PythonOperator)

### Key Features

- **MLflow Integration**: All model training and evaluation tasks log to MLflow with automatic model registration
- **Drift Detection**: Evidently AI monitors feature-level drift and triggers retraining when thresholds are exceeded
- **Conditional Branching**: BranchPythonOperator enables intelligent workflow routing based on drift analysis
- **XCom Communication**: Tasks pass data paths and results through Airflow's XCom system
- **Error Handling**: Comprehensive try-catch blocks with detailed logging and MLflow error tracking
- **Retry Logic**: Automatic retry on failure (2 retries with 5-minute delays)
- **Model Registry**: Automatic model registration for models exceeding accuracy thresholds

### Scheduling Rationale

The DAG uses `schedule_interval=None` for manual triggering because:
- ML model training should be triggered based on data availability or performance degradation
- Drift detection requires careful analysis rather than blind retraining on schedules
- Allows for controlled execution during development and testing
- Easy to modify for production scheduling (daily, weekly, etc.) with drift-aware triggers

### Monitoring DAGs in Airflow UI

- **Graph View**: Visual representation of branching logic and task dependencies
- **Gantt Chart**: Timeline view showing conditional execution paths
- **Task Logs**: Detailed logs for each task execution with MLflow tracking URLs
- **XCom Browser**: Inspect drift detection results and model performance data
- **MLflow Links**: Direct links to experiment tracking and model registry from task logs

## Reflection

Building this improved pipeline taught me several key lessons:

1. **Read the requirements**: You may have noticed that this repo says RAG model when it is not a RAG model. That's because I initially thought I could just adapt a portion of our capstone project for this homework to hit two birds with one stone. I even thought it would be doing testing also for our capstone. Only when I read the actual requirements did I realize that I actually had to use a classical ML model for submission, hence this mishmash to hastily readapt my existing code.
2. **Testing is Essential**: Comprehensive tests catch issues early and make refactoring safer, especially when identify what edge cases or branches you haven't looked too hard into.
3. **Configuration Management**: Centralizing configuration makes the code more maintainable, and will make it easier to pass these to differing modules.
4. **Logging vs Print**: Proper logging provides better debugging and monitoring capabilities, especially when the pipeline isn't always idempotent.
5. **Error Handling**: Explicit error handling makes the pipeline more robust
6. **Pre-commit Hooks**: Automated quality checks prevent issues from reaching the repository and running the pytest would prevent you from submitting bad code.
7. **Modular design**: Breaking things to modules helps it easier to keep track of dependencies and avoid a really large monolithic structure codebase. It allso helps with using LLMs in debugging and doing quick iterations as this makes the code fit within the context window.

### Docker and Airflow Challenge

**Challenge**: One significant challenge was handling Python path issues in Airflow tasks when importing custom modules from the `src/` directory. Initially, the DAG couldn't import functions from the ML pipeline modules because the Python path wasn't properly configured in the containerized Airflow environment.

**Resolution**: I resolved this by explicitly adding the project root to the Python path in the DAG (`sys.path.insert(0, '/opt/airflow')`) and ensuring the source code was properly mounted as volumes in the docker-compose.yml. Additionally, I had to modify the task functions to handle data loading within each task rather than relying on the original function signatures, which improved DAG idempotency and made data flow between tasks more explicit through XCom usage. This taught me the importance of designing functions with containerized orchestration in mind from the beginning.
