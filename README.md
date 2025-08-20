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
- **Docker Containerization**: Reproducible environment using an Airflow-based image
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

1. **Build the complete stack:**
   ```bash
   # Build all services including MLflow and Airflow
   docker compose build --no-cache

   # OR build individual services
   docker build -t ml-pipeline:latest .
   docker build -f Dockerfile.mlflow -t mlflow-server .
   ```

2. **Start the orchestrated stack:**
   ```bash
   # Start all services (PostgreSQL, Redis, Airflow, MLflow)
   docker compose up -d

   # Check service health
   docker compose ps

   # View logs if needed
   docker compose logs -f
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
   # Start all services (PostgreSQL with separated databases, Redis, Airflow webserver/scheduler, MLflow)
   docker compose up -d

   # Wait for services to be healthy (may take 2-3 minutes)
   docker compose ps

   # Check logs if services aren't starting
   docker compose logs airflow-scheduler
   docker compose logs mlflow
   ```

3. **Access Airflow UI:**
   - Open [http://localhost:8080](http://localhost:8080)
   - Login with credentials: `airflow` / `airflow`
   - Find the `ml_pipeline_news_classification` DAG

4. **Access MLflow UI:**
   - Open [http://localhost:5000](http://localhost:5000)
   - View experiment tracking and model registry

5. **Run the ML Pipeline DAG:**
   - Enable the DAG by toggling the switch
   - Click "Trigger DAG" to start the pipeline
   - Monitor progress in the Graph View or Gantt Chart

6. **Stop Airflow services:**
   ```bash
   docker compose down

   # To remove volumes and start fresh
   docker compose down -v
   ```

### Database Architecture

The system uses **separated PostgreSQL databases** for better isolation and performance:

```sql
-- Two dedicated databases in single PostgreSQL instance
CREATE DATABASE airflow OWNER airflow;  -- Airflow metadata
CREATE DATABASE mlflow OWNER mlflow;    -- MLflow experiments
```

**Benefits:**
- **Isolation**: Airflow and MLflow operations don't interfere
- **Security**: Separate user credentials and permissions
- **Backup**: Independent backup strategies for each service
- **Scaling**: Can easily migrate to separate database instances later

## How to Run the Pipeline

### Option 1: Local Execution (Traditional)

To execute the entire pipeline from data download to model evaluation, run the following command from the project root:

```bash
python src/run_pipeline.py
```

### Option 2: Docker Compose (Recommended)

Start services and run via Airflow UI or trigger tasks with the CLI:

```bash
# Start complete MLOps stack
docker compose up -d

# Run pipeline via container with proper networking
docker run --rm --network news-classifier_default \
  --env MLFLOW_TRACKING_URI=http://mlflow:5000 \
  --mount type=bind,src=$(pwd)/data,dst=/opt/airflow/data \
  --mount type=bind,src=$(pwd)/models,dst=/opt/airflow/models \
  --mount type=bind,src=$(pwd)/reports,dst=/opt/airflow/reports \
  ml-pipeline:latest python src/run_pipeline.py

# Stop services
docker compose down
```

### Option 3: Airflow Orchestration (Recommended)

Use the Airflow DAG for production-grade workflow management:

1. **Start Airflow:** `docker-compose up -d`
2. **Access UI:** [http://localhost:8080](http://localhost:8080)
3. **Enable DAG:** Toggle the `ml_pipeline_news_classification` DAG
4. **Trigger Run:** Click "Trigger DAG" to start execution
5. **Monitor:** Watch progress in Graph View with detailed task logs

### Pipeline Outputs

All methods produce comprehensive outputs for MLOps workflows:

**Data Artifacts:**
- `data/raw/ag_news_raw.csv` - Raw downloaded AG News dataset
- `data/processed/train.csv` and `data/processed/test.csv` - Clean train/test splits
- `data/drifted_train.csv` and `data/drifted_test.csv` - Synthetic drift datasets for testing

**Model Artifacts:**
- `models/model.pkl` - Trained Logistic Regression classifier
- `models/tfidf_vectorizer.pkl` - Fitted TF-IDF vectorizer

**Reports & Monitoring:**
- `reports/metrics.txt` - Model performance metrics (accuracy, F1-score)
- `reports/evaluation_results.json` - Detailed evaluation results
- `reports/drift_report.json` - TF-IDF aware drift detection analysis
- `pipeline.log` - Detailed pipeline execution log

**MLflow Artifacts:**
- `mlflow/runs/` - Experiment run metadata and artifacts
- `mlflow/artifacts/` - Model registry and deployment artifacts

**Volume Persistence:**
All artifacts persist across container restarts via Docker volumes, ensuring data continuity in production environments.

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
# Build and start all services
docker compose build --no-cache
docker compose up -d

# Wait for services to be healthy (check with)
docker compose ps

# Test the complete pipeline with drift detection locally
python src/run_pipeline.py

# OR Test via container with proper network and volume mounts
docker run --rm --network news-classifier_default \
  --env MLFLOW_TRACKING_URI=http://mlflow:5000 \
  --mount type=bind,src=$(pwd)/data,dst=/opt/airflow/data \
  --mount type=bind,src=$(pwd)/models,dst=/opt/airflow/models \
  --mount type=bind,src=$(pwd)/reports,dst=/opt/airflow/reports \
  ml-pipeline:latest python src/run_pipeline.py

# Verify MLflow tracking is working
# Navigate to http://localhost:5000 to see experiment tracking

# Test the Airflow DAG (use correct DAG ID)
# Navigate to http://localhost:8080, enable and trigger ml_pipeline_news_classification

# Test the DAG from command line (use correct DAG ID)
docker compose exec airflow-webserver airflow dags test ml_pipeline_news_classification 2025-08-02

# List all tasks in the DAG to verify structure
docker compose exec airflow-webserver airflow tasks list ml_pipeline_news_classification

# Test drift detection specifically using TF-IDF aware method
python -c "
from src.drift_detection import detect_drift
try:
    # This uses the new TF-IDF aware drift detection
    result = detect_drift('data/processed/test.csv', 'data/drifted_test.csv')
    print('Drift detection results:', result)
    if result.get('dataset_drift', result.get('drift_detected', False)):
        print('✅ Drift detected as expected!')
    else:
        print('⚠️ No drift detected - check drifted data generation')
except Exception as e:
    print(f'❌ Drift detection failed: {e}')
"

# Verify healthy pipeline (no drift scenario)
python tests/test_healthy_pipeline.py

# Run comprehensive pipeline tests
python tests/test_full_pipeline.py
```
### Test Structure

- `tests/test_data.py`: Tests for data preprocessing functionality
- `tests/test_feature_engineering.py`: Tests for TF-IDF vectorization
- `tests/test_model_training.py`: Tests for model training
- `tests/test_evaluation.py`: Tests for model evaluation
- `tests/test_pipeline.py`: Integration tests for the complete pipeline
- `tests/test_healthy_pipeline.py`: End-to-end healthy scenario testing (no drift)
- `tests/test_full_pipeline.py`: Comprehensive Docker integration and service validation

### HW3 Verification Tests

The test suite includes specialized verification for HW3 requirements:

```bash
# Test healthy pipeline (no drift scenario)
python tests/test_healthy_pipeline.py

# Comprehensive service integration tests
python tests/test_full_pipeline.py
```

**`test_full_pipeline.py` validates:**
- Docker container health and connectivity
- MLflow tracking server accessibility and experiment creation
- TF-IDF aware drift detection functionality
- Airflow DAG structure and task execution
- Database separation and service isolation
- Network communication between services

**`test_healthy_pipeline.py` demonstrates:**
- Complete pipeline execution without drift
- MLflow integration with experiment logging
- Model registration for high-accuracy models
- Proper handling of non-drifted scenarios

## Model Drift Detection

The model drift detection system monitors data distribution changes that could degrade model performance over time. Our implementation uses a sophisticated **TF-IDF aware approach** with Evidently AI's DataDriftPreset for statistical drift analysis.

### TF-IDF Aware Drift Detection

Instead of analyzing raw text directly, our implementation transforms text into numeric TF-IDF features that match the actual model input, providing more meaningful drift detection:

1. **Feature Engineering**: Extracts TF-IDF features, vocabulary statistics, and label distributions
2. **Statistical Testing**: Uses Evidently's DataDriftPreset for robust statistical comparison
3. **Cosine Similarity**: Computes dataset-level centroid similarity as an additional metric
4. **Comprehensive Reporting**: Generates detailed drift reports with per-feature analysis

### Key Metrics Monitored

- **Vocabulary Overlap**: Measures shared vocabulary between reference and current datasets
- **OOV (Out-of-Vocabulary) Rate**: Tracks new words not seen during training
- **Document Length Distribution**: Monitors changes in text length patterns
- **Label Distribution**: Detects shifts in class proportions
- **TF-IDF Feature Drift**: Statistical tests on the actual model features
- **Centroid Cosine Similarity**: Overall semantic similarity between datasets

### Implementation Details

The drift detection system (`src/detect_drift_tfidf.py`) implements:

```python
def detect_drift_tfidf_aware(reference_csv, current_csv, report_path):
    """
    TF-IDF aware drift detection using Evidently.

    1. Fits TF-IDF vectorizer on reference data only
    2. Transforms both datasets using the same vectorizer
    3. Computes vocabulary and statistical features
    4. Runs Evidently DataDriftPreset on numeric features
    5. Returns comprehensive drift analysis
    """
```

**Advantages over naive text comparison:**
- Tests the same features the ML model uses
- Robust statistical testing with proper p-values
- Handles high-dimensional sparse data correctly
- Provides interpretable feature-level drift scores

### Drift Detection in Pipeline

Drift detection is integrated at multiple levels:

1. **Main Pipeline** (`src/run_pipeline.py`): Compares processed test vs drifted test data
2. **Airflow DAG**: Automated drift analysis with branching logic for retraining
3. **Test Suite**: Validation scripts for both healthy and drifted scenarios

The system automatically raises errors when significant drift is detected, triggering model retraining workflows in production environments.

## MLflow Integration

MLflow provides comprehensive experiment tracking and model lifecycle management throughout our ML pipeline. The integration automatically logs model parameters, metrics, and artifacts during training, enabling reproducible experiments and model versioning.

### Architecture & Configuration

Our MLflow setup uses a **separated database architecture** for better isolation and scalability:

- **MLflow Tracking Server**: Dedicated container running on port 5000
- **PostgreSQL Backend**: Separate `mlflow` database with dedicated user credentials
- **Artifact Storage**: Persistent volume storage for model artifacts and experiment data
- **Network Communication**: Container-to-container communication via Docker Compose networking

### Key Configuration Details

```yaml
# docker-compose.yml MLflow service
mlflow:
  build:
    dockerfile: Dockerfile.mlflow
  environment:
    MLFLOW_BACKEND_STORE_URI: postgresql://mlflow:mlflow@postgres:5432/mlflow
    MLFLOW_DEFAULT_ARTIFACT_ROOT: /mlflow/artifacts
  ports:
    - "5000:5000"
```

```sql
-- config/init-db.sql - Database separation
CREATE USER airflow WITH PASSWORD 'airflow';
CREATE DATABASE airflow OWNER airflow;

CREATE USER mlflow WITH PASSWORD 'mlflow';
CREATE DATABASE mlflow OWNER mlflow;
```

### Custom PyFunc Model Implementation

Our custom PyFunc model wrapper ensures seamless deployment compatibility while maintaining access to preprocessing artifacts:

```python
class NewsClassificationModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """Load model and preprocessing artifacts"""

    def predict(self, context, model_input):
        """Predict with preprocessing pipeline"""
```

### Experiment Logging

The integration automatically logs comprehensive experiment data:

- **3 Hyperparameters for Logistic Regression**:
  - `max_iter`: Maximum iterations (1000) - crucial for text classification convergence
  - `random_state`: Random seed (42) - ensures reproducible experiments
  - `penalty`: Regularization type ("l2") - prevents overfitting on sparse TF-IDF features

- **Performance Metrics**:
  - `accuracy`: Overall classification accuracy
  - `f1_score`: Weighted F1-score for multi-class evaluation
  - Training completion status and timestamps

- **Model Artifacts**:
  - Trained classifier (`model.pkl`)
  - Fitted TF-IDF vectorizer (`tfidf_vectorizer.pkl`)
  - Custom PyFunc model for deployment

### Model Registration

Models achieving accuracy scores above 0.8 are automatically registered in the MLflow Model Registry, enabling version control and deployment management.

### Verification & Access

```bash
# MLflow UI accessible at http://localhost:5000 (host)
# Inside containers: http://mlflow:5000

# Environment variables for containers
MLFLOW_TRACKING_URI=http://mlflow:5000

# Verify MLflow connectivity
curl http://localhost:5000  # Should return MLflow UI HTML
```

### Integration Points

MLflow tracking is integrated throughout the pipeline:

1. **Training Pipeline** (`src/model_training.py`): Automatic experiment logging with custom model registration
2. **Evaluation Phase** (`src/evaluation.py`): Metrics logging and conditional model registration
3. **Airflow DAG**: MLflow tracking URLs in task logs for experiment traceability
4. **Drift Detection**: Drift results logged as MLflow parameters for monitoring

## Code Structure

### Core Modules

- `src/config.py`: Centralized configuration management
- `src/utils.py`: Utility functions for logging, error handling, and validation
- `src/download_data.py`: Data download from Hugging Face datasets
- `src/data_preprocessing.py`: Data cleaning and train/test splitting with drift data generation
- `src/feature_engineering.py`: TF-IDF vectorization
- `src/model_training.py`: Logistic regression model training with MLflow integration
- `src/evaluation.py`: Model evaluation, metrics calculation, and MLflow registration
- `src/drift_detection.py`: TF-IDF aware drift detection using Evidently
- `src/detect_drift_tfidf.py`: Core TF-IDF feature extraction and statistical drift analysis
- `src/drift_features.py`: TF-IDF feature engineering for drift monitoring
- `src/drift_generation.py`: Synthetic drift data generation for testing
- `src/run_pipeline.py`: Main pipeline orchestration with drift handling and MLflow tracking

### Airflow DAG Components

- `deploy/airflow/dags/ml_pipeline_dag.py`: Complete 8-task DAG with branching logic
  - **5 Core Tasks**: preprocess_data, feature_engineering, train_model, evaluate_model, drift_detection
  - **Branching Logic**: branch_on_drift routes to retrain_model or pipeline_complete
  - **MLflow Integration**: Experiment tracking throughout all tasks
  - **Error Handling**: Comprehensive retry logic and error reporting

### Testing Infrastructure

- `tests/test_healthy_pipeline.py`: Healthy scenario testing (no drift expected)
- `tests/test_full_pipeline.py`: Comprehensive Docker and service integration tests
- Additional unit tests for each module with pytest fixtures and mocking

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
├── vector_store/           # Vector database storage for similarity search (currently unused)
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
├── Dockerfile              # Airflow-based container definition
├── Dockerfile.mlflow       # MLflow service container
├── docker-compose.yml      # Multi-service orchestration (Airflow + MLflow)
├── .dockerignore           # Docker build context optimization
├── pyproject.toml          # Project dependencies and metadata
├── .pre-commit-config.yaml # Pre-commit hooks configuration
└── README.md               # This file
```

**Key Design Decisions:**

- **`mlflow/` Directory**: Dedicated storage for MLflow experiment tracking, separating metadata and artifacts from core pipeline data for better organization and backup strategies
- **Enhanced `reports/` Directory**: Now includes evaluation metrics, drift detection reports, and MLflow experiment data, providing comprehensive model monitoring capabilities
- **`deploy/` Directory**: Isolating DAGs in `airflow/dags/` ensures modularity, allowing independent testing of workflow tasks without affecting the core ML code
- **Separated Database Architecture**: `config/init-db.sql` creates dedicated users and databases for Airflow and MLflow, enabling better isolation and scaling
- **Volume Strategy**: Docker Compose uses named volumes (`pipeline-data`, `pipeline-models`, `pipeline-reports`) to avoid Windows path mounting issues while maintaining persistence
- **Multi-service Architecture**: Separate Dockerfiles (`Dockerfile` for Airflow, `Dockerfile.mlflow` for MLflow) enable independent scaling and updates of each service
- **TF-IDF Aware Drift Detection**: Custom implementation in `src/detect_drift_tfidf.py` analyzes the same features the model uses rather than raw text

## Pre-commit Configuration

This project uses `pre-commit` to enforce code quality and consistency. The following hooks are configured:

1. **Ruff**: Fast Python linter and formatter for code quality
2. **Standard Hooks**: File formatting (`trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-dockerfile`)
3. **Hadolint**: Dockerfile linting for best practices and security vulnerabilities (prevents issues like unnecessary root privileges)
4. **yamllint**: YAML file validation for docker-compose.yml and other configuration files
5. **Pytest**: Local test hook provided; runs manually (not blocking by default)

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

The project uses an Airflow-based Dockerfile with the following key features:

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

1. **Airflow base image**: DAGs and pipeline run in same runtime
2. **pip inside container**: `uv` is recommended for local dev only
3. **Non-root user**: Security best practice to avoid running containers as root
4. **Health checks**: Managed via docker-compose service checks
5. **Volume mounts**: Persistent data storage for `data/`, `models/`, and `reports/`

### Build and Run

Use docker-compose to build and run the complete MLOps stack including separated databases, MLflow tracking, and Airflow orchestration:

```bash
# Build all services
docker compose build --no-cache

# Start the complete stack
docker compose up -d

# Check service health
docker compose ps

# Stop and cleanup
docker compose down -v
```

### Volume Strategy

The system uses **Docker named volumes** to avoid Windows path mounting issues while ensuring data persistence:

```yaml
volumes:
  pipeline-data:      # Persistent data storage (raw, processed, drifted datasets)
  pipeline-models:    # Model artifacts and TF-IDF vectorizers
  pipeline-reports:   # Evaluation metrics and drift reports
  mlflow-artifacts:   # MLflow experiment artifacts and model registry
  postgres-db-volume: # PostgreSQL data for both Airflow and MLflow databases
```

**Benefits:**
- **Cross-Platform Compatibility**: Works on Windows, macOS, and Linux
- **Data Persistence**: Survives container restarts and rebuilds
- **Performance**: Better I/O performance than bind mounts on Windows
- **Isolation**: Each service's data is properly separated

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

Building this comprehensive MLOps pipeline with MLflow integration and TF-IDF aware drift detection taught me several critical lessons about production ML systems:

### Technical Learnings

1. **Service Architecture Matters**: Initially attempted to share databases between Airflow and MLflow, which caused schema conflicts. The final separated database architecture (`config/init-db.sql`) provides better isolation, security, and scaling capabilities.

2. **TF-IDF Drift Detection Complexity**: Raw text drift detection was insufficient for ML monitoring. Implementing TF-IDF aware drift detection (`src/detect_drift_tfidf.py`) that analyzes the same features the model uses provided much more meaningful and actionable drift signals.

3. **Container Orchestration Challenges**: Docker Compose networking, volume mounting (especially on Windows), and service health checks required careful consideration. Named volumes solved cross-platform compatibility issues while maintaining data persistence.

4. **MLflow Integration Depth**: Beyond basic experiment tracking, implementing custom PyFunc models, automated model registration, and proper artifact management created a production-ready model lifecycle.

### Process Improvements

5. **Comprehensive Testing Strategy**: The two-tier testing approach (`test_healthy_pipeline.py` for scenarios, `test_full_pipeline.py` for infrastructure) provided both functional validation and operational confidence.

6. **Configuration Management**: Centralized configuration in `src/config.py` with environment-specific overrides made the system much more maintainable across development, testing, and production environments.

7. **Error Handling**: Implementing proper retry logic, comprehensive logging, and graceful degradation in the Airflow DAG prevented many operational issues during development.

### Project Evolution

This project started as an attempt to reuse RAG model code but evolved into a proper news classification pipeline when I carefully read the HW3 requirements. This taught me the critical importance of:

- **Requirements Analysis**: Thoroughly understanding specifications before implementation
- **Iterative Development**: Being willing to refactor when requirements become clearer
- **Documentation**: Maintaining accurate documentation as the codebase evolves

The final architecture successfully demonstrates enterprise-grade MLOps patterns: containerized deployment, experiment tracking, automated drift detection, and workflow orchestration with proper monitoring and alerting capabilities.

### Challenge: Docker and Airflow Integration

**Challenge**: The most significant challenge was properly integrating the ML pipeline with Airflow's containerized environment while maintaining MLflow tracking and handling Python import paths.

**Multiple Issues Resolved:**

1. **Database Schema Conflicts**: Initially, Airflow attempted to use MLflow's database, causing "airflow db upgrade" errors. Resolved by implementing separated PostgreSQL databases with dedicated users (airflow/airflow, mlflow/mlflow).

2. **Python Path Issues**: DAG couldn't import custom modules from `src/` directory. Fixed by explicitly adding the project root to Python path in the DAG (`sys.path.insert(0, '/opt/airflow')`) and ensuring proper volume mounts.

3. **Health Check Problems**: Scheduler health checks were failing with curl on wrong ports. Resolved by switching to `airflow jobs check` commands that properly validate Airflow service status.

4. **DAG ID Mismatches**: Testing commands used incorrect DAG names. Identified the actual DAG ID (`ml_pipeline_news_classification`) through systematic debugging.

5. **MLflow Network Communication**: Container-to-container MLflow communication required proper Docker Compose networking (`http://mlflow:5000` instead of `localhost:5000`).

**Final Solution Architecture:**
- Separated PostgreSQL databases for Airflow and MLflow
- Proper Docker Compose networking for service communication
- Corrected health checks using appropriate Airflow commands
- Volume mounts for persistent data and proper Python module access
- Comprehensive error handling and retry logic in DAG tasks

This taught me the importance of designing containerized MLOps architectures with proper service isolation, networking, and data persistence from the beginning rather than retrofitting traditional code.
