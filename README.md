# News Topic Classification Project

This project is for Homework 1 of the MLOPS2025B course. It builds a complete, production-driven ML pipeline to classify news articles into four topics with comprehensive testing, logging, and error handling.

## Project Overview

The goal is to classify news headlines from the AG News dataset into one of four categories: World, Sports, Business, or Sci/Tech. This is a classic multi-class text classification problem. I chose this dataset because it's a standard benchmark, easy to access programmatically via the `datasets` library, and fits the size constraints of the assignment (<100MB). This allows me to focus on building a robust MLOps pipeline rather than complex data engineering.

## Key Features

- **Comprehensive Testing**: Unit tests for all modules with pytest
- **Logging**: Structured logging throughout the pipeline
- **Error Handling**: Robust error handling with custom decorators
- **Configuration Management**: Centralized configuration for easy maintenance
- **Data Validation**: Input validation for data quality assurance
- **Pre-commit Hooks**: Automated code quality checks including pytest execution

## How to Get the Data

The AG News dataset is downloaded automatically when you run the pipeline. The `src/download_data.py` script uses the Hugging Face `datasets` library to fetch and save the data to the `data/` directory. No manual download is required.

## Setup Instructions

This project uses `uv` for environment and dependency management.

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Install uv:** If you don't have it, follow the official instructions at [astral.sh](https://astral.sh/uv/install.sh).

3.  **Create the virtual environment and install dependencies:**
    ```bash
    # Create and activate the virtual environment
    uv venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate

    # Install dependencies
    uv sync
    ```

4.  **Set up pre-commit hooks:**
    ```bash
    pre-commit install
    ```

## How to Run the Pipeline

To execute the entire pipeline from data download to model evaluation, run the following command from the project root:

```bash
python src/run_pipeline.py
```

The script will produce:
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
```

### Test Structure

- `tests/test_data.py`: Tests for data preprocessing functionality
- `tests/test_feature_engineering.py`: Tests for TF-IDF vectorization
- `tests/test_model_training.py`: Tests for model training
- `tests/test_evaluation.py`: Tests for model evaluation
- `tests/test_pipeline.py`: Integration tests for the complete pipeline

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

### Key Improvements

1. **Configuration Management**: All settings centralized in `config.py`
2. **Logging**: Structured logging with file and console output
3. **Error Handling**: Comprehensive error handling with custom decorators
4. **Data Validation**: Input validation for data quality assurance
5. **Type Hints**: Improved code documentation and IDE support
6. **Testing**: Comprehensive test suite with fixtures and mocking
7. **Documentation**: Enhanced docstrings and comments

## Folder Structure

The project uses a structured layout to ensure clarity and reproducibility, inspired by the Medallion Architecture:

```
├── data/
│   ├── raw/                 # Bronze: Raw, untouched data
│   └── processed/           # Silver: Cleaned and split data
├── models/                  # Gold: Trained models and artifacts
├── reports/                 # Gold: Performance metrics and reports
├── src/                     # Python source code
│   ├── config.py           # Configuration management
│   ├── utils.py            # Utility functions
│   ├── download_data.py    # Data download module
│   ├── data_preprocessing.py # Data preprocessing
│   ├── feature_engineering.py # Feature engineering
│   ├── model_training.py   # Model training
│   ├── evaluation.py       # Model evaluation
│   └── run_pipeline.py     # Main pipeline script
├── tests/                   # Test suite
│   ├── test_data.py        # Data processing tests
│   ├── test_feature_engineering.py # Feature engineering tests
│   ├── test_model_training.py # Model training tests
│   ├── test_evaluation.py  # Evaluation tests
│   └── test_pipeline.py    # Integration tests
├── pyproject.toml          # Project dependencies and metadata
├── .pre-commit-config.yaml # Pre-commit hooks configuration
└── README.md               # This file
```

## Pre-commit Configuration

This project uses `pre-commit` to enforce code quality and consistency. The following hooks are configured:

1. **Ruff**: Fast Python linter and formatter for code quality
2. **Standard Hooks**: File formatting (`trailing-whitespace`, `end-of-file-fixer`, `check-yaml`)
3. **Pytest**: All tests must pass before commits are allowed

The pre-commit hooks ensure that:
- Code is properly formatted and linted
- All tests pass before committing
- Files are properly formatted
- YAML files are valid

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

## Reflection

Building this improved pipeline taught me several key lessons:

1. **Testing is Essential**: Comprehensive tests catch issues early and make refactoring safer
2. **Configuration Management**: Centralizing configuration makes the code more maintainable
3. **Logging vs Print**: Proper logging provides better debugging and monitoring capabilities
4. **Error Handling**: Explicit error handling makes the pipeline more robust
5. **Pre-commit Hooks**: Automated quality checks prevent issues from reaching the repository

The modular design makes it easy to extend the pipeline with new features, different models, or additional data sources. The testing framework ensures that changes don't break existing functionality.
