# News Topic Classification Project

This project is for Homework 1 of the MLOPS2025B course. It builds a complete, production-driven ML pipeline to classify news articles into four topics.

## Project Overview

The goal is to classify news headlines from the AG News dataset into one of four categories: World, Sports, Business, or Sci/Tech. This is a classic multi-class text classification problem. I chose this dataset because it's a standard benchmark, easy to access programmatically via the `datasets` library, and fits the size constraints of the assignment (<100MB). This allows me to focus on building a robust MLOps pipeline rather than complex data engineering.

## How to Get the Data

The AG News dataset is downloaded automatically when you run the pipeline. The `src/data_preprocessing.py` script uses the Hugging Face `datasets` library to fetch and save the data to the `data/` directory. No manual download is required.

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
    # Create the virtual environment
    uv venv

    # Activate the environment
    source .venv/bin/activate

    # Install dependencies from pyproject.toml
    uv pip install -r requirements.txt
    ```
    *Note: Generate a `requirements.txt` with `uv pip freeze > requirements.txt` for this instruction.*


4. **Set up pre-commit hooks:**
    ```bash
    pre-commit install
    ```

## How to Run the Pipeline

To execute the entire pipeline from data download to model evaluation, run the following command from the project root:

```bash
python src/run_pipeline.py
```

The script will produce:
- `data/processed/train.csv` and `data/processed/test.csv`
- `models/model.pkl` (the trained classifier)
- `models/tfidf_vectorizer.pkl` (the fitted vectorizer)
- `reports/metrics.txt` (containing the model's accuracy)

## Folder Structure

The project uses a structured layout to ensure clarity and reproducibility, inspired by the Medallion Architecture:

-   `data/`: Contains all data.
    -   `raw/` (Bronze): For raw, untouched data.
    -   `processed/` (Silver): For cleaned and split data (train/test sets).
-   `models/` (Gold): Stores the final trained model (`model.pkl`) and other artifacts like the vectorizer.
-   `reports/` (Gold): Contains the model's performance metrics (`metrics.txt`).
-   `src/`: All Python source code, with each script handling a specific stage of the ML lifecycle.
-   `tests/`: For optional unit tests.
-   `pyproject.toml`: Manages all project dependencies for `uv`.
-   `.pre-commit-config.yaml`: Defines the pre-commit hooks for code quality.

This structure separates concerns, making the project easy to navigate, debug, and automate.

## Pre-commit Configuration

This project uses `pre-commit` to enforce code quality and consistency. The following hooks are configured in `.pre-commit-config.yaml`:

1.  **Ruff**: An extremely fast Python linter and formatter. It checks for style errors, potential bugs, and ensures all code follows a consistent format. This reduces merge conflicts and improves readability.
2.  **Standard Hooks (`trailing-whitespace`, `end-of-file-fixer`, `check-yaml`)**: These hooks handle common file formatting issues, ensuring clean and consistent files across the repository.

## Reflection

One challenge I faced was setting up the `uv` environment for the first time. I initially tried to install packages one by one without activating the virtual environment, leading to global installations. I resolved this by carefully following the `uv venv` and `source .venv/bin/activate` sequence. This reinforced the importance of isolated environments. The modular script design also posed a challenge in passing data between steps; I solved this by having each function return the path or object needed by the next step, making the pipeline flow explicit and easy to follow.
