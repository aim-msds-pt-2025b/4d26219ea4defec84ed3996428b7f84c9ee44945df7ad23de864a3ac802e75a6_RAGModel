# src/utils.py

import logging
import functools
import os
import pandas as pd
from typing import List


def setup_logging(log_file: str = "pipeline.log"):
    """
    Set up logging configuration for the ML pipeline.

    Args:
        log_file (str): Path to the log file. Defaults to "pipeline.log".
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def handle_errors(func):
    """
    Decorator to handle errors in pipeline functions.

    Args:
        func: The function to wrap with error handling.

    Returns:
        The wrapped function with error handling.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}")
            raise

    return wrapper


def validate_file_exists(filepath: str):
    """
    Validate that a file exists.

    Args:
        filepath (str): Path to the file to validate.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")


def validate_dataframe(
    df: pd.DataFrame, required_columns: List[str], min_rows: int = 1
) -> bool:
    """
    Validate a DataFrame meets basic requirements.

    Args:
        df (pd.DataFrame): The DataFrame to validate.
        required_columns (List[str]): List of required column names.
        min_rows (int): Minimum number of rows required. Defaults to 1.

    Returns:
        bool: True if validation passes.

    Raises:
        ValueError: If validation fails.
    """
    if df.empty:
        raise ValueError("DataFrame is empty")

    if len(df) < min_rows:
        raise ValueError(f"DataFrame has {len(df)} rows, minimum required: {min_rows}")

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return True


def validate_text_data(df: pd.DataFrame, text_column: str = "text") -> bool:
    """
    Validate text data quality.

    Args:
        df (pd.DataFrame): DataFrame containing text data.
        text_column (str): Name of the text column. Defaults to 'text'.

    Returns:
        bool: True if validation passes.
    """
    if df[text_column].isnull().any():
        logging.warning(f"Found null values in {text_column} column")

    empty_texts = df[text_column].str.strip().str.len() == 0
    if empty_texts.any():
        logging.warning(f"Found {empty_texts.sum()} empty text entries")

    return True
