"""
Data ingestion utilities.

Provides a function to download or load the dataset into the local data folder
without changing the existing HW1/HW2 download behavior.
"""

from src.download_data import download_raw_data
from src.config import config


def ingest_data() -> str:
    """Download/load dataset into data folder and return raw CSV path.

    Uses existing download_raw_data implementation to preserve behavior.
    """
    # Ensure paths exist and fetch the raw data
    raw_path = download_raw_data()
    return raw_path or config.raw_data_path
