"""
Test module for utility functions.

This module contains comprehensive tests for the utility functions.
"""

import pytest
import pandas as pd
from unittest.mock import patch
from pathlib import Path
import sys
import tempfile
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils import (
    setup_logging,
    handle_errors,
    validate_dataframe,
    validate_text_data,
)


class TestUtils:
    """Test class for utility functions."""

    def test_setup_logging_file_creation(self):
        """Test that setup_logging creates a log file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")

            # Setup logging
            setup_logging(log_file=log_file)

            # Check that log file was created
            assert os.path.exists(log_file)

    def test_handle_errors_decorator_success(self):
        """Test handle_errors decorator with successful function."""

        @handle_errors
        def successful_function():
            return "success"

        result = successful_function()
        assert result == "success"

    def test_handle_errors_decorator_with_exception(self):
        """Test handle_errors decorator with exception."""

        @handle_errors
        def failing_function():
            raise ValueError("Test error")

        with patch("src.utils.logging.error") as mock_error:
            with pytest.raises(ValueError, match="Test error"):
                failing_function()

            mock_error.assert_called_once()

    def test_validate_dataframe_success(self):
        """Test validate_dataframe with valid DataFrame."""
        df = pd.DataFrame(
            {
                "text": ["Sample text 1", "Sample text 2", "Sample text 3"],
                "label": [0, 1, 2],
            }
        )

        result = validate_dataframe(df, min_rows=2, required_columns=["text", "label"])
        assert result is True

    def test_validate_dataframe_empty(self):
        """Test validate_dataframe with empty DataFrame."""
        df = pd.DataFrame()

        with pytest.raises(ValueError, match="DataFrame is empty"):
            validate_dataframe(df, min_rows=1, required_columns=["text"])

    def test_validate_dataframe_insufficient_rows(self):
        """Test validate_dataframe with insufficient rows."""
        df = pd.DataFrame({"text": ["Sample text"], "label": [0]})

        with pytest.raises(
            ValueError, match="DataFrame has 1 rows, minimum required: 5"
        ):
            validate_dataframe(df, min_rows=5, required_columns=["text", "label"])

    def test_validate_dataframe_missing_columns(self):
        """Test validate_dataframe with missing required columns."""
        df = pd.DataFrame(
            {"text": ["Sample text 1", "Sample text 2"], "wrong_column": [0, 1]}
        )

        with pytest.raises(ValueError, match="Missing required columns: \\['label'\\]"):
            validate_dataframe(df, min_rows=1, required_columns=["text", "label"])

    def test_validate_text_data_success(self):
        """Test validate_text_data with valid text data."""
        df = pd.DataFrame({"text": ["Sample text 1", "Sample text 2", "Sample text 3"]})

        result = validate_text_data(df, text_column="text")
        assert result is True

    def test_validate_text_data_with_null_values(self):
        """Test validate_text_data with null values."""
        df = pd.DataFrame({"text": ["Sample text 1", None, "Sample text 3"]})

        with patch("src.utils.logging.warning") as mock_warning:
            result = validate_text_data(df, text_column="text")
            assert result is True
            mock_warning.assert_called_with("Found null values in text column")

    def test_validate_text_data_with_empty_strings(self):
        """Test validate_text_data with empty strings."""
        df = pd.DataFrame({"text": ["Sample text 1", "", "   ", "Sample text 4"]})

        with patch("src.utils.logging.warning") as mock_warning:
            result = validate_text_data(df, text_column="text")
            assert result is True
            mock_warning.assert_called_with("Found 2 empty text entries")

    def test_validate_text_data_custom_column(self):
        """Test validate_text_data with custom column name."""
        df = pd.DataFrame(
            {"content": ["Sample text 1", "Sample text 2", "Sample text 3"]}
        )

        result = validate_text_data(df, text_column="content")
        assert result is True

    def test_validate_text_data_with_null_and_empty(self):
        """Test validate_text_data with both null and empty values."""
        df = pd.DataFrame({"text": ["Sample text 1", None, "", "   ", "Sample text 5"]})

        with patch("src.utils.logging.warning") as mock_warning:
            result = validate_text_data(df, text_column="text")
            assert result is True

            # Check that both warnings were called
            calls = mock_warning.call_args_list
            assert len(calls) == 2
            assert any(
                "Found null values in text column" in str(call) for call in calls
            )
            assert any("Found 2 empty text entries" in str(call) for call in calls)
