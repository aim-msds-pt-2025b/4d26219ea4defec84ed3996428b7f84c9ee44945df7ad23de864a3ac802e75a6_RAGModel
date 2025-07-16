"""
Test module for main.py functionality.

This module contains tests for the main entry point of the RAG Model project.
"""

import pytest
from unittest.mock import patch
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import main


class TestMain:
    """Test class for main.py functionality."""

    @patch("main.run_pipeline")
    def test_main_success(self, mock_run_pipeline):
        """Test successful pipeline execution."""
        mock_run_pipeline.return_value = None

        # Capture print output
        with patch("builtins.print") as mock_print:
            main.main()

        mock_run_pipeline.assert_called_once()
        mock_print.assert_any_call("Starting RAG Model ML Pipeline...")
        mock_print.assert_any_call("ML Pipeline completed successfully!")

    @patch("main.run_pipeline")
    def test_main_with_exception(self, mock_run_pipeline):
        """Test pipeline execution with exception."""
        test_error = Exception("Test pipeline error")
        mock_run_pipeline.side_effect = test_error

        with patch("builtins.print") as mock_print:
            with pytest.raises(Exception, match="Test pipeline error"):
                main.main()

        mock_print.assert_any_call("Starting RAG Model ML Pipeline...")
        mock_print.assert_any_call("ML Pipeline failed with error: Test pipeline error")
