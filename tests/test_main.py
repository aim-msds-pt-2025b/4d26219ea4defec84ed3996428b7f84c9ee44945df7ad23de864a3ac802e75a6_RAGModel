"""
Test module for main.py functionality in a Docker setup.

Covers the main entry point of the RAG Model project.
"""

import pytest
from unittest.mock import patch
import sys
from pathlib import Path

import importlib


@pytest.fixture(scope="module", autouse=True)
def ensure_src_in_syspath():
    """Ensure src/ is in sys.path for imports."""
    src_path = str(Path(__file__).parent.parent / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    yield
    # Optional: sys.path.remove(src_path) after tests, if you want to clean up


class TestMain:
    """Tests for main.py functionality."""

    @pytest.mark.unit
    @pytest.mark.fast
    @patch("src.run_pipeline.main")
    def test_main_success(self, mock_run_pipeline):
        """Test successful pipeline execution."""
        mock_run_pipeline.return_value = None

        import main

        importlib.reload(main)  # Ensure fresh module state

        with patch("builtins.print") as mock_print:
            main.main()

        mock_run_pipeline.assert_called_once()
        mock_print.assert_any_call("Starting RAG Model ML Pipeline...")
        mock_print.assert_any_call("ML Pipeline completed successfully!")

    @pytest.mark.unit
    @pytest.mark.fast
    @patch("src.run_pipeline.main")
    def test_main_with_exception(self, mock_run_pipeline):
        """Test pipeline execution with exception."""
        mock_run_pipeline.side_effect = Exception("Test pipeline error")

        import main

        importlib.reload(main)

        with patch("builtins.print") as mock_print:
            with pytest.raises(Exception, match="Test pipeline error"):
                main.main()

        mock_print.assert_any_call("Starting RAG Model ML Pipeline...")
        mock_print.assert_any_call("ML Pipeline failed with error: Test pipeline error")
