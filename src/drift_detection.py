"""
TF-IDF aware drift detection using Evidently.

Analyzes drift between reference (training) and current (test) datasets using
numeric TF-IDF features instead of raw text for reliable statistical testing.
"""

from typing import Any, Dict
from src.detect_drift_tfidf import detect_drift_tfidf_aware


def detect_drift(reference_data_path: str, current_data_path: str) -> Dict[str, Any]:
    """
    Detect drift using TF-IDF aware features with Evidently.

    This function replaces the previous hardcoded marker-based approach with
    a more realistic TF-IDF aware drift detection that measures the same
    signal that the ML model uses.

    Args:
        reference_data_path: Path to reference (training) CSV file
        current_data_path: Path to current (test) CSV file

    Returns:
        Dictionary containing drift detection results with:
        - dataset_drift: boolean indicating if drift was detected
        - feature_drifts: dict of feature-specific drift scores
        - overall_drift_score: float overall drift score
        - diagnostics: dict with additional metrics
    """
    return detect_drift_tfidf_aware(reference_data_path, current_data_path)
