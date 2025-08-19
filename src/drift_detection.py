"""
Drift detection using Evidently.

Analyzes drift between reference (training) and current (test) datasets and
writes a JSON report to reports/drift_report.json
"""

from typing import Any, Dict
import json
import os
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evidently.report import Report  # type: ignore
    from evidently.metric_preset import DataDriftPreset  # type: ignore

_evidently_available = True
try:  # runtime optional import to prevent import errors during tests
    from evidently.report import Report  # type: ignore
    from evidently.metric_preset import DataDriftPreset  # type: ignore
except Exception:
    _evidently_available = False


def detect_drift(reference_data_path: str, current_data_path: str) -> Dict[str, Any]:
    if not _evidently_available:
        # Avoid failing at import time; raise clear error only when invoked
        raise ImportError(
            "evidently is not installed. Install 'evidently==0.7.11' to enable drift detection."
        )
    # Load CSVs
    ref_df = pd.read_csv(reference_data_path)
    cur_df = pd.read_csv(current_data_path)

    # Extract features only (exclude target if present)
    feature_cols = [c for c in ref_df.columns if c != "label"]
    ref_features = ref_df[feature_cols]
    cur_features = cur_df[feature_cols]

    # Build report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_features, current_data=cur_features)
    result = report.as_dict()

    # Extract top-level drift flag
    drift_detected = False
    try:
        drift_detected = bool(result["metrics"][0]["result"]["dataset_drift"])  # type: ignore[index]
    except Exception:
        pass

    # Extract per-feature drift scores
    feature_drifts = {}
    try:
        drift_by_columns = result["metrics"][1]["result"]["drift_by_columns"]  # type: ignore[index]
        # Select up to 3 features deterministically (first 3)
        for feat in list(drift_by_columns.keys())[:3]:
            info = drift_by_columns[feat]
            # Prefer p_value if available; otherwise, take drift_score
            score = info.get("p_value")
            if score is None:
                score = info.get("drift_score")
            feature_drifts[feat] = float(score) if score is not None else 0.0
    except Exception:
        pass

    # Overall drift score as simple average
    overall = 0.0
    if feature_drifts:
        overall = sum(feature_drifts.values()) / len(feature_drifts)

    payload = {
        "drift_detected": drift_detected,
        "feature_drifts": feature_drifts,
        "overall_drift_score": overall,
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/drift_report.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload
