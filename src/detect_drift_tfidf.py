"""
TF-IDF aware drift detection using Evidently.

Uses numeric TF-IDF features instead of raw text for reliable drift detection.
"""

from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
from typing import Any, Dict

from src.drift_features import (
    fit_reference_vectorizer,
    compute_tfidf_features,
    dataset_centroid_cosine,
)

# Evidently imports with error handling
_evidently_available = True
try:
    from evidently import Report
    from evidently.presets import DataDriftPreset
except ImportError:
    _evidently_available = False


def detect_drift_tfidf_aware(
    reference_csv: str, current_csv: str, report_path: str = "reports/drift_report.json"
) -> Dict[str, Any]:
    """
    Detect drift using TF-IDF aware features.

    Args:
        reference_csv: Path to reference (training) data
        current_csv: Path to current (test) data
        report_path: Path to save drift report

    Returns:
        Drift detection results dictionary
    """
    if not _evidently_available:
        raise ImportError("evidently is not installed. Install 'evidently==0.7.11'.")

    # Load data
    ref = pd.read_csv(reference_csv)
    cur = pd.read_csv(current_csv)

    assert {"text", "label"}.issubset(ref.columns), (
        "reference missing text/label columns"
    )
    assert {"text", "label"}.issubset(cur.columns), "current missing text/label columns"

    # Fit vectorizer on reference text only
    vec = fit_reference_vectorizer(ref["text"])

    # Build TF-IDF aware feature frames
    ref_feats, ref_diag = compute_tfidf_features(vec, ref["text"])
    cur_feats, cur_diag = compute_tfidf_features(vec, cur["text"])

    # Add label to allow label prior drift detection
    ref_feats["label"] = ref["label"].to_numpy()
    cur_feats["label"] = cur["label"].to_numpy()

    # Optional: global similarity metric
    centroid_cos = dataset_centroid_cosine(vec, ref["text"], cur["text"])

    # Run Evidently on these numeric/categorical features
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_feats, current_data=cur_feats)

    # Extract results
    try:
        if hasattr(report, "json"):
            result = report.json()
            if isinstance(result, str):
                result = json.loads(result)
        elif hasattr(report, "as_dict"):
            result = report.as_dict()
        else:
            # Fallback: try to get the result from report object
            result = {}
    except Exception as e:
        print(f"Warning: Could not extract report data: {e}")
        result = {}

    # If we couldn't extract from Evidently, use manual drift detection
    if not result or not result.get("metrics"):
        # Simple fallback drift detection based on statistical differences
        drift_detected = False
        # Check if OOV rate or other features show significant difference
        oov_diff = abs(cur_diag["mean_oov_rate"] - ref_diag["mean_oov_rate"])
        rare_diff = abs(cur_diag["mean_rare_rate"] - ref_diag["mean_rare_rate"])
        cos_sim = centroid_cos

        # Simple thresholds for drift detection
        if oov_diff > 0.02 or rare_diff > 0.005 or cos_sim < 0.95:
            drift_detected = True

        result = {
            "metrics": [
                {
                    "result": {
                        "dataset_drift": drift_detected,
                        "drift_by_columns": {
                            "oov_rate": {"drift_score": oov_diff},
                            "rare_token_rate": {"drift_score": rare_diff},
                            "centroid_similarity": {"drift_score": 1.0 - cos_sim},
                        },
                    }
                }
            ]
        }

    # Extract dataset drift flag
    drift_detected = False
    for m in result.get("metrics", []):
        r = m.get("result", {})
        if "dataset_drift" in r:
            drift_detected = bool(r["dataset_drift"])
            break

    # Extract per-feature p-values/scores
    feature_drifts: Dict[str, float] = {}
    for m in result.get("metrics", []):
        r = m.get("result", {})
        by_col = r.get("drift_by_columns")
        if isinstance(by_col, dict):
            for col, info in by_col.items():
                if isinstance(info, dict):
                    p = info.get("p_value", info.get("drift_score"))
                    if p is not None:
                        feature_drifts[col] = float(p)

    # Build final payload
    payload = {
        "dataset_drift": drift_detected,
        "feature_drifts": feature_drifts,
        "overall_drift_score": float(np.mean(list(feature_drifts.values())))
        if feature_drifts
        else 0.0,
        "diagnostics": {
            "ref": ref_diag,
            "cur": cur_diag,
            "centroid_cosine": float(centroid_cos),
            "vocab_size": int(len(vec.vocabulary_)),
        },
    }

    print(f"TF-IDF aware drift detection result: {payload}")

    # Save report
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload
