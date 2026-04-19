from __future__ import annotations

# -----------------------------------------------------------------------------
# AI-assisted / AI-generated code (Cursor). Educational demo only — not for
# clinical use.
# -----------------------------------------------------------------------------

import joblib
import math
import pandas as pd
from pathlib import Path
from typing import Any

from utils import MODEL_PATH


def _rank_top_factors(
    patient: dict[str, Any],
    features: list[str],
    train_median: dict[str, float],
    train_std: dict[str, float],
    importances: list[float],
    top_k: int = 4,
) -> list[dict[str, Any]]:
    """Heuristic ranking: large |z| vs training cohort, weighted by global permutation importance."""
    scores: list[tuple[float, str, float, str]] = []
    for i, name in enumerate(features):
        val = patient.get(name)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue
        med = float(train_median.get(name, 0.0))
        std = float(train_std.get(name, 1.0)) or 1.0
        z = abs((float(val) - med) / std)
        imp = float(importances[i]) if i < len(importances) else 0.0
        direction = "higher than cohort median" if float(val) > med else "lower than cohort median"
        scores.append((z * imp, name, float(val), direction))
    scores.sort(key=lambda x: x[0], reverse=True)
    out: list[dict[str, Any]] = []
    for _, name, val, direction in scores[:top_k]:
        out.append({"feature": name, "value": round(val, 3), "vs_median": direction})
    return out


class RiskPredictor:
    """Loads joblib bundle from train.py and scores a single patient dict (UI or API)."""

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.features: list[str] = bundle["features"]
        self.train_median: dict[str, float] = bundle.get("train_median", {})
        self.train_std: dict[str, float] = bundle.get("train_std", {})
        self.feature_importances: list[float] = bundle.get("feature_importances", [])

    def predict(self, patient: dict[str, Any]) -> dict[str, Any]:
        row = pd.DataFrame([{k: patient.get(k) for k in self.features}])
        prob = float(self.model.predict_proba(row)[0, 1])
        label = int(prob >= 0.5)  # 0.5 threshold; bands below are for UX only
        # Display bands for Streamlit — arbitrary cutpoints, not clinical tiers.
        if prob < 0.33:
            band = "Low"
        elif prob < 0.66:
            band = "Medium"
        else:
            band = "High"
        top = _rank_top_factors(
            patient,
            self.features,
            self.train_median,
            self.train_std,
            self.feature_importances,
        )
        return {
            "risk_probability": round(prob, 4),
            "risk_label": label,
            "risk_band": band,
            "top_factors": top,
        }
