"""
AI-assisted / AI-generated code (Cursor). Educational reporting helper only.

Regenerate the same train/test split as train.py (random_state=42, stratify=y),
evaluate the saved model on the test set, and write figures for a report.

Outputs (under figures/):
  - confusion_matrix.png
  - roc_curve.png

Run from hospital-flow-agent/ after prepare_data.py and train.py:
  python make_report_figures.py
"""

from __future__ import annotations

import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from utils import MODEL_PATH, TRAINING_CSV, ensure_dirs

FIG_DIR = Path(__file__).resolve().parent / "figures"


def main() -> None:
    """Recreate holdout split, evaluate saved bundle, export PNGs for write-ups."""
    if not MODEL_PATH.exists():
        raise SystemExit(f"Missing {MODEL_PATH}. Run train.py first.")
    if not TRAINING_CSV.exists():
        raise SystemExit(f"Missing {TRAINING_CSV}. Run prepare_data.py first.")

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("Install matplotlib: pip install matplotlib") from e

    # Must use same sklearn/numpy stack as training when unpickling.
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_cols: list[str] = bundle["features"]

    df = pd.read_csv(TRAINING_CSV)
    X = df[feature_cols]
    y = df["risk_label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    print("Test set size:", len(y_test))
    print("ROC-AUC (holdout):", round(auc, 4))
    print("Confusion matrix [[TN, FP], [FN, TP]]:\n", cm)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=[0, 1], ax=ax, colorbar=False, values_format="d"
    )
    ax.set_xlabel("Predicted label (0 = other, 1 = inpatient/emergency proxy)")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix (holdout test set)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax, name="Model")
    ax.set_title(f"ROC curve (AUC = {auc:.3f})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)

    print(f"Saved {FIG_DIR / 'confusion_matrix.png'}")
    print(f"Saved {FIG_DIR / 'roc_curve.png'}")


if __name__ == "__main__":
    ensure_dirs()
    main()
