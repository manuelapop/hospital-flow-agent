from __future__ import annotations

# -----------------------------------------------------------------------------
# AI-assisted / AI-generated code (Cursor). Educational demo only — not for
# clinical use. Review metrics and model choices before any real deployment.
# -----------------------------------------------------------------------------

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from utils import MODEL_PATH, TRAINING_CSV, ensure_dirs


def load_training(path: Path) -> pd.DataFrame:
    """Load CSV produced by prepare_data.py; validates required columns exist."""
    df = pd.read_csv(path)
    feature_cols = [
        "age",
        "heart_rate",
        "resp_rate",
        "spo2",
        "systolic_bp",
        "diastolic_bp",
        "temperature",
        "glucose",
        "bmi",
    ]
    missing = [c for c in feature_cols + ["risk_label"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Run prepare_data.py first.")
    return df


def main() -> None:
    ensure_dirs()
    df = load_training(TRAINING_CSV)
    feature_cols = [c for c in df.columns if c != "risk_label"]
    X = df[feature_cols]
    y = df["risk_label"].astype(int)

    # Single pipeline: median imputation (per feature) + tree classifier.
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_cols),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    random_state=42,
                    max_depth=8,
                    learning_rate=0.08,
                    max_iter=200,
                ),
            ),
        ]
    )

    # Stratified split keeps class balance in train/test (label is imbalanced).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # P(class 1) for ROC-AUC and UI bands
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

    # Cohort stats + permutation importance for lightweight "top factors" in the UI
    # (HistGradientBoosting has no per-sample feature importances).
    train_med = X_train.median(numeric_only=True)
    train_std = X_train.std(numeric_only=True).replace(0, np.nan)
    subsample = min(2000, len(X_train))
    rng = np.random.RandomState(42)
    idx = rng.choice(X_train.index, size=subsample, replace=False)
    perm = permutation_importance(
        model,
        X_train.loc[idx],
        y_train.loc[idx],
        n_repeats=4,
        random_state=42,
        n_jobs=-1,
    )
    importances = perm.importances_mean

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Bundle everything inference.py needs without retraining.
    joblib.dump(
        {
            "model": model,
            "features": feature_cols,
            "train_median": train_med.to_dict(),
            "train_std": train_std.to_dict(),
            "feature_importances": importances.tolist(),
        },
        MODEL_PATH,
    )
    print(f"Saved model bundle to {MODEL_PATH}")


if __name__ == "__main__":
    main()
