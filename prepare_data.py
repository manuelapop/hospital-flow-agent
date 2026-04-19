from __future__ import annotations

"""
AI-assisted / AI-generated code (Cursor). Educational demo only — not for clinical use.

Build encounter-level training rows from Synthea CSV exports:
observations (vitals/labs) + patients (age) + encounters (proxy acuity label).

Label risk_label=1 if ENCOUNTERCLASS is inpatient or emergency (higher-acuity setting),
else 0. This is a teaching / demo proxy for escalation risk, not a clinical endpoint.
"""

import pandas as pd
from pathlib import Path

from utils import TRAINING_CSV, ensure_dirs, synthea_csv_dir

# LOINC codes used in many Synthea exports
CODE_MAP = {
    "8867-4": "heart_rate",
    "9279-1": "resp_rate",
    "2708-6": "spo2",
    "8480-6": "systolic_bp",
    "8462-4": "diastolic_bp",
    "8310-5": "temperature",
    "2345-7": "glucose",
    "39156-5": "bmi",
}

HIGH_ACUITY = {"inpatient", "emergency"}


def load_observations_long(csv_dir: Path) -> pd.DataFrame:
    """Stream large observations.csv in chunks; keep only numeric LOINC rows we map to features."""
    path = csv_dir / "observations.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    usecols = ["DATE", "PATIENT", "ENCOUNTER", "CODE", "VALUE", "TYPE"]
    codes = set(CODE_MAP.keys())
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, low_memory=False, chunksize=250_000):
        c = chunk[chunk["TYPE"] == "numeric"]
        c = c[c["CODE"].isin(codes)]
        c = c.dropna(subset=["ENCOUNTER", "VALUE"])
        c["VALUE"] = pd.to_numeric(c["VALUE"], errors="coerce")
        c = c.dropna(subset=["VALUE"])
        if len(c):
            parts.append(c)
    obs = pd.concat(parts, ignore_index=True)
    obs["feature"] = obs["CODE"].map(CODE_MAP)
    return obs


def pivot_encounter_features(obs: pd.DataFrame) -> pd.DataFrame:
    # Wide table: one row per ENCOUNTER; mean if multiple values per (encounter, feature).
    g = obs.groupby(["ENCOUNTER", "feature"], as_index=False)["VALUE"].mean()
    wide = g.pivot(index="ENCOUNTER", columns="feature", values="VALUE").reset_index()
    wide.columns.name = None
    return wide


def main() -> None:
    """Join vitals to encounters + patients, compute proxy label, write TRAINING_CSV."""
    ensure_dirs()
    csv_dir = synthea_csv_dir()
    print(f"Reading Synthea CSVs from: {csv_dir.resolve()}")

    enc = pd.read_csv(
        csv_dir / "encounters.csv",
        usecols=["Id", "START", "PATIENT", "ENCOUNTERCLASS"],
        low_memory=False,
    )
    enc = enc.rename(columns={"Id": "encounter_id"})

    pat = pd.read_csv(
        csv_dir / "patients.csv",
        usecols=["Id", "BIRTHDATE"],
        low_memory=False,
    )
    pat = pat.rename(columns={"Id": "patient_id"})
    pat["BIRTHDATE"] = pd.to_datetime(pat["BIRTHDATE"], errors="coerce")

    obs = load_observations_long(csv_dir)
    feat = pivot_encounter_features(obs)

    merged = feat.merge(enc, left_on="ENCOUNTER", right_on="encounter_id", how="inner")
    merged = merged.merge(pat, left_on="PATIENT", right_on="patient_id", how="inner")

    start = pd.to_datetime(merged["START"], utc=True, errors="coerce").dt.tz_localize(None)
    merged["age"] = ((start - merged["BIRTHDATE"]).dt.days / 365.25).clip(lower=0, upper=120)

    # Proxy outcome: synthetic "high acuity setting" — not a validated deterioration label.
    merged["risk_label"] = merged["ENCOUNTERCLASS"].str.lower().isin(HIGH_ACUITY).astype(int)

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
    for c in feature_cols:
        if c not in merged.columns:
            merged[c] = float("nan")

    out = merged[feature_cols + ["risk_label"]].copy()
    before = len(out)
    # Require minimal vitals so the model always has core inputs (same as UI defaults intent).
    out = out.dropna(subset=["heart_rate", "spo2", "systolic_bp", "age"])
    print(f"Rows before dropna (core vitals): {before}, after: {len(out)}")
    print(out["risk_label"].value_counts())

    out.to_csv(TRAINING_CSV, index=False)
    print(f"Wrote {TRAINING_CSV}")


if __name__ == "__main__":
    main()
