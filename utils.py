from __future__ import annotations

# -----------------------------------------------------------------------------
# AI-assisted / AI-generated code (Cursor). Paths and env loading for the demo.
# -----------------------------------------------------------------------------

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
# Local secrets: override=True so `.env` wins if the shell has an empty OPENAI_API_KEY.
load_dotenv(PROJECT_ROOT / ".env", override=True)

# Avoid treating whitespace-only or quoted secrets as "set"
_k = os.environ.get("OPENAI_API_KEY")
if _k is not None:
    _k = _k.strip().strip('"').strip("'")
    if _k:
        os.environ["OPENAI_API_KEY"] = _k
    else:
        os.environ.pop("OPENAI_API_KEY", None)

# Parent of this package = default folder containing Synthea CSV exports
DEFAULT_SYNTHEA_DIR = PROJECT_ROOT.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
TRAINING_CSV = DATA_DIR / "patients_training.csv"
MODEL_PATH = MODELS_DIR / "risk_model.pkl"


def synthea_csv_dir() -> Path:
    return Path(os.environ.get("SYNTHEA_DATA_DIR", DEFAULT_SYNTHEA_DIR))


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def openai_api_key() -> str:
    return (os.environ.get("OPENAI_API_KEY") or "").strip()


def is_openai_configured() -> bool:
    return bool(openai_api_key())
