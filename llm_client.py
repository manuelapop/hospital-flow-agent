from __future__ import annotations

# -----------------------------------------------------------------------------
# AI-assisted / AI-generated code (Cursor). Calls third-party OpenAI API; do not
# log or expose API keys. Educational demo only.
# -----------------------------------------------------------------------------

import os
from typing import Any

from prompts import build_messages

from utils import openai_api_key


def explain_risk_openai(
    patient: dict[str, Any],
    prediction: dict[str, Any],
    model: str | None = None,
) -> str:
    """Call OpenAI Chat Completions; API key from utils.openai_api_key() / .env."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("Install openai: pip install openai") from e

    key = openai_api_key()
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY in `.env` or the environment.")

    client = OpenAI(api_key=key)
    # Default model is small/cheap; override with OPENAI_MODEL env or argument.
    use_model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=use_model,
        messages=build_messages(patient, prediction),
        temperature=0.3,
        max_tokens=400,
    )
    choice = resp.choices[0].message.content
    return (choice or "").strip()
