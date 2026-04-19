from __future__ import annotations

# -----------------------------------------------------------------------------
# AI-assisted / AI-generated code (Cursor). Prompts shape LLM behavior; review
# for safety and policy alignment before any production use.
# -----------------------------------------------------------------------------

from typing import Any


def build_explanation_prompt(patient: dict[str, Any], prediction: dict[str, Any]) -> str:
    """Single user message body: vitals + model outputs + formatting rules for the LLM."""
    factors = prediction.get("top_factors") or []
    lines = "\n".join(
        f"  - {f['feature']}: {f['value']} ({f['vs_median']})" for f in factors
    )
    return f"""You are a hospital operations decision-support assistant (not a clinician).

Structured inputs for one patient snapshot:
- Age: {patient.get('age')}
- Heart rate: {patient.get('heart_rate')}
- Respiratory rate: {patient.get('resp_rate')}
- Oxygen saturation: {patient.get('spo2')}
- Systolic BP: {patient.get('systolic_bp')}
- Diastolic BP: {patient.get('diastolic_bp')}
- Temperature (C): {patient.get('temperature')}
- Glucose (mg/dL): {patient.get('glucose')}
- BMI: {patient.get('bmi')}

Model output (demo proxy for higher-acuity encounter risk):
- Risk probability: {prediction.get('risk_probability')}
- Risk band: {prediction.get('risk_band')}
- Ranked deviation highlights:
{lines if lines else '  (none)'}

Write a short response with numbered sections:
1) Main reasons for the risk band (tie to vitals; no diagnosis).
2) Operational concern level (Low/Medium/High workload concern).
3) Suggested next operational step (monitoring frequency / reassessment / escalation for review).

Rules: do not claim certainty; no diagnosis or treatment orders; under 150 words; plain language."""


def build_messages(patient: dict[str, Any], prediction: dict[str, Any]) -> list[dict[str, str]]:
    """Chat Completions format: short system guardrails + user block from build_explanation_prompt."""
    return [
        {
            "role": "system",
            "content": "You help hospital operations staff prioritize monitoring and reassessment. You never diagnose or prescribe medications.",
        },
        {
            "role": "user",
            "content": build_explanation_prompt(patient, prediction),
        },
    ]
