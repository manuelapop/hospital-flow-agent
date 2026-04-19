from __future__ import annotations

# -----------------------------------------------------------------------------
# AI-assisted / AI-generated code (Cursor). Educational demo only — not for
# clinical or operational use without proper validation and human oversight.
# -----------------------------------------------------------------------------

# Load `.env` before other project imports so OPENAI_API_KEY is available.
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

import streamlit as st

from inference import RiskPredictor
from llm_client import explain_risk_openai
from prompts import build_explanation_prompt
from utils import MODEL_PATH, is_openai_configured

st.set_page_config(page_title="Hospital Flow Optimization Agent", layout="centered")

st.title("Hospital Flow Optimization Agent")
st.caption(
    "Demo: ML risk score from Synthea vitals + LLM operational summary. "
    "Not for clinical use. Label is a proxy (inpatient/emergency vs other encounter types)."
)

if not MODEL_PATH.exists():
    st.error(
        f"Model not found at `{MODEL_PATH}`. Run `python prepare_data.py` then `python train.py`."
    )
    st.stop()

# One shared predictor instance per Streamlit rerun (model path from utils).
predictor = RiskPredictor()

# Manual vitals snapshot — same feature names as training CSV / bundle.
age = st.number_input("Age", min_value=0.0, max_value=120.0, value=65.0, step=1.0)
heart_rate = st.number_input("Heart rate (/min)", min_value=20.0, max_value=220.0, value=95.0)
resp_rate = st.number_input("Respiratory rate (/min)", min_value=5.0, max_value=60.0, value=18.0)
spo2 = st.number_input("Oxygen saturation (%)", min_value=50.0, max_value=100.0, value=96.0)
systolic_bp = st.number_input("Systolic BP", min_value=60.0, max_value=220.0, value=120.0)
diastolic_bp = st.number_input("Diastolic BP", min_value=30.0, max_value=130.0, value=78.0)
temperature = st.number_input("Temperature (C)", min_value=34.0, max_value=42.0, value=37.0)
glucose = st.number_input("Glucose (mg/dL)", min_value=40.0, max_value=500.0, value=110.0)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=26.0)

patient = {
    "age": age,
    "heart_rate": heart_rate,
    "resp_rate": resp_rate,
    "spo2": spo2,
    "systolic_bp": systolic_bp,
    "diastolic_bp": diastolic_bp,
    "temperature": temperature,
    "glucose": glucose,
    "bmi": bmi,
}

col1, col2 = st.columns(2)
run_model = col1.button("Run risk model", type="primary")
# LLM step requires a non-empty OPENAI_API_KEY (from .env or environment).
run_llm = col2.button("Generate LLM explanation", disabled=not is_openai_configured())

# Persist model output across widget interactions within the session.
if "prediction" not in st.session_state:
    st.session_state["prediction"] = None
if "explanation" not in st.session_state:
    st.session_state["explanation"] = None

if run_model:
    st.session_state["prediction"] = predictor.predict(patient)
    st.session_state["explanation"] = None  # Invalidate prior LLM text when inputs rerun

if st.session_state["prediction"]:
    pred = st.session_state["prediction"]
    st.subheader("Model output")
    st.json(pred)

    with st.expander("LLM prompt (debug)"):
        st.code(build_explanation_prompt(patient, pred), language="text")

if run_llm:
    if not st.session_state["prediction"]:
        st.warning("Run the risk model first.")
    else:
        try:
            st.session_state["explanation"] = explain_risk_openai(
                patient, st.session_state["prediction"]
            )
        except Exception as e:
            st.error(str(e))

if st.session_state.get("explanation"):
    st.subheader("LLM explanation")
    st.write(st.session_state["explanation"])

if not is_openai_configured():
    st.info(
        "Set `OPENAI_API_KEY` in `hospital-flow-agent/.env` (see `.env.example`) or export "
        "it in the shell, then **restart Streamlit** (Rerun may not reload `.env`)."
    )
