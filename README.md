# Hospital Flow Optimization Agent (demo)

A small portfolio-style system that combines **tabular ML** on synthetic EHR-style data with an **LLM layer** for operational, plain-language summaries. It is intended for learning and demos—not for clinical decisions.

## What this project does

1. **Data preparation** — Builds one training row per **encounter** that has linked vitals/labs in `observations.csv`, plus patient age and encounter type from Synthea exports.
2. **Risk model** — Trains a classifier that outputs a **probability** and a **Low / Medium / High** band, plus a short list of features that deviate most from the training cohort (relative to global permutation importance).
3. **Streamlit app** — Lets you enter vitals manually, run the model, and optionally call **OpenAI** to generate a short operational explanation (monitoring / reassessment / escalation for review), without diagnosing or prescribing.

## Data we use

The pipeline expects **Synthea-style CSV exports** in a directory (by default the **parent folder** of this project, i.e. the folder that contains `hospital-flow-agent/`).

| File | Role |
|------|------|
| `observations.csv` | LOINC-coded measurements; we keep numeric rows for vitals/labs listed below and aggregate **per encounter** (mean if multiple values). |
| `patients.csv` | `BIRTHDATE` to compute **age at encounter**. |
| `encounters.csv` | `ENCOUNTERCLASS` and encounter timestamps to join observations and define the **proxy label**. |

**Observation codes mapped to features** (LOINC):

| Code | Feature |
|------|---------|
| 8867-4 | Heart rate |
| 9279-1 | Respiratory rate |
| 2708-6 | Oxygen saturation (%) |
| 8480-6 | Systolic BP |
| 8462-4 | Diastolic BP |
| 8310-5 | Body temperature (°C) |
| 2345-7 | Glucose (mg/dL) |
| 39156-5 | BMI |

Rows are dropped if core fields are missing: **age**, **heart rate**, **SpO₂**, or **systolic BP**.

### Target label (`risk_label`) — important

The label is a **deliberate teaching proxy**, not a validated clinical outcome:

- **`risk_label = 1`** if the encounter’s `ENCOUNTERCLASS` is **`inpatient`** or **`emergency`** (higher-acuity care setting in the synthetic data).
- **`risk_label = 0`** for other classes (e.g. ambulatory, wellness, outpatient).

The model therefore learns patterns associated with **“this encounter type in Synthea”**, not proven ICU escalation or mortality. When you present this project, say clearly that the label is a **proxy for operational acuity** used to demo ML + LLM wiring on public synthetic data.

## What you should see when you run it

### After `python prepare_data.py`

- Console prints row counts before/after dropping incomplete rows and a **`risk_label` value counts** table.
- File created: **`data/patients_training.csv`** (encounter-level features + `risk_label`).

### After `python train.py`

- Printed **classification report** (precision, recall, F1 per class) and **ROC-AUC**.
- On synthetic Synthea-like data, metrics are often **very high** because the proxy label and vitals can be strongly correlated in the generator—treat this as a **pipeline check**, not proof of real-world performance.
- File created: **`models/risk_model.pkl`** (model + feature list + cohort medians/stds + permutation importances for ranking factors in the UI).

### In the Streamlit app (`streamlit run app.py`)

1. Adjust sliders/inputs for age and vitals.
2. **Run risk model** — You should see JSON-like output with:
   - `risk_probability` (0–1)
   - `risk_label` (0/1 from threshold 0.5 on probability)
   - `risk_band` — **Low** (probability below 0.33), **Medium** (0.33 to 0.66), **High** (above 0.66)
   - `top_factors` — a few features ranked by how far they sit from the training median, weighted by permutation importance
3. **Generate LLM explanation** (requires `OPENAI_API_KEY`) — A short structured paragraph: reasons tied to vitals, operational concern level, suggested next **operational** step (no diagnosis).

If the API key is unset, the LLM button stays disabled; the model-only path still works.

## How to run

### 1. Environment

```bash
cd hospital-flow-agent
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
```

Use **Python 3.9+** (3.10–3.12 work well; Conda “base” on Apple Silicon is often 3.12). If `pip install` tries to **compile** a large package from source and fails (for example `pkg_resources` / `setuptools` errors), run `python -m pip install -U pip setuptools wheel` and try again—the updated `requirements.txt` uses ranges so pip can pick **pre-built wheels** for your Python version.

### 2. Point at your Synthea CSV folder (optional)

By default, scripts read CSVs from the **directory above** `hospital-flow-agent/` (the folder that contains both `observations.csv` and this repo).

To use another path:

```bash
export SYNTHEA_DATA_DIR="/absolute/path/to/folder/containing/observations.csv"
```

That folder must contain at least: `observations.csv`, `patients.csv`, `encounters.csv`.

### 3. Build training data and train

```bash
python prepare_data.py
python train.py
```

### 4. OpenAI (optional, for explanations)

**Option A — `.env` file (recommended)**  

In the `hospital-flow-agent/` folder, copy the example file and edit it:

```bash
cp .env.example .env
# Edit `.env` and set OPENAI_API_KEY=... (and optionally OPENAI_MODEL)
```

On startup, `utils.py` loads `hospital-flow-agent/.env` into the environment (via `python-dotenv`). Variables you can set there include `OPENAI_API_KEY`, `OPENAI_MODEL`, and optionally `SYNTHEA_DATA_DIR`. The file `.env` is gitignored so you do not commit secrets.

**Format:** `OPENAI_API_KEY` must be a **single line** — `OPENAI_API_KEY=sk-...` with the full key on that line. A blank `OPENAI_API_KEY=` with the key on the next line is **not** read by `python-dotenv`, so the LLM button stays disabled.

**Option B — shell exports**

```bash
export OPENAI_API_KEY="your-key-here"
export OPENAI_MODEL="gpt-4o-mini"   # optional
```

### 5. Launch the UI

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).

## Repository layout

```text
hospital-flow-agent/
├── README.md
├── .env.example       # copy to `.env` for local secrets
├── app.py
├── prepare_data.py
├── train.py
├── inference.py
├── prompts.py
├── llm_client.py
├── utils.py
├── requirements.txt
├── data/              # generated; gitignored
├── models/            # generated; gitignored
└── notebooks/         # optional; add your own exploration
```

## Attribution

Initial code and documentation in this folder were **drafted with AI assistance** (e.g. Cursor). Treat as a starting point: review, test, and adapt before any serious or regulated use.

## Disclaimer

This software is for **education and demonstration** only. It does not provide medical advice, diagnosis, or treatment recommendations. Do not use it for patient care or operational decisions without proper validation, governance, and clinical oversight.
