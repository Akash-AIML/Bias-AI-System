# Bias AI System

AI Audit System for bias detection, traceability, and mitigation simulation with a modern UI dashboard.

## Architecture

```text
User -> Frontend (Next.js + shadcn UI)
     -> API Call
     -> Backend (FastAPI)
         -> Preprocessing
         -> Fairness Engine (deterministic)
         -> Traceability + Drift
         -> Mitigation Simulation
         -> LLM (narrative + compliance language)
     -> Response
     -> Frontend Dashboard
```

## Features

- Bias Severity Index (BSI) with core fairness metrics.
- Proxy discrimination tracing to identify sensitive proxies.
- Temporal drift detection for fairness changes over time.
- Free-form text bias analysis using local embeddings.
- Mitigation simulation with before/after comparisons.
- Compliance-grade narrative and report output.

## Project Structure

```text
frontend/
backend/
shared/
```

## Backend Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Optional environment variable (frontend):

```bash
export NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

## API

### `POST /analyze`

Multipart form fields:
- `file`: CSV file
- `target`: target column name
- `sensitive`: sensitive attribute column name
- `query`: optional audit goal text
- `prediction_column`: optional prediction column name
- `org_name`: optional name
- `dataset_name`: optional dataset name
- `time_column`: optional time column name
- `text_columns`: optional comma-separated text column names

Response includes (non-exhaustive):
- `bias_severity`, `dp_diff`, `eo_diff`
- `group_metrics`
- `proxy_traces`
- `temporal_drift`
- `text_bias`
- `mitigation_simulations`
- `summary`, `explanation`, `report_text`

## Troubleshooting

- CSVs must include header names on the first row.
- `target` and `sensitive` columns are required to run the audit.
- Large files may take longer; wait for the progress bar to finish.
- If the frontend cannot reach the API, set `NEXT_PUBLIC_API_BASE_URL`.
- For text analysis, provide comma-separated column names in `text_columns`.

## Notes

- LLM role: narrative and compliance language only.
- Backend role: fairness math, traceability, drift, and mitigation simulation.
- Frontend role: upload, configuration, and audit dashboard views.
