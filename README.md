# Bias AI System

An AI fairness audit system combining statistical bias detection with LLM-powered interpretability and a modern UI dashboard.

## Architecture

```text
User -> Frontend (shadcn UI)
     -> API Call
     -> Backend (FastAPI)
         -> Preprocessing
         -> LLM (Intent)
         -> Fairness Engine
         -> Mitigation Engine
         -> LLM (Explanation)
     -> Response
     -> Frontend Dashboard (shadcn)
```

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

## API

### `POST /analyze`

Multipart form fields:
- `file`: CSV file
- `target`: target column name
- `sensitive`: sensitive attribute column name
- `query`: optional text like `check bias`

Response includes:
- `bias`, `dp_diff`, `eo_diff`
- `group_metrics`
- `suggestions`
- `explanation`, `summary`, `report_text`

## Notes

- LLM role: intent classification and explanation only.
- Backend role: fairness mathematics and mitigation rules.
- Frontend role: visualization and interaction.
