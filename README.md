# Rare & Common Disease Detector

Rare & Common Disease Detector is an experimental educational prototype for exploring symptom patterns. It accepts a free-text symptom description, uses AI-assisted symptom understanding when Groq is configured (with a local rule-based fallback), and lets the user review extracted symptoms before requesting results. Results present experimental ML pattern scores for five common conditions separately from literature-based hallmark matches for four selected rare conditions.

> [!IMPORTANT]
> **Educational prototype only. This is not a medical diagnostic tool.** The system is not clinically validated and does not provide diagnosis, treatment recommendations, or medical advice. Common-condition scores are experimental model outputs, not clinical probabilities. Rare-condition matches are heuristic literature indicators, not diagnoses or diagnostic probabilities. Consult a qualified healthcare professional about health concerns.

## Project Structure

- `frontend/`: React 19, TypeScript, and Vite application for symptom entry, review, and separate common/rare results.
- `backend/`: FastAPI API for request validation, symptom understanding, and prediction coordination.
- `ml_pipeline/`: Scikit-learn models and the curated literature-based rare-condition matcher.

The common-condition model covers common cold, allergy, migraine, gastroesophageal reflux disease (GERD), and urinary tract infection (UTI). The separate rare-condition matcher includes systemic lupus erythematosus, myasthenia gravis, Wilson disease, and hereditary angioedema.

## Requirements

- Python 3.12 or newer
- Node.js 18 or newer and npm
- Git, if cloning the repository

## Setup: Ubuntu/Linux

Run the backend and frontend in separate terminals from the project root.

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r ../ml_pipeline/requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

To run backend tests, stop the server with Ctrl+C, then run `pytest -v` from `backend/`.

### Frontend

```bash
cd frontend
npm install
npm test
npm run build
npm run dev
```

Open the local URL printed by Vite (normally `http://localhost:5173`). The frontend's default API URL points to the local backend at port 8000. If the backend runs elsewhere, set `VITE_API_BASE_URL` in a local `frontend/.env` file.

## Setup: Windows PowerShell

Run the backend and frontend in separate PowerShell terminals from the project root.

### Backend

```powershell
Set-Location backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r ..\ml_pipeline\requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

If PowerShell blocks virtual-environment activation, use a session-only policy with `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again. To run backend tests, stop the server with Ctrl+C and run `pytest -v` from `backend/`.

### Frontend

```powershell
Set-Location frontend
npm install
npm test
npm run build
npm run dev
```

Open the local URL printed by Vite (normally `http://localhost:5173`). The frontend's default API URL points to the local backend at port 8000. If the backend runs elsewhere, set `VITE_API_BASE_URL` in a local `frontend/.env` file.

## Groq Configuration

Groq is optional. When it is not configured or unavailable, the backend uses its local rule-based symptom-understanding fallback. To enable Groq, configure `GROQ_API_KEY` privately in `backend/.env` or in the backend process environment. Never put a real key in frontend variables, source files, screenshots, logs, or version control. The repository's `.env.example` files contain placeholders only; replace them locally and keep `.env` files untracked.

## Workflow and API

The frontend sends symptom descriptions to `POST /api/v1/predict`. The user reviews the extracted symptom assertions and can affirm, deny, mark uncertain or historical, or remove symptoms. Confirmation sends only the remaining affirmed canonical symptom tags in a fresh request; it does not rerun Groq on edited labels or resend the original narrative. Common-condition pattern scores and rare-condition literature matches are displayed in separate sections. API and connection errors are shown directly; the frontend does not substitute mock predictions.

- `GET /health`: basic service health check.
- `GET /api/v1/health`: versioned API and model availability.
- `POST /api/v1/predict`: symptom understanding and pattern results.

## Data and Limitations

The text classifier uses the Gretel AI symptom-to-diagnosis dataset, and the structured symptom classifier uses selected classes from the Kaggle disease-prediction dataset. The rare-condition matcher uses curated hallmark profiles and source references including Orphanet, NIH, and NORD. These data and methods are limited in scope; model outputs may be incomplete or incorrect. A score or match must not be used to make health decisions or delay professional care.
