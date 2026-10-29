# Care Atlas (Health AI Assistant)

Care Atlas is an educational and research prototype exploring multimodal symptom pattern analysis. It pairs machine learning classification for common condition patterns with literature-grounded hallmark matching for rare diseases.

> [!IMPORTANT]
> **Educational & Research Prototype Only**:
> Care Atlas is **not clinically validated** and is not a certified medical device or diagnostic instrument. It does not provide medical diagnosis, clinical likelihoods, or treatment recommendations.
> - **Common condition scores** are experimental, uncalibrated model outputs derived from small reference datasets.
> - **Rare condition matches** are literature-based heuristic matching indicators citing authoritative biomedical sources (Orphanet, NIH/MedlinePlus, NORD), not clinical probabilities.
> - Always consult a licensed healthcare professional for medical concerns.

---

## 1. System Architecture

The application is structured into three decoupled sub-projects:

- **`frontend/`**: Single-page application built with React 19, TypeScript, and Vite. Implements natural-language symptom intake, a review of backend symptom assertions, distinct common vs. rare result presentation, and prominent disclaimers.
- **`backend/`**: FastAPI REST API service. Handles request validation, CORS management, privacy-preserving in-memory inference coordination, and health endpoints.
- **`ml_pipeline/`**: Scikit-learn inference pipeline. Includes a TF-IDF + Logistic Regression text classifier, a Multinomial Naive Bayes structured symptom classifier (trained on deduplicated patterns to eliminate data leakage), and a curated hallmark matcher for rare condition profiles.

---

## 2. API Endpoints

The backend serves the following endpoints:

| Method | Path | Description | Sample Status |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Unversioned legacy root health probe | `200 OK` (`{"status": "ok"}`) |
| `GET` | `/api/v1/health` | Versioned API and ML model availability status | `200 OK` (`model_status: "available"` or `"unavailable"`) |
| `POST` | `/api/v1/predict` | Predict condition patterns from symptoms | `200 OK` (or `422 Unprocessable Entity` on invalid inputs) |

### Sample `POST /api/v1/predict` Request

```json
{
  "free_text_symptoms": "Experiencing frequent headaches, throbbing pain, and nausea after work",
  "structured_symptoms": ["headache", "nausea"],
  "age_group": "adult",
  "duration": {
    "value": 3,
    "unit": "days"
  }
}
```

---

## 3. Installation & Local Setup

### Prerequisites
- Python 3.12+
- Node.js 18+ and npm

### Backend Setup

```bash
cd backend

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run backend test suite (12 tests)
pytest -v

# Start FastAPI development server (runs on http://127.0.0.1:8000)
uvicorn app.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend

# Install node dependencies
npm install

# Run frontend build verification
npm run build

# Run frontend interaction tests
npm test

# Configure the backend URL if it differs from localhost:8000
cp .env.example .env

# Start Vite development server (runs on http://localhost:5173)
npm run dev
```

The frontend always sends requests to the live `POST /api/v1/predict` endpoint and displays connection or API errors without substituting mock data. The first request returns symptom understanding for review. After confirmation, the frontend sends a second request containing only canonical symptom keys still labeled affirmed as `structured_symptoms`; removed, denied, uncertain, and historical symptoms are excluded, and the original narrative is not resent. The second request drives the displayed common and rare pattern results. Confirmation does not rerun Groq on edits: the backend processes these confirmed keys as structured input.

---

## 4. Dataset Attribution & Literature Provenance

Care Atlas utilizes both public academic reference datasets and curated biomedical authority databases:

1. **Gretel AI Symptom-to-Diagnosis Dataset** (`gretelai/symptom_to_diagnosis`):
   - Used for the natural language symptom text classifier.
   - Licensed under Apache 2.0.
   - Filtered strictly to shared target classes and evaluated on the official test partition.
2. **Kaggle Disease Prediction via ML** (`kaushil268/disease-prediction-using-machine-learning`):
   - Used for structured symptom binary indicator training.
   - Deduplicated from 4,616 redundant records down to 36 unique clinical symptom patterns to prevent train/test data leakage.
3. **Biomedical Literature Knowledge Base for Rare Conditions**:
   - Curated hallmark signs and clinical profiles for:
     - **Systemic Lupus Erythematosus (SLE)**: Orphanet (`ORPHA:536`), ICD-10 `M32`.
     - **Myasthenia Gravis (MG)**: Orphanet (`ORPHA:589`), ICD-10 `G70.0`.
     - **Wilson Disease**: Orphanet (`ORPHA:905`), ICD-10 `E83.0`.
     - **Hereditary Angioedema (HAE)**: Orphanet (`ORPHA:913`), ICD-10 `D84.1`.
   - Authoritative sources include Orphanet, NIH NINDS, NIH NIDDK, and NORD.

---

## 5. Model Limitations & Safety Guardrails

- **Experimental Scores vs Literature Indicators**:
  - **Common condition scores** (e.g., 74%) represent experimental classifier rankings based on synthetic/augmented distributions. They are **not clinical likelihoods**.
  - **Rare condition indicators** (e.g., 68%) represent heuristic symptom hallmark matching against peer-reviewed profiles. They are **not diagnostic probabilities**.
- **Data Privacy & Ephemerality**:
  - User-submitted symptom text and tags are processed entirely in memory.
  - Symptom narratives are never logged to console or persistent storage.
- **Graceful Fault Isolation**:
  - If model artifacts are missing or fail to deserialize, the backend returns an explicit `status: "model_unavailable"` response rather than crashing or returning misleading synthetic predictions.
  - Internal exceptions are shielded: stack traces and filesystem paths are never returned to clients.

---

## 6. Project Demonstration Checklist

Use this checklist during the live project demonstration:

- [ ] **Live Prediction Flow**:
  - Enter description: *"Experiencing severe throbbing headache, light sensitivity, and nausea for 2 days"*.
  - Observe that common condition predictions display under **Common Condition Patterns** labeled as *experimental scores*.
- [ ] **Rare Condition Literature Matches**:
  - Enter or check symptoms like *"skin rash, fatigue, joint pain"* or mention *"butterfly rash"*.
  - Observe that **Systemic Lupus Erythematosus (SLE)** is displayed under **Rare Condition Literature Matches** with Orpha code (`ORPHA:536`), matched hallmark tags, and external citation link opening Orphanet in a safe tab (`rel="noopener noreferrer"`).
- [ ] **Input Validation Handling**:
  - Clear both text and checkboxes, then click submit.
  - Observe the client validation banner stating at least one symptom must be provided.
- [ ] **Model Unavailable Handling**:
  - Stop the backend or simulate unavailable models.
  - Observe that the frontend displays the dedicated amber alert banner explaining that the ML service is unavailable, maintaining clean UI stability.
- [ ] **Disclaimer Visibility**:
  - Confirm the top-level prototype notice, candidate card notes, and bottom sticky disclaimer remain visible throughout all workflows.

