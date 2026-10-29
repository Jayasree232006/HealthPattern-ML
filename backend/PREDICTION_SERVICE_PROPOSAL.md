"""# ML Model & Prediction Service Interface Proposal

This document outlines the proposed contract between the Backend API (`PredictionService`) and the ML Pipeline (`ml_pipeline/`).
It serves as a coordination specification to align artifacts, schemas, preprocessing, and inference without guessing or hardcoding unconfirmed model files.

---

## 1. Context & Goal
The backend provides a RESTful interface for the **Health AI Assistant** educational prototype.
Currently, when a request is made to `POST /api/v1/predict`, the backend validates the request schema and returns a `model_unavailable` response because no verified ML model artifact is loaded.

To integrate the ML pipeline cleanly:
1. The ML pipeline training script must produce standardized artifact(s).
2. The ML pipeline provides a loading/inference function or class that can be called by `PredictionService`.
3. Clear input and output schemas are agreed upon.

---

## 2. Input Contract (API -> ML Pipeline)

The backend passes structured input data to the ML prediction layer:

```python
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class SymptomInput:
    free_text_symptoms: Optional[str]        # e.g., "Persistent cough, sore throat, mild fever"
    structured_symptoms: Optional[List[str]] # e.g., ["cough", "fever", "fatigue"]
    age_group: Optional[str]                 # e.g., "adult" (infant, child, adolescent, adult, senior)
    duration_value: Optional[int]            # e.g., 3
    duration_unit: Optional[str]             # e.g., "days" (hours, days, weeks, months, years)
```

### Questions for ML Agent:
- Does the initial model support free-text, structured symptom tags, or both?
- If free-text is used, will TF-IDF/CountVectorizer or a Transformer/embeddings model be bundled in the pipeline artifact (e.g., `sklearn.pipeline.Pipeline`)?
- Are demographic features (`age_group`, `duration`) used in version 1 of the model, or should they be passed as optional auxiliary features?

---

## 3. Output Contract (ML Pipeline -> API)

The backend expects the model to return ranked condition candidates with confidence scores:

```python
@dataclass
class PredictedCondition:
    condition: str                           # Condition label, e.g. "Acute Bronchitis"
    confidence: float                        # Confidence score in [0.0, 1.0] (e.g. from predict_proba)
    description: Optional[str] = None        # Optional short summary or educational note

@dataclass
class InferenceResult:
    model_version: str                       # e.g., "v0.1.0-tfidf-rf"
    predictions: List[PredictedCondition]    # Ranked list of top-k conditions
    metadata: dict                           # Extra telemetry (e.g. matched symptoms count, latency)
```

---

## 4. Proposed Artifact Specification

### Recommended Options:
- **Option A (Recommended: Sklearn Pipeline `joblib` / `cloudpickle`):**
  - Path: `backend/app/models/model_pipeline.joblib` (or configurable via `MODEL_ARTIFACT_PATH` env var).
  - Encapsulates: Preprocessing/vectorizer + Classifier (`predict_proba` support).
  - Accompanied by: `metadata.json` containing:
    ```json
    {
      "model_version": "0.1.0",
      "model_type": "MultinomialNB",
      "created_at": "2026-10-27T00:00:00Z",
      "target_classes": ["Common Cold", "Influenza", "Allergic Rhinitis"],
      "supported_features": ["free_text_symptoms", "structured_symptoms"]
    }
    ```

- **Option B (Python Callable Module in `ml_pipeline`):**
  - Backend imports an inference wrapper (e.g., `from ml_pipeline.inference import predict_condition`).
  - Decouples artifact format handling directly inside the ML pipeline package.

---

## 5. Error Handling & Safety
1. **Fallback if artifact missing / corrupt:**
   - Backend gracefully returns `PredictResponse(status="model_unavailable", message=...)`.
   - Backend will never crash on startup if the model file is not yet generated.
2. **Confidence Threshold:**
   - Predictions below a given threshold (e.g., `< 0.15`) can be filtered or flagged as low confidence.
3. **Medical Disclaimer:**
   - Every response retains the mandatory disclaimer:
     `"This prototype is strictly for educational purposes and does not provide medical diagnosis or advice."`
