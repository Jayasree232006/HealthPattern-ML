# Rare & Common Disease Detector ML Pipeline (Version 0.1.0)

This directory contains the training routines, dataset preprocessing, NLP phrase/negation parser, rare disease matching module, and inference service for the **Rare & Common Disease Detector** experimental educational prototype.

> [!WARNING]
> **Educational & Research Prototype Only**:
> Models and matching algorithms in this repository are trained on small, synthetic, or augmented datasets. They are **not clinically validated**, do not yield calibrated medical diagnosis probabilities, and must never be used for self-diagnosis or clinical decision-making.

---

## 1. Architecture Overview

The pipeline supports three distinct capabilities:
1. **Free-Text Baseline Classifier**:
   - Model: Scikit-learn `Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)), ('clf', LogisticRegression(class_weight='balanced'))])`.
   - Data Source: `gretelai/symptom_to_diagnosis` (trained strictly on the official training split, tested on the test split).
2. **Structured Symptom Classifier**:
   - Model: Scikit-learn `MultinomialNB(alpha=1.0)` trained on 132 binary symptom indicators.
   - Data Source: `kaushil268/disease-prediction-using-machine-learning`.
   - Leakage Prevention & Deduplication: 4,616 duplicated rows were removed down to **36 unique symptom patterns** across the 5 common classes. Train and test partitions are stratified over unique patterns so **zero identical symptom vectors leak between train and test**.
3. **Source-Attributed Rare Condition Matcher**:
   - Covers 4 rare diseases: Systemic Lupus Erythematosus (SLE), Myasthenia Gravis (MG), Wilson Disease, and Hereditary Angioedema (HAE).
   - Profiles built with literature citations from Orphanet, NIH/MedlinePlus/NINDS/NIDDK, and NORD.
  - **Score Separation**: Rare disease similarity scores are computed via symptom overlap + hallmark presentation bonuses and are **kept strictly separate from experimental common-condition pattern scores**.

---

## 2. Five Shared Disease Classes & Explicit Mapping

The initial baseline models focus on the 5 conditions present in both raw datasets:

| Canonical ID | Display Name | Kaggle Raw Label (`prognosis`) | Gretel Raw Label (`output_text`) |
| :--- | :--- | :--- | :--- |
| `common_cold` | Common Cold | `Common Cold` | `common cold` |
| `allergy` | Allergy | `Allergy` | `allergy` |
| `migraine` | Migraine | `Migraine` | `migraine` |
| `gerd` | Gastroesophageal Reflux Disease (GERD) | `GERD` | `gastroesophageal reflux disease` |
| `urinary_tract_infection` | Urinary Tract Infection (UTI) | `Urinary tract infection` | `urinary tract infection` |

*Note: Raw label variations such as trailing whitespace (`Diabetes `, `Hypertension `) and typos in non-target classes (`Peptic ulcer diseae`, `Osteoarthristis`) are documented and safely ignored by strict dictionary mapping.*

---

## 3. Data Cleaning & Partition Statistics

### Kaggle Deduplication Analysis
- **Raw Rows across 5 classes**: 600 rows (120 per disease).
- **Deduplicated Unique Patterns**: **36 unique symptom vectors**.
  - `migraine`: 10 unique patterns
  - `common_cold`: 9 unique patterns
  - `gerd`: 7 unique patterns
  - `allergy`: 5 unique patterns
  - `urinary_tract_infection`: 5 unique patterns
- **Data Leakage Elimination**: In the raw dataset, the official 5-row test set was 100% identical to patterns in the training set. In our pipeline, unique patterns were split using `StratifiedShuffleSplit` (train = 27 unique patterns, test = 9 unique patterns) ensuring **0 overlapping patterns**.
- **Dropped Column**: The entirely null `Unnamed: 133` artifact column was dropped.

### Gretel Train/Test Split
- Official training examples (5 classes): 189 rows (`allergy`: 40, `common cold`: 39, `gerd`: 39, `urinary tract infection`: 39, `migraine`: 32).
- Official testing examples (5 classes): 49 rows (`allergy`: 10, `common cold`: 10, `gerd`: 10, `migraine`: 10, `urinary tract infection`: 9).
- Feature vectorizers are fitted **strictly on training data**.

---

## 4. Evaluation Results

### Text Classifier (TF-IDF + Logistic Regression)
- **Accuracy**: 95.92% (47 / 49 test samples)
- **Macro-F1**: 0.9599 | **Weighted-F1**: 0.9591
- **Per-Class Breakdown**:
  - `allergy`: Precision 0.90, Recall 0.90, F1 0.90, Support 10
  - `common_cold`: Precision 0.91, Recall 1.00, F1 0.95, Support 10
  - `gerd`: Precision 1.00, Recall 0.90, F1 0.95, Support 10
  - `migraine`: Precision 1.00, Recall 1.00, F1 1.00, Support 10
  - `urinary_tract_infection`: Precision 1.00, Recall 1.00, F1 1.00, Support 9
- **Confusion Matrix** (rows: true, cols: pred `[allergy, common_cold, gerd, migraine, uti]`):
  ```text
  [[ 9,  1,  0,  0,  0],
   [ 0, 10,  0,  0,  0],
   [ 1,  0,  9,  0,  0],
   [ 0,  0,  0, 10,  0],
   [ 0,  0,  0,  0,  9]]
  ```

### Structured Symptom Classifier (Multinomial NB on Deduplicated Patterns)
- **Accuracy**: 100% on the 9 unseen unique test patterns.
- **Macro-F1**: 1.00 | **Support**: 9 unique symptom vectors.
- *Notice*: While accuracy is 1.0 on this test set, support is very small (9 patterns) because the original dataset only contained 36 unique combinations for these 5 diseases.

---

## 5. Backend Prediction Service Integration

The ML pipeline implements the contract proposed in `backend/PREDICTION_SERVICE_PROPOSAL.md`.

### Artifacts in `ml_pipeline/artifacts/`
1. `text_classifier.joblib`: Serialized TF-IDF vectorizer and Logistic Regression classifier.
2. `symptom_classifier.joblib`: Dictionary with `{"model": MultinomialNB, "feature_names": [...]}`.
3. `metadata.json`: Telemetry, dataset provenance, per-class evaluation metrics, and disclaimer.

### Python Interface for `PredictionService`

```python
from ml_pipeline.src.inference import predict_condition, CareAtlasPredictor, SymptomInput

# Option A: Functional call
result = predict_condition(
    free_text_symptoms="Severe throbbing headache with nausea and light sensitivity, but no fever",
    structured_symptoms=["headache", "nausea"]
)

# Option B: Object-oriented predictor instance
predictor = CareAtlasPredictor()
inp = SymptomInput(
    free_text_symptoms="Patient has malar butterfly rash on face, fatigue, and joint pain",
    structured_symptoms=["fatigue", "joint_pain", "skin_rash"]
)
res = predictor.predict(inp)
```

### Returned Output Schema
```json
{
  "model_version": "0.1.0",
  "predictions": [
    {
      "condition": "Migraine",
      "confidence": 0.4111,
      "description": "Predicted common condition (migraine)."
    },
    {
      "condition": "Common Cold",
      "confidence": 0.219,
      "description": "Predicted common condition (common_cold)."
    }
  ],
  "metadata": {
    "mode_used": "ensemble_text_and_symptom",
    "extracted_affirmed_symptoms": ["headache", "nausea"],
    "extracted_negated_symptoms": ["high_fever"],
    "total_affirmed_features": 2,
    "disclaimer": "Educational research prototype only. Scores do not represent medical probabilities or diagnostic advice."
  },
  "rare_condition_matches": [
    {
      "condition_id": "systemic_lupus_erythematosus",
      "display_name": "Systemic Lupus Erythematosus (SLE)",
      "orpha_code": "ORPHA:536",
      "icd10_code": "M32",
      "match_score": 0.675,
      "matched_symptoms": ["fatigue", "joint_pain", "skin_rash"],
      "sources": [
        {"authority": "Orphanet", "identifier": "ORPHA:536", "url": "https://www.orpha.net/consor/cgi-bin/OC_Exp.php?Lng=GB&Expert=536"}
      ],
      "disclaimer": "Educational prototype matching score based on clinical literature. This is not a diagnostic tool and does not provide clinical probability."
    }
  ]
}
```

---

## 6. How to Reproduce & Run Tests

```bash
# 1. Activate isolated virtual environment
cd /mnt/storage/rareDis/health-ai-assistant/ml_pipeline
source .venv/bin/activate

# 2. Run training script and generate artifacts
python -c "from src.train import build_and_save_artifacts; build_and_save_artifacts()"

# 3. Run pytest test suite
pytest -v
```
