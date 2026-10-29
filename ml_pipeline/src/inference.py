"""Prediction service interface compatible with backend/PREDICTION_SERVICE_PROPOSAL.md.

Implements unified inference taking free_text_symptoms and/or structured_symptoms,
handling NLP phrase matching + negation, running the appropriate classifier(s),
and returning ranked predicted conditions alongside separate rare condition matches.
"""

import os
import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set
import joblib
import numpy as np

from .constants import SHARED_5_CLASSES, DISPLAY_LABELS
from .nlp_rules import extract_symptoms_with_negation, vector_from_symptoms, KAGGLE_SYMPTOM_FEATURES
from .rare_matcher import RareConditionMatcher, RareMatchResult


@dataclass
class SymptomInput:
    free_text_symptoms: Optional[str] = None
    structured_symptoms: Optional[List[str]] = None
    age_group: Optional[str] = None
    duration_value: Optional[int] = None
    duration_unit: Optional[str] = None


@dataclass
class PredictedCondition:
    condition: str
    confidence: float
    description: Optional[str] = None


@dataclass
class InferenceResult:
    model_version: str
    predictions: List[PredictedCondition]
    metadata: Dict[str, Any]
    rare_condition_matches: List[Dict[str, Any]] = field(default_factory=list)


class CareAtlasPredictor:
    """Predictor wrapping text classifier, structured classifier, and rare condition matcher."""

    def __init__(self, artifacts_dir: Optional[str] = None):
        if artifacts_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            artifacts_dir = os.path.join(base_dir, "artifacts")
        self.artifacts_dir = artifacts_dir
        self.text_model = None
        self.symptom_model = None
        self.feature_names: List[str] = KAGGLE_SYMPTOM_FEATURES
        self.metadata: Dict[str, Any] = {}
        self.rare_matcher = RareConditionMatcher()
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load serialized joblib models and metadata if available."""
        meta_path = os.path.join(self.artifacts_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        text_path = os.path.join(self.artifacts_dir, "text_classifier.joblib")
        if os.path.exists(text_path):
            self.text_model = joblib.load(text_path)

        symptom_path = os.path.join(self.artifacts_dir, "symptom_classifier.joblib")
        if os.path.exists(symptom_path):
            symptom_data = joblib.load(symptom_path)
            if isinstance(symptom_data, dict):
                self.symptom_model = symptom_data["model"]
                self.feature_names = symptom_data.get("feature_names", self.feature_names)
            else:
                self.symptom_model = symptom_data

    def predict(self, input_data: SymptomInput) -> InferenceResult:
        """Run prediction pipeline based on provided inputs."""
        free_text = (input_data.free_text_symptoms or "").strip()
        structured = set(input_data.structured_symptoms or [])

        # NLP symptom extraction and negation detection from free text
        extracted_affirmed, extracted_negated = extract_symptoms_with_negation(
            free_text, feature_set=set(self.feature_names)
        )

        # Merge affirmed structured symptoms with extracted symptoms, excluding negated
        all_affirmed: Set[str] = (structured.union(extracted_affirmed)) - extracted_negated

        # Predict common condition candidates
        predictions: List[PredictedCondition] = []
        mode_used = "none"

        # Strategy:
        # If free text is present and text_model loaded, get text model probabilities
        # If structured symptoms present and symptom_model loaded, get symptom probabilities
        # Combine if both present, else use available
        has_text = bool(free_text and self.text_model is not None)
        has_symptoms = bool(all_affirmed and self.symptom_model is not None)

        if has_text and has_symptoms:
            mode_used = "ensemble_text_and_symptom"
            text_probs = self.text_model.predict_proba([free_text])[0]
            text_classes = list(self.text_model.classes_)

            feat_vec = vector_from_symptoms(all_affirmed, self.feature_names)
            symp_probs = self.symptom_model.predict_proba([feat_vec])[0]
            symp_classes = list(self.symptom_model.classes_)

            combined_probs: Dict[str, float] = {}
            for cls_name in SHARED_5_CLASSES:
                p_text = float(text_probs[text_classes.index(cls_name)]) if cls_name in text_classes else 0.0
                p_symp = float(symp_probs[symp_classes.index(cls_name)]) if cls_name in symp_classes else 0.0
                combined_probs[cls_name] = round((p_text + p_symp) / 2.0, 4)

            for cls_name, score in sorted(combined_probs.items(), key=lambda x: x[1], reverse=True):
                predictions.append(
                    PredictedCondition(
                        condition=DISPLAY_LABELS.get(cls_name, cls_name),
                        confidence=score,
                        description=f"Predicted common condition ({cls_name})."
                    )
                )

        elif has_text:
            mode_used = "text_classifier"
            text_probs = self.text_model.predict_proba([free_text])[0]
            text_classes = list(self.text_model.classes_)
            ranked_idx = np.argsort(text_probs)[::-1]
            for idx in ranked_idx:
                cls_name = text_classes[idx]
                predictions.append(
                    PredictedCondition(
                        condition=DISPLAY_LABELS.get(cls_name, cls_name),
                        confidence=round(float(text_probs[idx]), 4),
                        description=f"Predicted common condition ({cls_name})."
                    )
                )

        elif has_symptoms:
            mode_used = "symptom_classifier"
            feat_vec = vector_from_symptoms(all_affirmed, self.feature_names)
            symp_probs = self.symptom_model.predict_proba([feat_vec])[0]
            symp_classes = list(self.symptom_model.classes_)
            ranked_idx = np.argsort(symp_probs)[::-1]
            for idx in ranked_idx:
                cls_name = symp_classes[idx]
                predictions.append(
                    PredictedCondition(
                        condition=DISPLAY_LABELS.get(cls_name, cls_name),
                        confidence=round(float(symp_probs[idx]), 4),
                        description=f"Predicted common condition ({cls_name})."
                    )
                )
        else:
            mode_used = "fallback_unmatched"
            # Return uniform/empty if no input features could be extracted
            for cls_name in SHARED_5_CLASSES:
                predictions.append(
                    PredictedCondition(
                        condition=DISPLAY_LABELS.get(cls_name, cls_name),
                        confidence=0.2,
                        description="Default baseline distribution (no features recognized)."
                    )
                )

        # 3. Rare condition matching (kept completely separate from classifier probabilities)
        rare_matches = self.rare_matcher.match(
            affirmed_symptoms=all_affirmed,
            free_text=free_text,
            top_k=4
        )
        rare_output = [
            {
                "condition_id": r.condition_id,
                "display_name": r.display_name,
                "orpha_code": r.orpha_code,
                "icd10_code": r.icd10_code,
                "match_score": r.match_score,
                "matched_symptoms": r.matched_symptoms,
                "sources": r.sources,
                "disclaimer": r.disclaimer,
            }
            for r in rare_matches
        ]

        metadata = {
            "mode_used": mode_used,
            "extracted_affirmed_symptoms": sorted(list(extracted_affirmed)),
            "extracted_negated_symptoms": sorted(list(extracted_negated)),
            "total_affirmed_features": len(all_affirmed),
            "disclaimer": (
                "Educational research prototype only. Scores do not represent medical probabilities or diagnostic advice."
            )
        }

        return InferenceResult(
            model_version=self.metadata.get("pipeline_version", "0.1.0"),
            predictions=predictions,
            metadata=metadata,
            rare_condition_matches=rare_output
        )


def predict_condition(
    free_text_symptoms: Optional[str] = None,
    structured_symptoms: Optional[List[str]] = None,
    artifacts_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience functional interface for backend consumption."""
    predictor = CareAtlasPredictor(artifacts_dir=artifacts_dir)
    inp = SymptomInput(
        free_text_symptoms=free_text_symptoms,
        structured_symptoms=structured_symptoms
    )
    res = predictor.predict(inp)
    return {
        "model_version": res.model_version,
        "predictions": [
            {"condition": p.condition, "confidence": p.confidence, "description": p.description}
            for p in res.predictions
        ],
        "metadata": res.metadata,
        "rare_condition_matches": res.rare_condition_matches
    }
