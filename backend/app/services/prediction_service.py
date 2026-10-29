"""Prediction service interface and model coordinator.

Connects FastAPI backend to the Care Atlas ML Pipeline (ml_pipeline).
Handles path resolution, model caching, safe error boundaries, and response mapping.
"""

import os
import sys
import logging
from typing import Optional, List, Dict, Any

from app.config import get_settings
from app.schemas import (
    PredictRequest,
    PredictResponse,
    ConditionPrediction,
    RareConditionMatch,
    RareConditionSource,
)

logger = logging.getLogger(__name__)


def _resolve_paths() -> tuple[Optional[str], Optional[str]]:
    """Determine the workspace ml_pipeline package root and artifacts directory."""
    settings = get_settings()

    # If explicitly configured via environment variable / settings
    custom_dir = settings.model_artifacts_dir
    if custom_dir and os.path.isdir(custom_dir):
        artifacts_dir = custom_dir
        pipeline_dir = os.path.dirname(os.path.normpath(custom_dir))
        return pipeline_dir, artifacts_dir

    # Otherwise discover relative to the current file / repository structure
    # This file is: backend/app/services/prediction_service.py
    backend_app_services = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(os.path.dirname(backend_app_services))
    repo_root = os.path.dirname(backend_dir)

    candidate_pipeline_dirs = [
        os.path.join(repo_root, "ml_pipeline"),
        os.path.join(os.getcwd(), "ml_pipeline"),
        os.path.join(os.getcwd(), "..", "ml_pipeline"),
    ]

    for p_dir in candidate_pipeline_dirs:
        p_norm = os.path.normpath(p_dir)
        art_dir = os.path.join(p_norm, "artifacts")
        if os.path.isdir(art_dir):
            return p_norm, art_dir

    return None, None


class PredictionService:
    """Manages ML inference, artifact discovery, and safety boundaries."""

    def __init__(self, model_artifacts_dir: Optional[str] = None):
        self._custom_artifacts_dir = model_artifacts_dir
        self._predictor = None
        self._is_loaded = False
        self._load_error: Optional[str] = None
        self._artifacts_dir: Optional[str] = None

        self._init_model()

    def _init_model(self) -> None:
        """Discover artifacts, import CareAtlasPredictor, and load model instances."""
        try:
            if self._custom_artifacts_dir:
                self._artifacts_dir = self._custom_artifacts_dir
                pipeline_dir = os.path.dirname(os.path.normpath(self._custom_artifacts_dir))
            else:
                pipeline_dir, self._artifacts_dir = _resolve_paths()

            if not self._artifacts_dir or not os.path.isdir(self._artifacts_dir):
                self._is_loaded = False
                self._load_error = "Model artifacts directory not found."
                logger.warning(self._load_error)
                return

            # Verify required artifact files exist before attempting import/loading
            meta_file = os.path.join(self._artifacts_dir, "metadata.json")
            text_model_file = os.path.join(self._artifacts_dir, "text_classifier.joblib")
            symp_model_file = os.path.join(self._artifacts_dir, "symptom_classifier.joblib")

            if not (os.path.isfile(meta_file) and os.path.isfile(text_model_file) and os.path.isfile(symp_model_file)):
                self._is_loaded = False
                self._load_error = "One or more required model artifacts are missing."
                logger.warning(self._load_error)
                return

            # Ensure parent of ml_pipeline is in sys.path so 'ml_pipeline' package can be imported
            repo_root = os.path.dirname(pipeline_dir)
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            if pipeline_dir not in sys.path:
                sys.path.insert(0, pipeline_dir)

            from ml_pipeline.src.inference import CareAtlasPredictor

            # Instantiate and cache predictor
            predictor = CareAtlasPredictor(artifacts_dir=self._artifacts_dir)
            if predictor.text_model is None or predictor.symptom_model is None:
                self._is_loaded = False
                self._load_error = "Artifact files exist but models failed to deserialize."
                logger.error(self._load_error)
                return

            self._predictor = predictor
            self._is_loaded = True
            self._load_error = None
            logger.info("Care Atlas ML predictor successfully initialized from %s", self._artifacts_dir)

        except Exception as e:
            self._is_loaded = False
            self._load_error = f"Error initializing ML prediction model: {type(e).__name__}"
            logger.exception("Failed to initialize PredictionService")

    @property
    def is_available(self) -> bool:
        return self._is_loaded and self._predictor is not None

    def predict(self, request: PredictRequest) -> PredictResponse:
        """Generate condition predictions for the given request.

        Safely maps request symptoms through Groq AI / rule-based symptom understanding,
        then queries the ML predictor. If models are unavailable, returns model_unavailable
        while preserving structured symptom understanding.
        """
        from app.services.groq_service import groq_service
        from app.schemas import SymptomAssertion

        # 1. Run symptom understanding on user narrative and structured input
        symptom_understanding = groq_service.analyze_symptoms(
            free_text=request.free_text_symptoms,
            structured_symptoms=request.structured_symptoms,
        )

        # 2. Extract strictly affirmed canonical symptoms for classifier input
        # Note: Negated, uncertain, or historical symptoms must NOT be converted to affirmed symptoms
        affirmed_symptoms_from_ai: list[str] = [
            s.symptom_key
            for s in symptom_understanding.symptoms
            if s.assertion == SymptomAssertion.AFFIRMED
        ]

        # Combine with any user-provided structured symptoms
        combined_structured = list(set((request.structured_symptoms or []) + affirmed_symptoms_from_ai))

        if not self.is_available:
            return PredictResponse(
                status="model_unavailable",
                model_version=None,
                predictions=[],
                rare_condition_matches=[],
                symptom_understanding=symptom_understanding,
                message=self._load_error or "ML prediction service is currently unavailable.",
                disclaimer="This prototype is strictly for educational purposes and does not provide medical diagnosis or advice.",
                metadata={
                    "received_free_text": bool(request.free_text_symptoms),
                    "received_structured_count": len(request.structured_symptoms or []),
                    "has_age_group": request.age_group is not None,
                    "has_duration": request.duration is not None,
                    "symptom_understanding_engine": symptom_understanding.engine_used,
                    "clarification_needed": symptom_understanding.clarification_needed,
                },
            )

        try:
            from ml_pipeline.src.inference import SymptomInput

            # Map request to pipeline dataclass using original free_text and combined verified symptoms
            symptom_input = SymptomInput(
                free_text_symptoms=request.free_text_symptoms,
                structured_symptoms=combined_structured,
                age_group=request.age_group.value if request.age_group else None,
                duration_value=request.duration.value if request.duration else None,
                duration_unit=request.duration.unit.value if request.duration else None,
            )

            # Execute inference
            result = self._predictor.predict(symptom_input)

            # Map predictions
            condition_preds: List[ConditionPrediction] = [
                ConditionPrediction(
                    condition=pred.condition,
                    confidence=pred.confidence,
                    description=pred.description,
                )
                for pred in result.predictions
            ]

            # Map rare condition matches (kept strictly separate from classifier predictions)
            rare_matches: List[RareConditionMatch] = []
            for r in getattr(result, "rare_condition_matches", []):
                sources = [
                    RareConditionSource(
                        authority=s.get("authority", ""),
                        title=s.get("title"),
                        identifier=s.get("identifier"),
                        url=s.get("url"),
                    )
                    for s in r.get("sources", [])
                ]
                rare_matches.append(
                    RareConditionMatch(
                        condition_id=r["condition_id"],
                        display_name=r["display_name"],
                        orpha_code=r["orpha_code"],
                        icd10_code=r["icd10_code"],
                        match_score=r["match_score"],
                        matched_symptoms=r.get("matched_symptoms", []),
                        sources=sources,
                        disclaimer=r.get("disclaimer", "Educational prototype matching score."),
                    )
                )

            # Combine telemetry metadata
            meta = {
                "received_free_text": bool(request.free_text_symptoms),
                "received_structured_count": len(request.structured_symptoms or []),
                "has_age_group": request.age_group is not None,
                "has_duration": request.duration is not None,
                "symptom_understanding_engine": symptom_understanding.engine_used,
                "clarification_needed": symptom_understanding.clarification_needed,
            }
            meta.update(result.metadata)

            return PredictResponse(
                status="success",
                model_version=result.model_version,
                predictions=condition_preds,
                rare_condition_matches=rare_matches,
                symptom_understanding=symptom_understanding,
                message="Prediction successfully generated from educational ML prototype.",
                disclaimer=(
                    "This prototype is strictly for educational purposes and does not provide medical diagnosis or advice. "
                    "Predicted condition scores are uncalibrated model outputs and do not represent clinical probability."
                ),
                metadata=meta,
            )

        except Exception as e:
            logger.exception("Inference error occurred during predict()")
            # Return graceful error response without leaking system internals
            return PredictResponse(
                status="error",
                model_version=getattr(self._predictor, "metadata", {}).get("pipeline_version"),
                predictions=[],
                rare_condition_matches=[],
                symptom_understanding=symptom_understanding,
                message="An internal error occurred during prediction processing.",
                disclaimer="This prototype is strictly for educational purposes and does not provide medical diagnosis or advice.",
                metadata={"error_type": type(e).__name__},
            )


# Singleton service instance for dependency injection across routes
prediction_service = PredictionService()
