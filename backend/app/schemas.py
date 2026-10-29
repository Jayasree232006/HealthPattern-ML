from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, model_validator


class AgeGroup(str, Enum):
    INFANT = "infant"        # 0-1
    CHILD = "child"          # 2-12
    ADOLESCENT = "adolescent"# 13-17
    ADULT = "adult"          # 18-64
    SENIOR = "senior"        # 65+


class DurationUnit(str, Enum):
    HOURS = "hours"
    DAYS = "days"
    WEEKS = "weeks"
    MONTHS = "months"
    YEARS = "years"


class Duration(BaseModel):
    value: int = Field(..., gt=0, description="Duration value, must be a positive integer.")
    unit: DurationUnit = Field(..., description="Time unit for duration (hours, days, weeks, months, years).")


class PredictRequest(BaseModel):
    free_text_symptoms: Optional[str] = Field(
        default=None,
        description="Free-text narrative describing the symptoms.",
    )
    structured_symptoms: Optional[list[str]] = Field(
        default=None,
        description="Optional list of structured symptom tags or keywords.",
    )
    age_group: Optional[AgeGroup] = Field(
        default=None,
        description="Categorical age group of the patient.",
    )
    duration: Optional[Duration] = Field(
        default=None,
        description="Duration of the experienced symptoms.",
    )

    @model_validator(mode="after")
    def validate_has_symptoms(self) -> "PredictRequest":
        free_text = (self.free_text_symptoms or "").strip()
        cleaned_structured = [
            s.strip() for s in (self.structured_symptoms or []) if s.strip()
        ]

        if not free_text and not cleaned_structured:
            raise ValueError(
                "At least one symptom must be provided in either 'free_text_symptoms' or 'structured_symptoms'."
            )

        # Normalize structured_symptoms to filtered list if provided
        if self.structured_symptoms is not None:
            self.structured_symptoms = cleaned_structured

        if self.free_text_symptoms is not None:
            self.free_text_symptoms = free_text or None

        return self


class SymptomAssertion(str, Enum):
    AFFIRMED = "affirmed"
    NEGATED = "negated"
    UNCERTAIN = "uncertain"
    HISTORICAL = "historical"


class UnderstoodSymptom(BaseModel):
    symptom_key: str = Field(..., description="Canonical symptom key matched against approved vocabulary.")
    display_name: str = Field(..., description="Human-readable symptom name.")
    assertion: SymptomAssertion = Field(
        default=SymptomAssertion.AFFIRMED,
        description="Assertion status: affirmed, negated, uncertain, or historical."
    )
    temporal_context: Optional[str] = Field(
        default=None,
        description="Extracted temporal context (e.g., 'past 2 days', 'onset yesterday', '3 weeks ago')."
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for this symptom extraction."
    )
    raw_mention: Optional[str] = Field(
        default=None,
        description="Original text mention matching this symptom."
    )


class SpellingCorrection(BaseModel):
    original: str = Field(..., description="Original raw term from user input.")
    suggested: str = Field(..., description="Suggested correction or canonical symptom.")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence of the suggested correction.")
    clarification_recommended: bool = Field(
        default=False,
        description="Whether clarification should be requested rather than assuming correction."
    )


class SymptomUnderstandingResult(BaseModel):
    status: str = Field(
        default="success",
        description="Status: 'success', 'disabled', 'fallback_nlp', or 'error'."
    )
    symptoms: list[UnderstoodSymptom] = Field(
        default_factory=list,
        description="List of recognized symptoms with assertions and temporal context."
    )
    spelling_corrections: list[SpellingCorrection] = Field(
        default_factory=list,
        description="Suggested spelling corrections."
    )
    ambiguities: list[str] = Field(
        default_factory=list,
        description="Flagged ambiguous statements or symptom descriptions."
    )
    clarification_needed: bool = Field(
        default=False,
        description="Flag indicating user clarification should be requested."
    )
    clarification_prompt: Optional[str] = Field(
        default=None,
        description="Helpful question prompting user for symptom clarification."
    )
    engine_used: str = Field(
        default="rule_based",
        description="Engine used: 'groq' or 'rule_based'."
    )
    notes: Optional[str] = Field(
        default=None,
        description="Educational note regarding symptom understanding."
    )


class ConditionPrediction(BaseModel):
    condition: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    description: Optional[str] = None


class RareConditionSource(BaseModel):
    authority: str
    title: Optional[str] = None
    identifier: Optional[str] = None
    url: Optional[str] = None


class RareConditionMatch(BaseModel):
    condition_id: str
    display_name: str
    orpha_code: str
    icd10_code: str
    match_score: float = Field(..., ge=0.0, le=1.0, description="Heuristic symptom/hallmark match score; not a diagnosis probability.")
    matched_symptoms: list[str] = Field(default_factory=list)
    sources: list[RareConditionSource] = Field(default_factory=list)
    disclaimer: str = Field(
        default="Educational prototype matching score based on clinical literature. This is not a diagnostic tool and does not provide clinical probability."
    )


class PredictResponse(BaseModel):
    status: str = Field(
        ...,
        description="Status code/flag such as 'model_unavailable' or 'success'.",
    )
    model_version: Optional[str] = Field(
        default=None,
        description="Identifier of the ML model version when available.",
    )
    predictions: list[ConditionPrediction] = Field(
        default_factory=list,
        description="Predicted health conditions ranked by confidence.",
    )
    rare_condition_matches: list[RareConditionMatch] = Field(
        default_factory=list,
        description="Separate source-attributed rare-condition profile matches.",
    )
    symptom_understanding: Optional[SymptomUnderstandingResult] = Field(
        default=None,
        description="Structured Groq AI or rule-based symptom extraction and understanding result.",
    )
    message: str = Field(
        ...,
        description="Informational message or reason if unavailable.",
    )
    disclaimer: str = Field(
        default="This prototype is strictly for educational purposes and does not provide medical diagnosis or advice.",
        description="Educational and medical disclaimer.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Diagnostic or processing metadata.",
    )


class HealthResponse(BaseModel):
    status: str
    service: str = "health-ai-assistant"
    version: str = "0.1.0"
    model_status: str = "unavailable"
    groq_status: str = "disabled"
    groq_model: Optional[str] = None
