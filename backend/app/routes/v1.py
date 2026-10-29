from fastapi import APIRouter, status
from app.schemas import HealthResponse, PredictRequest, PredictResponse
from app.services.prediction_service import prediction_service

router = APIRouter(prefix="/api/v1", tags=["v1"])


@router.get("/health", response_model=HealthResponse, summary="Versioned health check")
def api_v1_health() -> HealthResponse:
    """Return API v1 service, model, and Groq integration health status."""
    from app.services.groq_service import groq_service

    return HealthResponse(
        status="ok",
        service="health-ai-assistant",
        version="0.1.0",
        model_status="available" if prediction_service.is_available else "unavailable",
        groq_status="configured" if groq_service.is_configured else "disabled",
        groq_model=groq_service.model_name if groq_service.is_configured else None,
    )


@router.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Experimental symptom-pattern exploration endpoint",
)
def api_v1_predict(request: PredictRequest) -> PredictResponse:
    """Validate symptoms and return prediction status.

    When ML pipeline models are unavailable, returns a model_unavailable
    response rather than synthetic predictions.
    """
    return prediction_service.predict(request)
