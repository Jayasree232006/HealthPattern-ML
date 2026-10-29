from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app
from app.services.prediction_service import prediction_service

client = TestClient(app)


def test_root_health_check():
    """Verify legacy GET /health remains working and returns 200."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_v1_health_check():
    """Verify GET /api/v1/health returns version and model status."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "health-ai-assistant"
    assert data["version"] == "0.1.0"
    # When ML pipeline artifacts are present, model_status is "available"
    assert data["model_status"] == ("available" if prediction_service.is_available else "unavailable")


def test_cors_preflight():
    """Verify CORS headers for Vite frontend origins."""
    headers = {
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
    }
    response = client.options("/api/v1/predict", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_predict_success_response():
    """Verify POST /api/v1/predict returns successful predictions with live ML pipeline."""
    payload = {
        "free_text_symptoms": "Mild headache and runny nose",
        "structured_symptoms": ["headache", "runny_nose"],
        "age_group": "adult",
        "duration": {
            "value": 3,
            "unit": "days"
        }
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["model_version"] == "0.1.0"
    assert len(data["predictions"]) > 0
    # Predictions must have condition, confidence in [0, 1]
    for pred in data["predictions"]:
        assert "condition" in pred
        assert 0.0 <= pred["confidence"] <= 1.0

    # Rare condition matches must be present and distinct from common condition predictions
    assert "rare_condition_matches" in data
    assert len(data["rare_condition_matches"]) == 4
    for r in data["rare_condition_matches"]:
        assert "condition_id" in r
        assert "display_name" in r
        assert "orpha_code" in r
        assert 0.0 <= r["match_score"] <= 1.0
        assert len(r["sources"]) >= 1

    assert "educational" in data["disclaimer"].lower()
    assert data["metadata"]["received_free_text"] is True
    assert data["metadata"]["received_structured_count"] == 2
    assert data["metadata"]["has_age_group"] is True
    assert data["metadata"]["has_duration"] is True


def test_predict_with_only_free_text_symptoms():
    """Verify valid request with only free-text symptoms."""
    payload = {"free_text_symptoms": "Experiencing sore throat and chills"}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["predictions"]) > 0


def test_predict_with_only_structured_symptoms():
    """Verify valid request with only structured symptoms."""
    payload = {"structured_symptoms": ["cough", "chills"]}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["predictions"]) > 0


def test_predict_model_unavailable_fallback():
    """Verify POST /api/v1/predict returns model_unavailable when models are disabled/missing."""
    with patch.object(prediction_service, "_is_loaded", False):
        payload = {
            "free_text_symptoms": "Mild headache",
            "structured_symptoms": ["headache"]
        }
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "model_unavailable"
        assert data["predictions"] == []
        assert data["rare_condition_matches"] == []
        assert "educational" in data["disclaimer"].lower()
        assert data["metadata"]["received_free_text"] is True


def test_health_check_reflects_unavailable_model():
    """Verify GET /api/v1/health reports 'unavailable' when model is not loaded."""
    with patch.object(prediction_service, "_is_loaded", False):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["model_status"] == "unavailable"


def test_predict_rejects_empty_symptoms():
    """Verify 422 Unprocessable Entity when no symptoms are provided."""
    # Empty payload
    response = client.post("/api/v1/predict", json={})
    assert response.status_code == 422

    # Whitespace-only free text and empty list
    payload = {
        "free_text_symptoms": "   ",
        "structured_symptoms": ["  ", ""],
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_invalid_age_group():
    """Verify validation rejects invalid age_group value."""
    payload = {
        "free_text_symptoms": "Headache",
        "age_group": "super-senior"
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_invalid_duration():
    """Verify validation rejects non-positive duration values."""
    payload = {
        "free_text_symptoms": "Headache",
        "duration": {
            "value": -1,
            "unit": "days"
        }
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


def test_rare_condition_scores_separated_from_common_predictions():
    """Verify that common condition prediction scores and rare condition match scores are distinct lists."""
    payload = {
        "free_text_symptoms": "I have butterfly malar rash on my face, severe fatigue, and joint pain",
        "structured_symptoms": ["skin_rash", "fatigue", "joint_pain"]
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Predictions contain common conditions
    common_names = [p["condition"] for p in data["predictions"]]
    assert any("Allergy" in name or "Cold" in name or "Migraine" in name for name in common_names)

    # Rare condition matches contain rare profiles
    rare_names = [r["display_name"] for r in data["rare_condition_matches"]]
    assert any("Systemic Lupus Erythematosus" in name for name in rare_names)

    # Top rare condition should be SLE with high score
    sle_match = next(r for r in data["rare_condition_matches"] if r["condition_id"] == "systemic_lupus_erythematosus")
    assert sle_match["match_score"] > 0.5
    assert "fatigue" in sle_match["matched_symptoms"]
    assert "joint_pain" in sle_match["matched_symptoms"]
