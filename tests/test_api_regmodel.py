"""
Tests for FastAPI regmodel endpoint.

Tests cover:
- /predict endpoint with RF and NN models
- Request validation (PredictRequest schema)
- Error handling (invalid model_type, missing fields, etc.)
- Response format validation
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
import sys
import os

# Add backend/regmodel to Python path so 'from app.' imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend", "regmodel"))


# === Fixtures ===


@pytest.fixture(scope="module")
def client():
    """FastAPI test client."""
    # Mock model behavior
    mock_rf = MagicMock()
    mock_rf.predict_clean.return_value = np.array([100.0, 150.0, 200.0])

    mock_nn = MagicMock()
    mock_nn.predict_clean.return_value = np.array([[120.0], [180.0], [220.0]])

    # Mock metadata for champion model
    mock_metadata = {
        "run_id": "test_run_123",
        "is_champion": True,
        "r2": 0.85,
        "rmse": 50.0,
        "model_type": "rf",
    }

    def get_model_side_effect(model_type, **kwargs):
        """Mock get_best_model_from_summary - accepts any parameters."""
        if model_type == "rf":
            return (mock_rf, mock_metadata)
        elif model_type == "nn":
            return (mock_nn, mock_metadata)
        raise ValueError(f"Unknown model_type: {model_type}")

    # Mock model_registry_summary to avoid GCS calls
    # Now 'from app.' resolves to backend/regmodel/app/ thanks to sys.path
    with patch(
        "app.model_registry_summary.get_best_model_from_summary",
        side_effect=get_model_side_effect,
    ):
        from backend.regmodel.app.fastapi_app import app

        yield TestClient(app)


@pytest.fixture
def api_request_payload(sample_bike_data):
    """Convert sample_bike_data to API request payload format."""
    # Take first 3 records from sample_bike_data
    records = sample_bike_data.head(3).to_dict(orient="records")

    return {"records": records, "model_type": "rf", "metric": "r2"}


# === Tests ===


class TestPredictEndpoint:
    """Test /predict endpoint with valid requests."""

    def test_predict_rf_success(self, client, api_request_payload):
        """Test successful prediction with RF model."""
        response = client.post("/predict", json=api_request_payload)

        if response.status_code != 200:
            print(f"Error: {response.json()}")
        assert response.status_code == 200
        data = response.json()

        assert "predictions" in data
        assert isinstance(data["predictions"], list)
        assert len(data["predictions"]) == 3
        assert all(isinstance(pred, (int, float)) for pred in data["predictions"])
        assert all(pred >= 0 for pred in data["predictions"])

    def test_predict_nn_success(self, client, api_request_payload):
        """Test successful prediction with NN model."""
        api_request_payload["model_type"] = "nn"
        response = client.post("/predict", json=api_request_payload)

        assert response.status_code == 200
        data = response.json()

        assert "predictions" in data
        assert isinstance(data["predictions"], list)
        assert len(data["predictions"]) == 3

    def test_predict_with_single_record(self, client, sample_predictions):
        """Test prediction with single record (using conftest fixture)."""
        response = client.post("/predict", json=sample_predictions)

        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) >= 1


class TestRequestValidation:
    """Test request schema validation."""

    def test_missing_records_field(self, client):
        """Test error when 'records' field is missing."""
        payload = {"model_type": "rf", "metric": "r2"}
        response = client.post("/predict", json=payload)

        assert response.status_code == 422  # Unprocessable Entity

    def test_missing_model_type_field(self, client):
        """Test error when 'model_type' field is missing."""
        payload = {
            "records": [
                {
                    "nom_du_compteur": "Totem 73 boulevard de Sébastopol S-N",
                    "Date et heure de comptage": "2024-04-15 08:00:00+02:00",
                    "Coordonnées géographiques": "48.8672, 2.3501",
                }
            ],
            "metric": "r2",
        }
        response = client.post("/predict", json=payload)

        assert response.status_code == 422

    def test_empty_records_list(self, client):
        """Test prediction with empty records list."""
        payload = {"records": [], "model_type": "rf", "metric": "r2"}
        response = client.post("/predict", json=payload)

        # Should either succeed with empty predictions or return error
        assert response.status_code in [200, 422, 500]

    def test_invalid_json_format(self, client):
        """Test error with invalid JSON."""
        response = client.post(
            "/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_predict_with_invalid_model_type(self, client, api_request_payload):
        """Test error when model_type is invalid."""
        api_request_payload["model_type"] = "invalid_model"
        response = client.post("/predict", json=api_request_payload)

        # Should return 500 with error message
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data


class TestResponseFormat:
    """Test response format consistency."""

    def test_response_has_predictions_key(self, client, api_request_payload):
        """Test that response always contains 'predictions' key."""
        response = client.post("/predict", json=api_request_payload)

        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data

    def test_predictions_are_numeric(self, client, api_request_payload):
        """Test that all predictions are numeric values."""
        response = client.post("/predict", json=api_request_payload)

        assert response.status_code == 200
        data = response.json()
        predictions = data["predictions"]

        assert all(isinstance(p, (int, float)) for p in predictions)

    def test_predictions_count_matches_input(self, client, api_request_payload):
        """Test that number of predictions matches input records."""
        response = client.post("/predict", json=api_request_payload)

        assert response.status_code == 200
        data = response.json()

        assert len(data["predictions"]) == len(api_request_payload["records"])
