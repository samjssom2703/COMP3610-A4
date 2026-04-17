"""
Test suite for the NYC Taxi Tip Prediction API.

Run with:  pytest test_app.py -v
"""

import pytest
from fastapi.testclient import TestClient
from app import app


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with lifespan events (model loading)."""
    with TestClient(app) as c:
        yield c

# Reference payload with valid values for reuse across tests
VALID_TRIP = {
    "trip_distance": 3.5,
    "passenger_count": 1,
    "fare_amount": 15.0,
    "pickup_hour": 14,
    "pickup_day_of_week": 2,
    "trip_duration_minutes": 12.0,
    "RatecodeID": 1,
    "extra": 1.0,
    "mta_tax": 0.5,
    "tolls_amount": 0.0,
    "improvement_surcharge": 1.0,
    "congestion_surcharge": 2.5,
    "Airport_fee": 0.0,
}


# ---- Test 1: Successful single prediction ---------------------------------
def test_single_prediction(client):
    """Valid input should return 200 with tip_amount, prediction_id, model_version."""
    response = client.post("/predict", json=VALID_TRIP)
    assert response.status_code == 200
    data = response.json()
    assert "tip_amount" in data
    assert "prediction_id" in data
    assert "model_version" in data
    assert isinstance(data["tip_amount"], float)


# ---- Test 2: Successful batch prediction -----------------------------------
def test_batch_prediction(client):
    """Batch of valid trips should return 200 with the correct number of predictions."""
    response = client.post("/predict/batch", json={"trips": [VALID_TRIP, VALID_TRIP, VALID_TRIP]})
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3
    assert len(data["predictions"]) == 3
    assert "processing_time_ms" in data
    for pred in data["predictions"]:
        assert "tip_amount" in pred
        assert "prediction_id" in pred


# ---- Test 3: Invalid input – missing required fields -----------------------
def test_invalid_input_missing_fields(client):
    """Omitting required fields should return HTTP 422."""
    response = client.post("/predict", json={"trip_distance": 3.5})
    assert response.status_code == 422


# ---- Test 4: Invalid input – out-of-range values ---------------------------
def test_invalid_input_out_of_range(client):
    """pickup_hour=25 is out of [0, 23]; the API must reject it with 422."""
    bad_trip = VALID_TRIP.copy()
    bad_trip["pickup_hour"] = 25
    response = client.post("/predict", json=bad_trip)
    assert response.status_code == 422


# ---- Test 5: Health-check endpoint -----------------------------------------
def test_health_check(client):
    """GET /health should report healthy status and model loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "model_version" in data
    assert "uptime_seconds" in data


# ---- Test 6: Edge case – near-zero distance trip ---------------------------
def test_zero_distance_trip(client):
    """A very short trip (distance ~0) should still produce a valid prediction."""
    edge_trip = VALID_TRIP.copy()
    edge_trip["trip_distance"] = 0.01
    edge_trip["fare_amount"] = 2.5
    edge_trip["trip_duration_minutes"] = 1.0
    response = client.post("/predict", json=edge_trip)
    assert response.status_code == 200
    assert isinstance(response.json()["tip_amount"], float)


# ---- Test 7: Edge case – extreme fare values -------------------------------
def test_extreme_fare_values(client):
    """High fare / long trip should still return a valid prediction."""
    extreme_trip = VALID_TRIP.copy()
    extreme_trip["fare_amount"] = 500.0
    extreme_trip["trip_distance"] = 50.0
    extreme_trip["trip_duration_minutes"] = 120.0
    response = client.post("/predict", json=extreme_trip)
    assert response.status_code == 200
    assert isinstance(response.json()["tip_amount"], float)


# ---- Test 8: Model info endpoint -------------------------------------------
def test_model_info(client):
    """GET /model/info should return model metadata."""
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "taxi-tip-regressor"
    assert "feature_names" in data
    assert "training_metrics" in data


# ---- Test 9: Batch exceeds max size ----------------------------------------
def test_batch_exceeds_max(client):
    """Sending >100 trips in a batch should return 422."""
    trips = [VALID_TRIP] * 101
    response = client.post("/predict/batch", json={"trips": trips})
    assert response.status_code == 422


# ---- Test 10: Swagger docs accessible --------------------------------------
def test_swagger_docs(client):
    """The auto-generated Swagger UI at /docs should be reachable."""
    response = client.get("/docs")
    assert response.status_code == 200
