"""
FastAPI application for NYC Taxi Tip Prediction Service.

Serves tip_amount predictions from a trained Random Forest model
using the NYC Yellow Taxi Trip Records dataset.
"""

import uuid
import os
import time
import logging
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List


# ---------------------------------------------------------------------------
# Configuration (environment variables with sensible defaults)
# ---------------------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "models/rf_regressor.joblib")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")
MODEL_NAME = "taxi-tip-regressor"

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tip-prediction-api")

# Global references – populated at startup
ml_model = None
start_time = None
model_metadata = None

# Feature order must match training
MODEL_FEATURES = [
    "passenger_count", "trip_distance", "RatecodeID",
    "fare_amount", "extra", "mta_tax", "tolls_amount",
    "improvement_surcharge", "congestion_surcharge", "Airport_fee",
    "pickup_hour", "pickup_day_of_week",
    "trip_duration_minutes", "trip_speed_mph",
    "log_trip_distance", "fare_per_mile", "fare_per_minute",
    "is_weekend",
]


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class TripInput(BaseModel):
    """Input schema for a single taxi trip prediction request."""
    trip_distance: float = Field(
        ..., gt=0, description="Trip distance in miles"
    )
    passenger_count: int = Field(
        ..., ge=1, le=9, description="Number of passengers (1-9)"
    )
    fare_amount: float = Field(
        ..., ge=0, description="Fare amount in USD"
    )
    pickup_hour: int = Field(
        ..., ge=0, le=23, description="Hour of pickup (0-23)"
    )
    pickup_day_of_week: int = Field(
        ..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)"
    )
    trip_duration_minutes: float = Field(
        ..., ge=0, description="Trip duration in minutes"
    )
    RatecodeID: int = Field(
        default=1, ge=1, le=6, description="Rate code ID (1-6)"
    )
    extra: float = Field(
        default=0.0, ge=0, description="Extra charges in USD"
    )
    mta_tax: float = Field(
        default=0.5, ge=0, description="MTA tax in USD"
    )
    tolls_amount: float = Field(
        default=0.0, ge=0, description="Tolls amount in USD"
    )
    improvement_surcharge: float = Field(
        default=1.0, ge=0, description="Improvement surcharge in USD"
    )
    congestion_surcharge: float = Field(
        default=2.5, ge=0, description="Congestion surcharge in USD"
    )
    Airport_fee: float = Field(
        default=0.0, ge=0, description="Airport fee in USD"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response schema for a single tip prediction."""
    prediction_id: str
    tip_amount: float
    model_version: str


class BatchInput(BaseModel):
    """Input schema for batch predictions (max 100 records)."""
    trips: List[TripInput] = Field(..., min_length=1, max_length=100)


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    predictions: List[PredictionResponse]
    count: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Response schema for the health-check endpoint."""
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    """Response schema for the model-info endpoint."""
    model_name: str
    model_version: str
    feature_names: List[str]
    training_metrics: dict


# ---------------------------------------------------------------------------
# Feature engineering helper
# ---------------------------------------------------------------------------
def compute_features(trip: TripInput) -> dict:
    """Derive model features from raw trip inputs."""
    # Speed (capped at 80 mph)
    if trip.trip_duration_minutes > 0:
        trip_speed_mph = min(
            trip.trip_distance / (trip.trip_duration_minutes / 60.0), 80.0
        )
    else:
        trip_speed_mph = 0.0

    log_trip_distance = float(np.log1p(trip.trip_distance))
    fare_per_mile = (
        trip.fare_amount / trip.trip_distance if trip.trip_distance > 0 else 0.0
    )
    fare_per_minute = (
        trip.fare_amount / trip.trip_duration_minutes
        if trip.trip_duration_minutes > 0
        else 0.0
    )
    is_weekend = 1 if trip.pickup_day_of_week >= 5 else 0

    return {
        "passenger_count": trip.passenger_count,
        "trip_distance": trip.trip_distance,
        "RatecodeID": trip.RatecodeID,
        "fare_amount": trip.fare_amount,
        "extra": trip.extra,
        "mta_tax": trip.mta_tax,
        "tolls_amount": trip.tolls_amount,
        "improvement_surcharge": trip.improvement_surcharge,
        "congestion_surcharge": trip.congestion_surcharge,
        "Airport_fee": trip.Airport_fee,
        "pickup_hour": trip.pickup_hour,
        "pickup_day_of_week": trip.pickup_day_of_week,
        "trip_duration_minutes": trip.trip_duration_minutes,
        "trip_speed_mph": trip_speed_mph,
        "log_trip_distance": log_trip_distance,
        "fare_per_mile": fare_per_mile,
        "fare_per_minute": fare_per_minute,
        "is_weekend": is_weekend,
    }


# ---------------------------------------------------------------------------
# Lifespan – load model once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model on startup; clean up on shutdown."""
    global ml_model, start_time, model_metadata

    # STARTUP
    start_time = time.time()
    try:
        ml_model = joblib.load(MODEL_PATH)
        model_metadata = {
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "feature_names": MODEL_FEATURES,
            "training_metrics": {
                "MAE": 1.18,
                "RMSE": 2.28,
                "R2": 0.64,
            },
        }
        logger.info("Model loaded successfully from %s", MODEL_PATH)
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        raise

    yield  # application runs here

    # SHUTDOWN
    logger.info("Shutting down API …")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="NYC Taxi Tip Prediction API",
    description=(
        "Predicts tip amounts for NYC Yellow Taxi trips using a trained "
        "Random Forest model.  Built for COMP 3610 Assignment 4."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Global exception handler – never expose internal details
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unexpected errors and return a structured 500 response."""
    logger.error("Unhandled error on %s %s: %s", request.method, request.url, exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please try again.",
        },
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
async def predict(trip: TripInput):
    """Return a predicted tip amount for a single taxi trip."""
    features = compute_features(trip)
    df = pd.DataFrame([features])[MODEL_FEATURES]
    prediction = ml_model.predict(df)[0]
    return PredictionResponse(
        prediction_id=str(uuid.uuid4()),
        tip_amount=round(float(prediction), 2),
        model_version=MODEL_VERSION,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: BatchInput):
    """Return predicted tip amounts for a batch of trips (max 100)."""
    start = time.time()
    rows = [compute_features(trip) for trip in batch.trips]
    df = pd.DataFrame(rows)[MODEL_FEATURES]
    preds = ml_model.predict(df)
    results = [
        PredictionResponse(
            prediction_id=str(uuid.uuid4()),
            tip_amount=round(float(p), 2),
            model_version=MODEL_VERSION,
        )
        for p in preds
    ]
    elapsed = (time.time() - start) * 1000
    return BatchPredictionResponse(
        predictions=results,
        count=len(results),
        processing_time_ms=round(elapsed, 2),
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Return API health status, model-loaded flag, and uptime."""
    return HealthResponse(
        status="healthy",
        model_loaded=ml_model is not None,
        model_version=MODEL_VERSION,
        uptime_seconds=round(time.time() - start_time, 1),
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Return metadata about the currently loaded model."""
    return ModelInfoResponse(**model_metadata)
