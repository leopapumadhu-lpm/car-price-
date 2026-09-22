"""
api/main.py
───────────
FastAPI backend for the Car Price Prediction application.

Endpoints
---------
GET  /health        — liveness probe
POST /predict       — predict selling price from car features
GET  /models        — model comparison metrics (from training)
GET  /features      — feature importance data (from training)
"""

import json
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import COMPARISON_PATH, FEATURE_IMPORTANCE_PATH, MODEL_PATH
from src.predict import make_prediction


# ── Model store (populated at startup) ────────────────────────────────────────
model_store: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model once when the server starts."""
    try:
        model_store["model"] = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    except FileNotFoundError:
        print(
            f"WARNING: {MODEL_PATH} not found. "
            "Run `python -m src.train` before starting the API."
        )
    yield
    model_store.clear()


# ── Pydantic request schema ────────────────────────────────────────────────────

class CarFeatures(BaseModel):
    """Raw car features sent by the client (matching training column names)."""

    fuel: str = Field(
        ...,
        description="One of: petrol, diesel, cng, lpg",
        examples=["petrol"],
    )
    seller_type: str = Field(
        ...,
        description="One of: individual, dealer, trustmark dealer",
        examples=["individual"],
    )
    transmission: str = Field(
        ...,
        description="One of: manual, automatic",
        examples=["manual"],
    )
    owner: str = Field(
        ...,
        description="One of: first owner, second owner, third owner, fourth & above owner, test drive car",
        examples=["first owner"],
    )
    car_age: int = Field(..., ge=0, le=30, description="Years since manufacture", examples=[5])
    km_driven: float = Field(..., ge=0, le=1_000_000, description="Total km driven", examples=[50000])
    mileage: float = Field(..., ge=0.0, le=50.0, description="Fuel efficiency (km/ltr/kg)", examples=[18.0])
    engine: float = Field(..., ge=500.0, le=6000.0, description="Engine displacement (CC)", examples=[1200.0])
    max_power: float = Field(..., ge=30.0, le=600.0, description="Max power (bhp)", examples=[82.0])
    seats: float = Field(..., ge=2.0, le=14.0, description="Number of seats", examples=[5.0])


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Car Price Prediction API",
    description="Predict used-car selling prices using a trained ML model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Utility"])
def health():
    """Liveness probe."""
    model_ready = "model" in model_store
    return {"status": "ok", "model_loaded": model_ready}


@app.post("/predict", tags=["Prediction"])
def predict(features: CarFeatures):
    """
    Predict the selling price of a used car.

    Accepts raw feature values (same units as the training data).
    Returns the predicted price in INR.
    """
    if "model" not in model_store:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python -m src.train` first.",
        )
    try:
        price = make_prediction(model_store["model"], features.model_dump())
        return {"predicted_price": round(price, 2), "currency": "INR"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/models", tags=["Metadata"])
def get_models():
    """Return model comparison metrics saved during training."""
    try:
        with open(COMPARISON_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Model comparison data not found. Run `python -m src.train` first.",
        )


@app.get("/features", tags=["Metadata"])
def get_features():
    """Return feature importance data saved during training."""
    try:
        with open(FEATURE_IMPORTANCE_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Feature importance data not found. Run `python -m src.train` first.",
        )
