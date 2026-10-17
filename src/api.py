"""FastAPI service that serves the exported 'production' model."""
import json
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from src import config

THRESHOLD = 0.5
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the server starts (not on every request)."""
    state["model"] = joblib.load(config.MODEL_PATH)
    state["info"] = json.loads(config.MODEL_INFO_PATH.read_text())
    yield
    state.clear()


app = FastAPI(title="Predictive Maintenance API", version="1.0.0", lifespan=lifespan)
# Finer latency buckets: the default (0.1, 0.5, 1 s) is too coarse for a ~ms API
Instrumentator().instrument(
    app, latency_lowr_buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1)
).expose(app)


class MachineReading(BaseModel):
    """One machine sensor reading. Ranges reject physically impossible values."""

    type: Literal["L", "M", "H"] = Field(description="Product quality variant")
    air_temperature_k: float = Field(ge=250, le=350, description="Air temperature [K]")
    process_temperature_k: float = Field(ge=250, le=350, description="Process temperature [K]")
    rotational_speed_rpm: float = Field(gt=0, le=5000, description="Rotational speed [rpm]")
    torque_nm: float = Field(ge=0, le=150, description="Torque [Nm]")
    tool_wear_min: float = Field(ge=0, le=400, description="Tool wear [min]")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "type": "M",
                    "air_temperature_k": 298.1,
                    "process_temperature_k": 308.6,
                    "rotational_speed_rpm": 1551,
                    "torque_nm": 42.8,
                    "tool_wear_min": 0,
                }
            ]
        }
    }


class Prediction(BaseModel):
    failure_probability: float
    predicted_failure: bool
    threshold: float
    model_version: str

    model_config = {"protected_namespaces": ()}


@app.get("/health")
def health() -> dict:
    """Simple liveness check that also reports which model version is loaded."""
    info = state.get("info", {})
    return {"status": "ok", "model_loaded": "model" in state, "model_version": info.get("version")}


@app.post("/predict", response_model=Prediction)
def predict(reading: MachineReading) -> Prediction:
    model = state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    features = pd.DataFrame([reading.model_dump()])
    features = features[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]
    probability = float(model.predict_proba(features)[0, 1])

    return Prediction(
        failure_probability=round(probability, 4),
        predicted_failure=probability >= THRESHOLD,
        threshold=THRESHOLD,
        model_version=str(state["info"]["version"]),
    )