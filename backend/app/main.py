from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import re
import numpy as np
import pandas as pd
import traceback

from .model_utils import load_model

app = FastAPI(title="Laptop Price Prediction API")

# CORS: allow same-origin and standard dev origins. In production restrict origins explicitly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    data: dict


def _to_number(value, default=np.nan):
    if value is None:
        return default
    match = re.search(r"(\d+(?:\.\d+)?)", str(value).replace(",", ""))
    return float(match.group(1)) if match else default


def _build_model_row(payload: dict) -> pd.DataFrame:
    """Map the frontend payload into the feature set expected by the trained pipeline."""
    display_size = float(payload.get("display_size", 15.6))
    resolution_width = float(payload.get("resolution_width", 1920))
    resolution_height = float(payload.get("resolution_height", 1080))
    ppi = float(np.sqrt(resolution_width**2 + resolution_height**2) / display_size) if display_size else 0.0
    ssd_size_gb = float(payload.get("ssd_size_gb", 256))
    hdd_size_gb = float(payload.get("hdd_size_gb", 0))

    return pd.DataFrame(
        [
            {
                "brand": payload.get("brand", "Unknown"),
                "spec_rating": float(payload.get("spec_rating", 70)),
                "warranty": float(payload.get("warranty", 1)),
                "display_size": display_size,
                "resolution_width": resolution_width,
                "resolution_height": resolution_height,
                "ram_gb": float(payload.get("ram_gb", 8)),
                "storage_gb": float(payload.get("storage_gb", ssd_size_gb + hdd_size_gb)),
                "ssd_size_gb": ssd_size_gb,
                "hdd_size_gb": hdd_size_gb,
                "total_storage_gb": float(payload.get("total_storage_gb", ssd_size_gb + hdd_size_gb)),
                "cpu_brand": payload.get("cpu_brand", "Other"),
                "cpu_family": payload.get("cpu_family", "Other"),
                "cpu_cores": float(payload.get("cpu_cores", 4)),
                "cpu_threads": float(payload.get("cpu_threads", 8)),
                "gpu_brand": payload.get("gpu_brand", "Other"),
                "gpu_memory_gb": float(payload.get("gpu_memory_gb", 2)),
                "os_family": payload.get("os_family", "Windows"),
                "ppi": ppi,
                "is_gaming": int(payload.get("is_gaming", 0)),
            }
        ]
    )


@app.on_event("startup")
def startup_event():
    try:
        global MODEL
        MODEL = load_model()
    except Exception as exc:
        # Keep server up but report on predictions
        MODEL = None
        print("Model load failed:", exc)


@app.post("/api/predict")
def predict(req: PredictRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not available on server")
    try:
        payload = req.data if isinstance(req.data, dict) else req.data[0]
        df = _build_model_row(payload)

        preds = MODEL.predict(df)
        # If single input, return single scalar
        if len(preds) == 1:
            return {"prediction": float(preds[0])}
        return {"prediction": [float(x) for x in preds]}
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(exc))
