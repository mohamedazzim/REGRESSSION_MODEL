from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import pandas as pd
import traceback

from model_utils import load_model

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
        # Accept a single record (dict) or batch (list of dicts)
        payload = req.data
        if isinstance(payload, dict):
            df = pd.DataFrame([payload])
        else:
            df = pd.DataFrame(payload)

        # Ensure columns are sanitized
        df.columns = [str(c).strip() for c in df.columns]

        preds = MODEL.predict(df)
        # If single input, return single scalar
        if len(preds) == 1:
            return {"prediction": float(preds[0])}
        return {"prediction": [float(x) for x in preds]}
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(exc))
