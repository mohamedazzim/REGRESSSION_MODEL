from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[2]
MODEL_CANDIDATES = [
    Path(__file__).resolve().parent / "models" / "model.pkl",
    ROOT / "models" / "laptop_price_prediction_model.pkl",
    ROOT / "models" / "model.pkl",
]


def resolve_model_path() -> Path:
    """Locate the deployed model without writing to the read-only Vercel filesystem."""
    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            return candidate
    available = ", ".join(str(path) for path in MODEL_CANDIDATES)
    raise FileNotFoundError(f"Trained model not found. Looked in: {available}")


def load_model():
    path = resolve_model_path()
    return joblib.load(path)
