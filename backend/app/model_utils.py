from pathlib import Path
import joblib
import shutil

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(__file__).resolve().parents[0] / 'models'
MODEL_PATH = MODEL_DIR / 'model.pkl'

def ensure_model_available():
    """Ensure model.pkl exists under backend/app/models. If not, attempt to copy from repository-level models/ folder."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        return MODEL_PATH
    alt = ROOT / 'models' / 'laptop_price_prediction_model.pkl'
    if alt.exists():
        shutil.copy2(alt, MODEL_PATH)
        return MODEL_PATH
    # last resort: look for any pkl in repo models
    for p in (ROOT / 'models').glob('*.pkl'):
        shutil.copy2(p, MODEL_PATH)
        return MODEL_PATH
    raise FileNotFoundError('Trained model not found. Place the trained pipeline at backend/app/models/model.pkl or models/*.pkl at repo root')

def load_model():
    path = ensure_model_available()
    return joblib.load(path)
