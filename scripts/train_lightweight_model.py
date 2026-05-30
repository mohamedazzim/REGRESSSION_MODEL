"""Train a Vercel-friendly scikit-learn model and export it to backend/app/models/model.pkl.

This script avoids xgboost/catboost so the deployed artifact can be loaded in
Vercel's Python runtime without extra heavy dependencies.
"""

from __future__ import annotations

from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data.csv"
OUTPUT_PATH = ROOT / "backend" / "app" / "models" / "model.pkl"


def to_number(value) -> float:
    if pd.isna(value):
        return np.nan
    match = re.search(r"(\d+(?:\.\d+)?)", str(value).replace(",", ""))
    return float(match.group(1)) if match else np.nan


def storage_to_gb(value) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).upper().replace(" ", "")
    match = re.search(r"(\d+(?:\.\d+)?)(TB|GB)", text)
    if not match:
        return np.nan
    amount = float(match.group(1))
    return amount * 1024 if match.group(2) == "TB" else amount


def extract_cpu_brand(text) -> str:
    text = str(text)
    if re.search(r"\bIntel\b", text, re.I):
        return "Intel"
    if re.search(r"\bAMD\b", text, re.I):
        return "AMD"
    if re.search(r"\bApple\b", text, re.I):
        return "Apple"
    return "Other"


def extract_cpu_family(text) -> str:
    text = str(text)
    patterns = [r"Core\s+Ultra\s+\d+", r"Core\s+i[3-9]", r"Ryzen\s+[3-9]", r"M\d+", r"Celeron", r"Pentium"]
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return re.sub(r"\s+", " ", match.group(0)).strip()
    return "Other"


def extract_gpu_brand(text) -> str:
    text = str(text)
    if re.search(r"\bNVIDIA\b", text, re.I):
        return "NVIDIA"
    if re.search(r"\bAMD\b", text, re.I):
        return "AMD"
    if re.search(r"\bIntel\b", text, re.I):
        return "Intel"
    if re.search(r"\bApple\b", text, re.I):
        return "Apple"
    return "Other"


def extract_gpu_memory_gb(text) -> float:
    if pd.isna(text):
        return np.nan
    match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*GB", str(text), re.I)
    return float(match.group(1)) if match else np.nan


def extract_os_family(text) -> str:
    text = str(text)
    for family in ["Windows", "Mac", "Chrome", "Ubuntu", "DOS", "Android", "Linux"]:
        if re.search(rf"\b{family}\b", text, re.I):
            return family
    return "Other"


def extract_cpu_cores(text) -> float:
    match = re.search(r"(\d+)\s*Cores?", str(text), re.I)
    if match:
        return float(match.group(1))
    match = re.search(r"(\d+)\s*Core", str(text), re.I)
    if match:
        return float(match.group(1))
    return np.nan


def extract_cpu_threads(text) -> float:
    match = re.search(r"(\d+)\s*Threads?", str(text), re.I)
    return float(match.group(1)) if match else np.nan


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    for column in ["display_size", "resolution_width", "resolution_height", "spec_rating", "warranty"]:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")

    working["ram_gb"] = working["Ram"].map(to_number) if "Ram" in working.columns else np.nan
    working["storage_gb"] = working["ROM"].map(storage_to_gb) if "ROM" in working.columns else np.nan
    working["ssd_size_gb"] = np.where(working["ROM_type"].astype(str).str.contains("SSD", case=False, na=False), working["storage_gb"], 0.0)
    working["hdd_size_gb"] = np.where(working["ROM_type"].astype(str).str.contains("Hard", case=False, na=False), working["storage_gb"], 0.0)
    working["total_storage_gb"] = working["storage_gb"]
    working["cpu_brand"] = working["processor"].map(extract_cpu_brand) if "processor" in working.columns else "Other"
    working["cpu_family"] = working["processor"].map(extract_cpu_family) if "processor" in working.columns else "Other"
    working["cpu_cores"] = working["CPU"].map(extract_cpu_cores) if "CPU" in working.columns else np.nan
    working["cpu_threads"] = working["CPU"].map(extract_cpu_threads) if "CPU" in working.columns else np.nan
    working["gpu_brand"] = working["GPU"].map(extract_gpu_brand) if "GPU" in working.columns else "Other"
    working["gpu_memory_gb"] = working["GPU"].map(extract_gpu_memory_gb) if "GPU" in working.columns else np.nan
    working["os_family"] = working["OS"].map(extract_os_family) if "OS" in working.columns else "Other"
    working["ppi"] = np.sqrt(working["resolution_width"] ** 2 + working["resolution_height"] ** 2) / working["display_size"]
    working["is_gaming"] = working["name"].astype(str).str.contains(
        "gaming|rog|victus|nitro|legion|omen|predator|tuf|loq|katana|cyborg|alienware|g15|g16",
        case=False,
        na=False,
    ).astype(int)

    feature_columns = [
        "brand",
        "spec_rating",
        "warranty",
        "display_size",
        "resolution_width",
        "resolution_height",
        "ram_gb",
        "storage_gb",
        "ssd_size_gb",
        "hdd_size_gb",
        "total_storage_gb",
        "cpu_brand",
        "cpu_family",
        "cpu_cores",
        "cpu_threads",
        "gpu_brand",
        "gpu_memory_gb",
        "os_family",
        "ppi",
        "is_gaming",
    ]
    return working[feature_columns].copy()


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    y = pd.to_numeric(df["price"], errors="coerce")
    X = build_features(df)

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = RandomForestRegressor(
        n_estimators=350,
        random_state=42,
        n_jobs=-1,
        max_depth=None,
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, preds),
        "MSE": mean_squared_error(y_test, preds),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "R2": r2_score(y_test, preds),
    }
    print(metrics)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, OUTPUT_PATH)
    print(f"Saved model to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
