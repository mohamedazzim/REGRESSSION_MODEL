# Laptop Price Prediction — Monorepo

This repository was reorganized into a production-ready monorepo containing a frontend (static), a FastAPI backend, model artifacts, and notebooks for analysis.

Structure
```
/
  /frontend                 # Static website (HTML/CSS/JS)
  /backend/app              # FastAPI app and model loading logic
  /notebooks                # Jupyter notebooks moved here
  /data                     # Dataset files
  /models                   # Trained model artifacts (optional, ignored by default)
  build_laptop_price_notebook.py
  requirements.txt
  vercel.json
```

Quick start (local)

1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. Ensure the trained model is available under `backend/app/models/model.pkl`. If you have an existing pipeline at `models/laptop_price_prediction_model.pkl`, the backend will attempt to copy it into place automatically at startup.

3. Run the API locally

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

4. Open the frontend by serving the `frontend/` folder (or deploy to Vercel as described below).

API

POST /api/predict

Request JSON body:
```
{ "data": { "ram_gb": 8, "display_size": 15.6, "resolution_width": 1920, "resolution_height":1080, "ssd_size_gb":256 } }
```

Deployment (Vercel)

This repo contains a `vercel.json` that instructs Vercel to serve the static frontend and route `/api/predict` to the FastAPI backend. Deploy directly from GitHub — ensure the required Python packages are available in `requirements.txt`.
