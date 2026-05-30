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

Vercel build note
------------------

During an attempted Vercel build the deployment failed because the Python dependency bundle exceeded Vercel's ephemeral storage limit (500 MB). To address this there are three options:

1. Use a scikit-learn-only model: Export or retrain a model that depends only on `scikit-learn` (or other lightweight libs). Place the pipeline at `backend/app/models/model.pkl` and the reduced `requirements.txt` will install and deploy successfully.

2. Host the heavy-model backend separately: Run the full-featured API on a service that allows larger bundles (Render, Railway, DigitalOcean, AWS) where `xgboost`, `catboost`, and `shap` can be installed. Keep the frontend on Vercel and point the `/api/predict` endpoint to that hosted API.

3. Use a container-based deployment: Deploy the app using Docker (Vercel's Docker or another container host) where you control the runtime image and have more space for dependencies.

I have removed heavy ML packages (`xgboost`, `catboost`, `shap`) from `requirements.txt` to allow Vercel builds to succeed with a smaller bundle. If you want, I can:

- Attempt to distill or retrain a scikit-learn compatible model from your dataset and export it to `backend/app/models/model.pkl`.
- Restore the heavy dependencies and change the deployment strategy to a container or external host.

