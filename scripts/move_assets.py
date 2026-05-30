"""Utility to move dataset and model into monorepo structure.
Run locally to relocate data.csv -> data/ and model -> backend/app/models/
"""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
src_data = ROOT / 'data.csv'
dst_data_dir = ROOT / 'data'
dst_data = dst_data_dir / 'data.csv'

dst_model_src = ROOT / 'models' / 'laptop_price_prediction_model.pkl'
dst_model_dir = ROOT / 'backend' / 'app' / 'models'
dst_model = dst_model_dir / 'model.pkl'

dst_data_dir.mkdir(parents=True, exist_ok=True)
dst_model_dir.mkdir(parents=True, exist_ok=True)

if src_data.exists():
    shutil.move(str(src_data), str(dst_data))
    print(f'Moved dataset to {dst_data}')
else:
    print(f'No dataset found at {src_data}; please place dataset in repo root first')

if dst_model_src.exists():
    shutil.copy2(dst_model_src, dst_model)
    print(f'Copied model to {dst_model}')
else:
    print(f'No model found at {dst_model_src}; place trained model at repo models/ first')
