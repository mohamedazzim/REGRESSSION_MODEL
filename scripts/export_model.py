"""Utility to copy an existing trained model into backend/app/models/model.pkl
Usage: python scripts/export_model.py
"""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / 'models' / 'laptop_price_prediction_model.pkl'
dst_dir = ROOT / 'backend' / 'app' / 'models'
dst = dst_dir / 'model.pkl'

dst_dir.mkdir(parents=True, exist_ok=True)
if src.exists():
    shutil.copy2(src, dst)
    print(f'Copied {src} -> {dst}')
else:
    print(f'No source model found at {src}; please place trained model there first')
