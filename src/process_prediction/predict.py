"""Reusable process-defect prediction (used by the API, SHAP, tests)."""
import json
from functools import lru_cache
import pandas as pd
from xgboost import XGBClassifier
 
from src.config import XGB_MODEL_PATH, XGB_THRESHOLD_PATH, FEATURES
 
 
@lru_cache(maxsize=1)
def load_model():
    model = XGBClassifier()
    model.load_model(XGB_MODEL_PATH)
    return model
 
 
@lru_cache(maxsize=1)
def load_threshold() -> float:
    return json.loads(XGB_THRESHOLD_PATH.read_text())["threshold"]
 
 
def predict_process(row: dict) -> dict:
    """row: {'temperature': 72, 'pressure': 5.2, ...} -> prediction dict."""
    X = pd.DataFrame([row])[FEATURES]
    prob = float(load_model().predict_proba(X)[0, 1])
    thr = load_threshold()
    if prob >= max(0.70, thr):
        risk = "HIGH"
    elif prob >= thr:
        risk = "MEDIUM"
    else:
        risk = "LOW"
    return {
        "label": "DEFECTIVE" if prob >= thr else "OK",
        "defect_probability": round(prob * 100, 2),
        "risk_level": risk,
        "threshold_used": round(thr, 3),
    }
