"""Reusable drift check (used by the API, tests)."""
from functools import lru_cache
import joblib
import pandas as pd
 
from src.config import DRIFT_MODEL_PATH, FEATURES
 
 
@lru_cache(maxsize=1)
def load_model():
    return joblib.load(DRIFT_MODEL_PATH)
 
 
def detect_drift(row: dict) -> dict:
    X = pd.DataFrame([row])[FEATURES]
    pipe = load_model()
    is_drift = bool(pipe.predict(X)[0] == -1)
    score = float(pipe.decision_function(X)[0])      # negative = unusual
    return {
        "status": "DRIFT DETECTED" if is_drift else "NORMAL",
        "is_drift": is_drift,
        "anomaly_score": round(score, 4),
    }
