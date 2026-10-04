"""Step 15 - Train the Isolation Forest on NORMAL process behaviour."""
import json
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
 
from src.config import (PROCESS_DIR, PROCESS_CSV, REPORTS_DIR, FEATURES, TARGET,
                        SEED, DRIFT_MODEL_PATH)
 
train = pd.read_csv(PROCESS_DIR / "train.csv")
test = pd.read_csv(PROCESS_DIR / "test.csv")
 
# "Normal" = readings from the OK products in the training data
normal = train[train[TARGET] == 0][FEATURES]
 
pipe = make_pipeline(
    StandardScaler(),
    IsolationForest(n_estimators=300, contamination=0.01, random_state=SEED),
)
pipe.fit(normal)
joblib.dump(pipe, DRIFT_MODEL_PATH)
print("Saved", DRIFT_MODEL_PATH)
 
# ---- how often does it raise an alert? ----
flag = pipe.predict(test[FEATURES]) == -1
print(f"\nTest set alert rate: {flag.mean():.1%}")
print(f"  among OK rows       : {flag[test[TARGET] == 0].mean():.1%}")
print(f"  among defective rows: {flag[test[TARGET] == 1].mean():.1%}")
 
# ---- sanity check with an obviously abnormal reading ----
normal_row = normal.median().to_frame().T
abnormal_row = pd.DataFrame([{
    "temperature": 98, "pressure": 7.4, "machine_speed": 1990, "vibration": 5.0,
    "humidity": 88, "material_thickness": 5.1, "cycle_time": 68, "tool_wear": 90}])
print("\nTypical reading  ->", "DRIFT" if pipe.predict(normal_row)[0] == -1 else "NORMAL")
print("Extreme reading  ->", "DRIFT" if pipe.predict(abnormal_row)[0] == -1 else "NORMAL")
 
# ---- summary for the dashboard overview page ----
full = pd.read_csv(PROCESS_CSV)
summary = {
    "total_products": int(len(full)),
    "defective_products": int(full[TARGET].sum()),
    "ok_products": int((full[TARGET] == 0).sum()),
    "drift_alerts": int((pipe.predict(full[FEATURES]) == -1).sum()),
}
(REPORTS_DIR / "overview_summary.json").write_text(json.dumps(summary, indent=2))
print("\nOverview summary:", summary)
