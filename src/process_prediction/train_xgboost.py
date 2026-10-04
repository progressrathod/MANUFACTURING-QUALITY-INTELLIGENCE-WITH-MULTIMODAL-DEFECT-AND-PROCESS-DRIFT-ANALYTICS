"""Steps 12 & 13 - Train and evaluate XGBoost."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             precision_recall_curve, ConfusionMatrixDisplay, RocCurveDisplay)
 
from src.config import (PROCESS_DIR, FEATURES, TARGET, SEED, REPORTS_DIR,
                        XGB_MODEL_PATH, XGB_THRESHOLD_PATH)
 
train = pd.read_csv(PROCESS_DIR / "train.csv")
val = pd.read_csv(PROCESS_DIR / "val.csv")
test = pd.read_csv(PROCESS_DIR / "test.csv")
X_train, y_train = train[FEATURES], train[TARGET]
X_val, y_val = val[FEATURES], val[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]
 
# defects are rare (~16%), so we tell XGBoost to give them more weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight = {scale_pos_weight:.2f}")
 
# Small, regularised trees work best here: the signal in this dataset is weak,
# so deeper trees only memorise noise.
model = XGBClassifier(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=2,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="auc",
    early_stopping_rounds=50,
    random_state=SEED,
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
print("Best iteration:", model.best_iteration)
 
# ---- choose the decision threshold on the VALIDATION set (best F1 for defects) ----
val_prob = model.predict_proba(X_val)[:, 1]
prec, rec, thr = precision_recall_curve(y_val, val_prob)
f1 = 2 * prec * rec / (prec + rec + 1e-9)
threshold = float(thr[np.argmax(f1[:-1])])
print(f"Chosen threshold: {threshold:.3f}  (default would be 0.5)")
 
# ---- evaluate on the TEST set ----
prob = model.predict_proba(X_test)[:, 1]
pred = (prob >= threshold).astype(int)
 
print("\n=== TEST RESULTS ===")
print(f"Accuracy : {accuracy_score(y_test, pred):.3f}")
print(f"Precision: {precision_score(y_test, pred):.3f}")
print(f"Recall   : {recall_score(y_test, pred):.3f}")
print(f"F1-score : {f1_score(y_test, pred):.3f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, prob):.3f}")
print("\n", classification_report(y_test, pred, target_names=["OK", "DEFECTIVE"], digits=3))
cm = confusion_matrix(y_test, pred)
print("Confusion matrix (rows = true, cols = predicted):\n", cm)
 
ConfusionMatrixDisplay(cm, display_labels=["OK", "DEFECTIVE"]).plot(cmap="Blues")
plt.title("XGBoost - test set")
plt.savefig(REPORTS_DIR / "xgb_confusion_matrix.png", dpi=150, bbox_inches="tight"); plt.close()
RocCurveDisplay.from_predictions(y_test, prob)
plt.savefig(REPORTS_DIR / "xgb_roc_curve.png", dpi=150, bbox_inches="tight"); plt.close()
 
# ---- save ----
model.save_model(XGB_MODEL_PATH)
XGB_THRESHOLD_PATH.write_text(json.dumps({"threshold": threshold}))
print("\nSaved:", XGB_MODEL_PATH, "and", XGB_THRESHOLD_PATH)
