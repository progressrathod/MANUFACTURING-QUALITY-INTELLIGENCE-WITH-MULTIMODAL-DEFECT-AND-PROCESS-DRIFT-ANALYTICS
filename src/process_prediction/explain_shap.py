"""Step 14 - SHAP: global plots (run as a script) + a reusable explain() function."""
from functools import lru_cache
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from src.config import PROCESS_DIR, REPORTS_DIR, FEATURES
from src.process_prediction.predict import load_model
 
 
@lru_cache(maxsize=1)
def get_explainer():
    return shap.TreeExplainer(load_model())
 
 
def explain(row: dict, top_n: int = 3) -> list:
    """Return the top factors behind ONE prediction.
    A positive shap value pushes towards DEFECT, a negative one pushes towards OK."""
    X = pd.DataFrame([row])[FEATURES]
    sv = np.asarray(get_explainer().shap_values(X))[0]
    order = np.argsort(-np.abs(sv))[:top_n]
    return [{
        "feature": FEATURES[i],
        "value": float(X.iloc[0, i]),
        "shap_value": round(float(sv[i]), 4),
        "effect": "increases defect risk" if sv[i] > 0 else "reduces defect risk",
    } for i in order]
 
 
if __name__ == "__main__":
    test = pd.read_csv(PROCESS_DIR / "test.csv")
    X_test = test[FEATURES]
    explainer = get_explainer()
    shap_values = explainer.shap_values(X_test)
 
    # 1. Feature importance (mean |SHAP|)
    importance = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURES).sort_values(ascending=False)
    print("Feature importance (mean |SHAP|):\n", importance.round(4))
 
    # 2. Bar plot
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout(); plt.savefig(REPORTS_DIR / "shap_importance.png", dpi=150, bbox_inches="tight"); plt.close()
 
    # 3. Summary (beeswarm) plot
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout(); plt.savefig(REPORTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight"); plt.close()
 
    # 4. One individual prediction (waterfall) - the row with the highest predicted risk
    probs = load_model().predict_proba(X_test)[:, 1]
    i = int(np.argmax(probs))
    shap.plots.waterfall(explainer(X_test.iloc[[i]])[0], show=False)
    plt.tight_layout(); plt.savefig(REPORTS_DIR / "shap_single_prediction.png", dpi=150, bbox_inches="tight"); plt.close()
 
    print(f"\nExplanation for test row {i} (defect probability {probs[i]:.1%}):")
    for f in explain(X_test.iloc[i].to_dict()):
        print(f"  {f['feature']:20s} = {f['value']:<9} -> {f['effect']}  (SHAP {f['shap_value']:+})")
    print("\nPlots saved to reports/")
