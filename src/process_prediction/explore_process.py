"""Steps 8, 9 and 11 - Validate and explore the process dataset."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from src.config import PROCESS_CSV, REPORTS_DIR, FEATURES, TARGET
 
df = pd.read_csv(PROCESS_CSV)
 
print("Shape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nFirst rows:\n", df.head())
print("\nMissing values per column:\n", df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nClass distribution:\n", df[TARGET].value_counts())
print("Defect rate: {:.1%}".format(df[TARGET].mean()))
print("\nFeature ranges:\n", df[FEATURES].describe().T[["min", "mean", "max", "std"]])
 
# ---- outliers (IQR rule) ----
print("\nOutliers per feature (1.5 x IQR rule):")
for c in FEATURES:
    q1, q3 = df[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    n = ((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum()
    print(f"  {c:20s} {n:5d} ({n / len(df):.1%})")
 
# ---- feature vs defect (Step 11) ----
print("\nMean of each feature by defect class:")
print(df.groupby(TARGET)[FEATURES].mean().round(2).T)
 
print("\nCorrelation of each feature with defect:")
print(df[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(ascending=False).round(3))
 
# histograms
df[FEATURES].hist(bins=40, figsize=(14, 8))
plt.tight_layout(); plt.savefig(REPORTS_DIR / "process_histograms.png", dpi=120); plt.close()
 
# box plots: feature split by defect / ok
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, c in zip(axes.ravel(), FEATURES):
    df.boxplot(column=c, by=TARGET, ax=ax)
    ax.set_title(c); ax.set_xlabel("defect (0 = OK, 1 = defective)")
plt.suptitle("")
plt.tight_layout(); plt.savefig(REPORTS_DIR / "process_boxplots.png", dpi=120); plt.close()
 
# correlation heatmap
corr = df[FEATURES + [TARGET]].corr()
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
plt.colorbar(im); plt.tight_layout()
plt.savefig(REPORTS_DIR / "process_correlation.png", dpi=120); plt.close()
print("\nPlots saved to reports/")
