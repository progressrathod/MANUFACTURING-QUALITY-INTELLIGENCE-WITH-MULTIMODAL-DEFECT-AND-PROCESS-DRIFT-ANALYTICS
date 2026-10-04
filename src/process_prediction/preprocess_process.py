"""Step 10 - Clean the process data and split it 70 / 15 / 15 (stratified)."""
import pandas as pd
from sklearn.model_selection import train_test_split
 
from src.config import PROCESS_CSV, PROCESS_DIR, FEATURES, TARGET, SEED
 
df = pd.read_csv(PROCESS_CSV)
print("Rows loaded:", len(df))
 
# 1. duplicates
df = df.drop_duplicates()
 
# 2. missing values (this dataset has none, but real data often does)
if df[FEATURES].isna().any().any():
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())
df = df.dropna(subset=[TARGET])
 
# 3. outliers: we KEEP them. XGBoost is not sensitive to them, and an extreme
#    reading (e.g. very high vibration) may be exactly what causes a defect.
 
# 4. feature selection: all 8 columns are real process variables, so keep them all.
df = df[FEATURES + [TARGET]]
 
train, temp = train_test_split(df, test_size=0.30, stratify=df[TARGET], random_state=SEED)
val, test = train_test_split(temp, test_size=0.50, stratify=temp[TARGET], random_state=SEED)
 
train.to_csv(PROCESS_DIR / "train.csv", index=False)
val.to_csv(PROCESS_DIR / "val.csv", index=False)
test.to_csv(PROCESS_DIR / "test.csv", index=False)
 
for name, part in [("train", train), ("val", val), ("test", test)]:
    print(f"{name:5s}: {len(part):5d} rows | defect rate {part[TARGET].mean():.1%}")
