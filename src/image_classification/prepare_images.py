"""Step 3 - Clean the images and split them 70 / 15 / 15 (stratified)."""
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split
 
from src.config import IMG_RAW, IMG_SPLIT, IMG_CLASSES, SEED
 
 
def is_valid(path):
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False
 
 
files, labels, skipped = [], [], 0
for cls in IMG_CLASSES:
    for p in sorted((IMG_RAW / cls).iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if is_valid(p):
            files.append(p)
            labels.append(cls)
        else:
            skipped += 1
print(f"Valid images: {len(files)}   Corrupted/skipped: {skipped}")
 
# 70% train, then split the remaining 30% in half -> 15% val, 15% test
tr_f, tmp_f, tr_l, tmp_l = train_test_split(
    files, labels, test_size=0.30, stratify=labels, random_state=SEED)
va_f, te_f, va_l, te_l = train_test_split(
    tmp_f, tmp_l, test_size=0.50, stratify=tmp_l, random_state=SEED)
 
if IMG_SPLIT.exists():
    shutil.rmtree(IMG_SPLIT)                      # makes the script safe to re-run
 
for split, fs, ls in [("train", tr_f, tr_l), ("val", va_f, va_l), ("test", te_f, te_l)]:
    for f, lab in zip(fs, ls):
        out = IMG_SPLIT / split / lab
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, out / f.name)
    print(f"{split:5s}: {len(fs)} images "
          f"({sum(l == IMG_CLASSES[0] for l in ls)} defective, "
          f"{sum(l == IMG_CLASSES[1] for l in ls)} ok)")
print("Saved to", IMG_SPLIT)
