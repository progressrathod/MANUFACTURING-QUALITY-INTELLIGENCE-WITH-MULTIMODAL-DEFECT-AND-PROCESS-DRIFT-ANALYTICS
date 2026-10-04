
"""Step 28 - Random production generator.
Builds each batch from your real production dataset (the 10,000-row CSV): every batch gets
50 products. Between MIN_DEFECTIVE_PER_BATCH and MAX_DEFECTIVE_PER_BATCH (see src/config.py)
of them are defective, the rest are OK, in random order. Each product gets an image of the
matching class: def_front for a defective product, ok_front for an OK product.
The plan is saved in the database, so a batch that is stopped and resumed continues with
the SAME products.
 
NOTE: in your data an image is not tied to a CSV row. The pairing is a DEMO only; in a real
plant the camera and the sensors report on the same physical product (barcode).
"""
import random
import time
from functools import lru_cache
 
import pandas as pd
from PIL import Image
 
from src.config import (PROCESS_CSV, IMG_RAW, IMG_SPLIT, FEATURES, TARGET,
                        MIN_DEFECTIVE_PER_BATCH, MAX_DEFECTIVE_PER_BATCH)
from src.database import get_conn
from src.batch_manager import start_batch, inspect_product
from src.image_classification.predict import predict_image
 
MIN_DEFECTIVE, MAX_DEFECTIVE = MIN_DEFECTIVE_PER_BATCH, MAX_DEFECTIVE_PER_BATCH
IMG_EXT = {".jpg", ".jpeg", ".png"}
FOLDER = {1: "def_front", 0: "ok_front"}        # CSV truth (1 = defective) -> image folder
PLAN_SCHEMA = """
CREATE TABLE IF NOT EXISTS batch_plan (         -- which CSV row and image each position uses
    batch_id   TEXT NOT NULL REFERENCES batches(batch_id),
    position   INTEGER NOT NULL,
    csv_row    INTEGER NOT NULL,
    truth      INTEGER NOT NULL,                -- 1 = defective row in the CSV
    image_file TEXT,
    PRIMARY KEY (batch_id, position)
);
"""
 
 
@lru_cache(maxsize=1)
def _data():
    return pd.read_csv(PROCESS_CSV)
 
 
def _search_dirs(folder):
    """Where images of one class may live: the original folder first, then the train/val/test
    copies made by prepare_images.py (used only if the original folder is missing or empty)."""
    return [[IMG_RAW / folder], [IMG_SPLIT / s / folder for s in ("train", "val", "test")]]
 
 
def _images(folder):
    """Names of all images of one class (def_front or ok_front)."""
    for group in _search_dirs(folder):
        names = sorted(p.name for d in group if d.exists() for p in d.iterdir()
                       if p.suffix.lower() in IMG_EXT)
        if names:
            return names
    return []
 
 
def _find_image(folder, name):
    """Full path of one image, or None."""
    for group in _search_dirs(folder):
        for d in group:
            if (d / name).exists():
                return d / name
    return None
 
 
def require_images():
    """Stop with a CLEAR message when the image folders cannot be found.
    (Before, missing images were skipped silently, so no product could ever be DEFECTIVE.)"""
    missing = [f for f in FOLDER.values() if not _images(f)]
    if missing:
        raise RuntimeError(
            f"No images found for: {', '.join(missing)}. Looked in {IMG_RAW} and in "
            f"{IMG_SPLIT}. Put the images in data/images/def_front and data/images/ok_front "
            f"(python -m src.setup_data does this), then try again.")
 
 
def _status(batch_id):
    with get_conn() as c:
        row = c.execute("SELECT status FROM batches WHERE batch_id=?", (batch_id,)).fetchone()
    if row is None:
        raise KeyError(f"Unknown batch {batch_id}")
    return row["status"]
 
 
def plan_batch(batch_id):
    """Pick the random products for a batch and save the plan. Returns the defective count."""
    require_images()
    df = _data()
    imgs = {truth: _images(folder) for truth, folder in FOLDER.items()}
    with get_conn() as c:
        row = c.execute("SELECT planned_size FROM batches WHERE batch_id=?", (batch_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown batch {batch_id}")
        size = row["planned_size"]
        c.execute(PLAN_SCHEMA)
        c.execute("DELETE FROM batch_plan WHERE batch_id=?", (batch_id,))
        n_bad = random.randint(MIN_DEFECTIVE, min(MAX_DEFECTIVE, size))
        chosen = (random.sample(df.index[df[TARGET] == 1].tolist(), n_bad)
                  + random.sample(df.index[df[TARGET] == 0].tolist(), size - n_bad))
        random.shuffle(chosen)                    # defects can appear anywhere in the batch
        for position, idx in enumerate(chosen, start=1):
            truth = int(df.at[idx, TARGET])
            image = random.choice(imgs[truth])    # def_front for defective, ok_front for OK
            c.execute("INSERT INTO batch_plan(batch_id, position, csv_row, truth, image_file) "
                      "VALUES (?,?,?,?,?)", (batch_id, position, int(idx), truth, image))
    return n_bad
 
 
def run_production(batch_id, delay=0.0):
    """Produce and inspect the remaining products of a batch.
    Stops early if an alert pauses the batch. Call it again after the manager resumes."""
    df = _data()
    with get_conn() as c:
        c.execute(PLAN_SCHEMA)
        done = c.execute("SELECT COUNT(*) FROM products WHERE batch_id=?", (batch_id,)).fetchone()[0]
        has_plan = c.execute("SELECT COUNT(*) FROM batch_plan WHERE batch_id=?",
                             (batch_id,)).fetchone()[0]
    if not has_plan:
        plan_batch(batch_id)
    with get_conn() as c:
        pending = c.execute("SELECT position, csv_row, truth, image_file FROM batch_plan "
                            "WHERE batch_id=? AND position>? ORDER BY position",
                            (batch_id, done)).fetchall()
    produced = 0
    for item in pending:
        if _status(batch_id) != "RUNNING":        # paused by an alert or closed
            break
        r = df.loc[item["csv_row"]]
        path = _find_image(FOLDER[item["truth"]], item["image_file"])
        if path is None:                          # never skip silently any more
            raise RuntimeError(f"Image {item['image_file']} ({FOLDER[item['truth']]}) was not found. "
                               f"Check the folder data/images/{FOLDER[item['truth']]}.")
        with Image.open(path) as im:
            res = predict_image(im)
        inspect_product(batch_id, {k: float(r[k]) for k in FEATURES},
                        res["label"], res["confidence"], item["image_file"])
        produced += 1
        if delay:
            time.sleep(delay)
    with get_conn() as c:
        inspected = c.execute("SELECT COUNT(*) FROM products WHERE batch_id=?",
                              (batch_id,)).fetchone()[0]
    return {"batch_id": batch_id, "produced": produced, "inspected": inspected,
            "status": _status(batch_id)}
 
 
def start_and_produce(operator="system"):
    """Start a new batch, plan its 50 random products and produce them."""
    require_images()                              # check BEFORE the batch is created, so a
    batch_id = start_batch(operator=operator)     # failure cannot leave a stuck RUNNING batch
    plan_batch(batch_id)
    return run_production(batch_id)
 
 
if __name__ == "__main__":
    print(start_and_produce("command-line"))