
"""Step 23 - Simulate production batches for the demo.
Run: python -m src.simulate_batches --batches 3
 
Every batch holds 50 products. MIN..MAX_DEFECTIVE_PER_BATCH (src/config.py) of them are
defective, so each batch is a mix.
Process rows come from your CSV; each product gets an image of the matching class.
NOTE: in your data an image is not tied to a CSV row - this pairing is a DEMO only.
In a real plant the camera and sensors report on the same physical product.
"""
import argparse
import random
import time
 
import pandas as pd
from PIL import Image
 
from src.config import (PROCESS_CSV, FEATURES, TARGET, BATCH_SIZE,
                        MIN_DEFECTIVE_PER_BATCH, MAX_DEFECTIVE_PER_BATCH)
from src.batch_manager import (start_batch, inspect_product, resume_batch, decide_batch,
                               current_batch, get_report)
from src.image_classification.predict import predict_image
from src.production import require_images, _images, _find_image
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=2)
    ap.add_argument("--delay", type=float, default=0.0, help="seconds between products")
    ap.add_argument("--no-resume", action="store_true",
                    help="leave a stopped batch paused (so you can resume it in the dashboard)")
    args = ap.parse_args()
 
    df = pd.read_csv(PROCESS_CSV)
    bad_rows, ok_rows = df[df[TARGET] == 1], df[df[TARGET] == 0]
    require_images()                                        # clear error if the images are missing
    imgs = {1: _images("def_front"), 0: _images("ok_front")}
 
    for _ in range(args.batches):
        batch_id = start_batch()
        n_bad = random.randint(MIN_DEFECTIVE_PER_BATCH, MAX_DEFECTIVE_PER_BATCH)
        rows = pd.concat([bad_rows.sample(n_bad), ok_rows.sample(BATCH_SIZE - n_bad)])
        rows = rows.sample(frac=1).reset_index(drop=True)   # shuffle: defects appear anywhere
        print(f"\n=== {batch_id}: {n_bad} defective products planned ===")
        for _, r in rows.iterrows():
            cur = current_batch()
            if cur is None or cur["batch_id"] != batch_id or cur["status"] not in ("RUNNING", "PAUSED"):
                break                                       # batch finished
            if cur["status"] == "PAUSED":
                if args.no_resume:
                    print("  Line STOPPED - waiting for the manager. Batch left paused.")
                    return
                resume_batch(batch_id, "demo-manager", "Demo: machine checked, continue")
                print("  (demo manager resumed the batch)")
            truth = int(r[TARGET])
            name = random.choice(imgs[truth])
            f = _find_image("def_front" if truth else "ok_front", name)
            with Image.open(f) as im:
                res = predict_image(im)
            out = inspect_product(batch_id, {k: float(r[k]) for k in FEATURES},
                                  res["label"], res["confidence"], name)
            print(f"  {out['product_id']}  {out['final_status']}")
            time.sleep(args.delay)
        report = get_report(batch_id)
        print(f"  -> {report['status']}: {report['ok']} OK, {report['defective']} defective, "
              f"{report['review']} to review")
        if report["status"] == "ON_HOLD":                   # the next batch cannot start until
            choice = "SCRAP" if report["defect_rate"] >= 20 else "RELEASE"   # a decision is made
            decide_batch(batch_id, choice, "demo-manager", f"Demo decision: {choice}")
            print(f"  (demo manager decided: {choice})")
 
 
if __name__ == "__main__":
    main()