"""Central place for paths and settings used by every script."""
from pathlib import Path
 
ROOT = Path(__file__).resolve().parents[1]
 
DATA_DIR = ROOT / "data"
IMG_RAW = DATA_DIR / "images"            # def_front/  ok_front/
IMG_SPLIT = DATA_DIR / "images_split"    # train/ val/ test/
PROCESS_DIR = DATA_DIR / "process_data"
PROCESS_CSV = PROCESS_DIR / "manufacturing_process_dataset_10000.csv"
 
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
for _d in (MODELS_DIR, REPORTS_DIR):
    _d.mkdir(exist_ok=True)
 
SEED = 42
 
# ---- image model ----
IMG_SIZE = 128
BATCH_SIZE = 32
IMG_CLASSES = ["def_front", "ok_front"]          # folder names
IMG_LABELS = ["DEFECTIVE", "OK"]                 # same order (ImageFolder sorts A-Z)
IMG_MODEL_PATH = MODELS_DIR / "image_model.pth"
 
# ---- process model ----
FEATURES = [
    "temperature", "pressure", "machine_speed", "vibration",
    "humidity", "material_thickness", "cycle_time", "tool_wear",
]
TARGET = "defect"
XGB_MODEL_PATH = MODELS_DIR / "process_xgboost.json"
XGB_THRESHOLD_PATH = MODELS_DIR / "process_threshold.json"
DRIFT_MODEL_PATH = MODELS_DIR / "drift_model.joblib"

# ---- batch tracking (Step 21) - add to the END of src/config.py ----
DB_PATH = ROOT / "data" / "quality.db"
LINE_ID = "LINE-1"
BATCH_SIZE = 50            # products per batch
 
# ---- production demo: how many DEFECTIVE products each simulated batch should contain ----
# The batch gets a random number between these two. Change them here - nowhere else.
# (The image model has a few % error, so the number it actually finds can differ by one.)
MIN_DEFECTIVE_PER_BATCH = 1
MAX_DEFECTIVE_PER_BATCH = 5
 
# Alert rules (starting values - tune them with your quality team)
WARN_DEFECTS = 8           # WARNING when a batch reaches this many defective products
STOP_DEFECTS = 11          # STOP request at this many
STREAK_WARN = 3            # WARNING when this many defective products come in a row
WARN_DRIFT = 3             # WARNING when this many readings in one batch look unusual
STOP_DRIFT = 6             # STOP request at this many
 
# EXAMPLE specification limits - replace with the real ones from your process engineer
SPEC_LIMITS = {
    "temperature": (50, 95), "pressure": (3.5, 7.0), "machine_speed": (1000, 1900),
    "vibration": (0, 4.0), "humidity": (30, 80), "material_thickness": (3.0, 5.5),
    "cycle_time": (30, 65), "tool_wear": (0, 80),
}
