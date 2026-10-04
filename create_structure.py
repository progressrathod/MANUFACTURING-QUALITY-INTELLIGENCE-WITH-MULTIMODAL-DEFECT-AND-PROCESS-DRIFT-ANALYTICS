"""Run once:  python create_structure.py   (works on Windows, macOS and Linux)"""
from pathlib import Path
 
folders = [
    "data/images/def_front", "data/images/ok_front", "data/process_data",
    "models", "notebooks", "reports",
    "src/image_classification", "src/process_prediction", "src/drift_detection",
    "api", "dashboard", "tests",
]
for f in folders:
    Path(f).mkdir(parents=True, exist_ok=True)
 
# __init__.py lets Python import our code as  src.xxx  and  api.xxx
for pkg in ["src", "src/image_classification", "src/process_prediction",
            "src/drift_detection", "api"]:
    (Path(pkg) / "__init__.py").touch()
 
# .gitkeep lets Git keep otherwise-empty folders
for f in ["models", "notebooks", "reports"]:
    (Path(f) / ".gitkeep").touch()
 
print("Project structure created.")
