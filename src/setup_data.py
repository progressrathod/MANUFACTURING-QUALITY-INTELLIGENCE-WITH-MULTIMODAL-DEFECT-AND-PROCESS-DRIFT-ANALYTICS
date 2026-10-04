"""Unzip the image dataset and copy the CSV into the project's data folders.
 
Put these two files in the project root first:
    casting_512x512.zip
    manufacturing_process_dataset_10000.csv
"""
import shutil
import zipfile
 
from src.config import ROOT, IMG_RAW, PROCESS_DIR, PROCESS_CSV
 
ZIP_FILE = ROOT / "casting_512x512.zip"
CSV_FILE = ROOT / "manufacturing_process_dataset_10000.csv"
 
 
def main():
    IMG_RAW.mkdir(parents=True, exist_ok=True)
    PROCESS_DIR.mkdir(parents=True, exist_ok=True)
 
    if ZIP_FILE.exists():
        tmp = ROOT / "_unzip_tmp"
        with zipfile.ZipFile(ZIP_FILE) as z:
            z.extractall(tmp)
        for cls in ("def_front", "ok_front"):
            src = next(tmp.rglob(cls))          # finds casting_512x512/<cls>
            dst = IMG_RAW / cls
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
        shutil.rmtree(tmp)
        print("Images ready in", IMG_RAW)
    else:
        print("Zip not found:", ZIP_FILE)
 
    if CSV_FILE.exists():
        shutil.copy(CSV_FILE, PROCESS_CSV)
        print("CSV ready in", PROCESS_CSV)
    else:
        print("CSV not found:", CSV_FILE)
 
 
if __name__ == "__main__":
    main()

