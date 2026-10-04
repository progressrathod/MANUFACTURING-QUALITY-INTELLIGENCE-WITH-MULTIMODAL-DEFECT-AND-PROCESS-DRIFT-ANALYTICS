"""Step 2 - Inspect the image dataset."""
from collections import Counter
from PIL import Image
 
from src.config import IMG_RAW, IMG_CLASSES
 
counts = {}
for cls in IMG_CLASSES:
    files = sorted(p for p in (IMG_RAW / cls).iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    sizes, modes, corrupt = Counter(), Counter(), []
    for f in files:
        try:
            with Image.open(f) as im:
                im.verify()                       # detects truncated / broken files
            with Image.open(f) as im:             # re-open (verify() closes the file)
                sizes[im.size] += 1
                modes[im.mode] += 1
        except Exception:
            corrupt.append(f.name)
    counts[cls] = len(files)
    print(f"\n[{cls}]")
    print("  images       :", len(files))
    print("  sizes        :", dict(sizes))
    print("  colour modes :", dict(modes))
    print("  corrupted    :", corrupt if corrupt else "none")
 
total = sum(counts.values())
print("\n=== SUMMARY ===")
print("Total images:", total)
for cls, n in counts.items():
    print(f"  {cls}: {n} ({n / total:.1%})")
print("Balanced?    :", "yes" if min(counts.values()) / max(counts.values()) > 0.8
      else "no - slight imbalance, we will use class weights")
