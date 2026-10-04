"""Step 7 - Evaluate the CNN on the unseen test set."""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)
 
from src.config import IMG_MODEL_PATH, REPORTS_DIR, IMG_LABELS
from src.image_classification.dataset import get_loaders
from src.image_classification.model import DefectCNN
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(IMG_MODEL_PATH, map_location=device)
model = DefectCNN().to(device)
model.load_state_dict(ckpt["state_dict"])
model.eval()
 
_, _, test_loader = get_loaders()
y_true, y_pred = [], []
with torch.no_grad():
    for x, y in test_loader:
        out = model(x.to(device))
        y_pred += out.argmax(1).cpu().tolist()
        y_true += y.tolist()
y_true, y_pred = np.array(y_true), np.array(y_pred)
 
# class index 0 = DEFECTIVE, so "positive" (what we want to catch) = 0
print(f"Accuracy : {accuracy_score(y_true, y_pred):.3f}")
print(f"Precision: {precision_score(y_true, y_pred, pos_label=0):.3f}  (defective)")
print(f"Recall   : {recall_score(y_true, y_pred, pos_label=0):.3f}  (defective)")
print(f"F1-score : {f1_score(y_true, y_pred, pos_label=0):.3f}  (defective)")
print("\n", classification_report(y_true, y_pred, target_names=IMG_LABELS, digits=3))
 
cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix (rows = true, cols = predicted):\n", cm)
ConfusionMatrixDisplay(cm, display_labels=IMG_LABELS).plot(cmap="Blues")
plt.title("Image model - test set")
plt.savefig(REPORTS_DIR / "image_confusion_matrix.png", dpi=150, bbox_inches="tight")
print("Saved confusion matrix to reports/")
