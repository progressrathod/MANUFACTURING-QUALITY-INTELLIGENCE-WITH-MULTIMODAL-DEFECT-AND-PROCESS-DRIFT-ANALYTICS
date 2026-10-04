"""Step 6 - Train the CNN.   Run:  python -m src.image_classification.train"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from src.config import IMG_MODEL_PATH, REPORTS_DIR, SEED, IMG_SIZE, IMG_CLASSES, IMG_LABELS
from src.image_classification.dataset import get_loaders
from src.image_classification.model import DefectCNN
 
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()
 
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
 
train_loader, val_loader, _ = get_loaders()
 
# class weights compensate for slightly more defective than OK images
counts = np.bincount(train_loader.dataset.targets)
weights = torch.tensor(counts.sum() / (len(counts) * counts),
                       dtype=torch.float32).to(device)
print("Train class counts:", dict(zip(IMG_CLASSES, counts.tolist())), " weights:", weights.tolist())
 
model = DefectCNN().to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
 
 
def run_epoch(loader, train: bool):
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n
 
 
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_loss = float("inf")
 
for epoch in range(1, args.epochs + 1):
    tl, ta = run_epoch(train_loader, True)
    vl, va = run_epoch(val_loader, False)
    scheduler.step(vl)
    for k, v in zip(history, (tl, vl, ta, va)):
        history[k].append(v)
    flag = ""
    if vl < best_val_loss:                         # keep the best model only
        best_val_loss = vl
        torch.save({"state_dict": model.state_dict(),
                    "img_size": IMG_SIZE,
                    "labels": IMG_LABELS}, IMG_MODEL_PATH)
        flag = "  <- saved"
    print(f"Epoch {epoch:02d}/{args.epochs} | train loss {tl:.4f} acc {ta:.3f} "
          f"| val loss {vl:.4f} acc {va:.3f}{flag}")
 
# ---- learning curves ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(history["train_loss"], label="train"); ax[0].plot(history["val_loss"], label="val")
ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
ax[1].plot(history["train_acc"], label="train"); ax[1].plot(history["val_acc"], label="val")
ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
plt.tight_layout()
plt.savefig(REPORTS_DIR / "image_training_curves.png", dpi=150)
print("Best model saved to", IMG_MODEL_PATH)
