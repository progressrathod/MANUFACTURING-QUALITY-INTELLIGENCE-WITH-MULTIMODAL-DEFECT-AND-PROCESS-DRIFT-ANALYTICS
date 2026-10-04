"""Reusable prediction function (used by the API and the tests)."""
from functools import lru_cache
import torch
from PIL import Image
from torchvision import transforms
 
from src.config import IMG_MODEL_PATH, IMG_SIZE, IMG_LABELS
from src.image_classification.model import DefectCNN
 
_tf = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
 
 
@lru_cache(maxsize=1)
def load_model():
    ckpt = torch.load(IMG_MODEL_PATH, map_location="cpu")
    model = DefectCNN()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
 
 
def predict_image(img: Image.Image) -> dict:
    """img: a PIL image. Returns label + probabilities in percent."""
    x = _tf(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(load_model()(x), dim=1)[0]
    idx = int(probs.argmax())
    return {
        "label": IMG_LABELS[idx],
        "confidence": round(float(probs[idx]) * 100, 2),
        "probabilities": {IMG_LABELS[i]: round(float(probs[i]) * 100, 2) for i in range(2)},
    }
 
 
if __name__ == "__main__":
    import sys
    print(predict_image(Image.open(sys.argv[1])))
