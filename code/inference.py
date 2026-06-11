"""
SageMaker inference handler for EuroSAT ResNet-18.

SageMaker calls these four functions in order:
  model_fn   → called once at endpoint startup to load the model
  input_fn   → called per request to deserialize + preprocess
  predict_fn → called per request to run the forward pass
  output_fn  → called per request to serialize the response
"""

import json
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

_MEAN = [0.344, 0.380, 0.408]
_STD = [0.177, 0.150, 0.142]

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])

_SUPPORTED_CONTENT_TYPES = {"application/octet-stream", "image/jpeg", "image/png"}


def model_fn(model_dir: str) -> torch.nn.Module:
    """Load model weights from model_dir (called once at endpoint start)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(
        torch.load(os.path.join(model_dir, "resnet18_eurosat.pth"), map_location=device)
    )
    model.to(device).eval()
    return model


def input_fn(request_body: bytes, content_type: str) -> torch.Tensor:
    """Decode image bytes and apply CLAHE + normalization."""
    if content_type not in _SUPPORTED_CONTENT_TYPES:
        raise ValueError(f"Unsupported content type: {content_type!r}. Expected one of {_SUPPORTED_CONTENT_TYPES}")

    buf = np.frombuffer(request_body, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image bytes — ensure the payload is a valid JPEG or PNG")

    # Match the CLAHE preprocessing applied by the FastAPI backend and ETL pipeline
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return _transform(Image.fromarray(rgb)).unsqueeze(0)


def predict_fn(input_data: torch.Tensor, model: torch.nn.Module) -> dict:
    """Run forward pass; return predicted class, confidence, and all class probabilities."""
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(input_data.to(device))
        probs = torch.softmax(logits, dim=1)[0]

    pred_idx = probs.argmax().item()
    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": round(probs[pred_idx].item(), 4),
        "probabilities": {cls: round(p, 4) for cls, p in zip(CLASS_NAMES, probs.tolist())},
    }


def output_fn(prediction: dict, accept: str) -> tuple[str, str]:
    """Serialize prediction to JSON."""
    return json.dumps(prediction), "application/json"
