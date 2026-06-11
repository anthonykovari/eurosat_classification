import os
import uuid

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import models, transforms

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev only — allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
model_path = os.path.join(project_root, "outputs", "resnet18_eurosat.pth")

if not os.path.isfile(model_path):
    model_path = "/app/outputs/resnet18_eurosat.pth"

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 10)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

mean = [0.344, 0.380, 0.408]
std = [0.177, 0.150, 0.142]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

class_names = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]


def apply_clahe(bgr: np.ndarray) -> np.ndarray:
    """Apply CLAHE per channel in LAB color space to enhance local contrast.

    Satellite images often have low contrast in individual bands; CLAHE
    normalizes each region independently without washing out the overall tone.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image — ensure the file is a valid JPEG or PNG")
    bgr = apply_clahe(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    return transform(image).unsqueeze(0)


UPLOAD_DIR = "uploaded"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/images", StaticFiles(directory=UPLOAD_DIR), name="images")


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(device)}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    print(f"Received file: {file.filename}")
    image_bytes = await file.read()

    ext = os.path.splitext(file.filename)[-1]
    safe_filename = f"{uuid.uuid4()}{ext}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    with open(file_path, "wb") as f:
        f.write(image_bytes)

    try:
        input_tensor = preprocess_image(image_bytes).to(device)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    with torch.no_grad():
        outputs = model(input_tensor)
        _, preds = torch.max(outputs, 1)

    predicted_class = class_names[preds.item()]
    return {
        "filename": safe_filename,
        "predicted_class": predicted_class,
        "image_url": f"/images/{safe_filename}"
    }
