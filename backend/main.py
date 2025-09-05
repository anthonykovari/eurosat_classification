import io
import torch
from torchvision import transforms, models
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image
import os
import sys
from fastapi.middleware.cors import CORSMiddleware
import uuid


app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev only — allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Get model path relative to script
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
model_path = os.path.join(project_root, "outputs", "resnet18_eurosat.pth")

# Fallback path inside Docker container (workdir /app)
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

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0)

UPLOAD_DIR = "uploaded"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Serve uploaded images
app.mount("/images", StaticFiles(directory=UPLOAD_DIR), name="images")

...

import uuid

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    print(f"Received file: {file.filename}")
    image_bytes = await file.read()
    
    # Save image with a safe unique name
    ext = os.path.splitext(file.filename)[-1]
    safe_filename = f"{uuid.uuid4()}{ext}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    with open(file_path, "wb") as f:
        f.write(image_bytes)

    input_tensor = preprocess_image(image_bytes).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, preds = torch.max(outputs, 1)

    predicted_class = class_names[preds.item()]
    return {
        "filename": safe_filename,
        "predicted_class": predicted_class,
        "image_url": f"/images/{safe_filename}"
    }
