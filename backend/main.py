import io
import json
import logging
import os
import time

import boto3
import cv2
import numpy as np
import torch
from botocore.exceptions import ClientError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
from prometheus_client import Counter, Histogram, Info
from prometheus_fastapi_instrumentator import Instrumentator
from torchvision import models, transforms
from transformers import SegformerForSemanticSegmentation

log = logging.getLogger(__name__)

S3_BUCKET      = os.environ.get("S3_BUCKET", "chicago-land-use")
LOCAL_DATA_DIR = os.environ.get("LOCAL_DATA_DIR", "")  # if set, read from filesystem before S3
s3 = boto3.client("s3")

CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]
CLASS_COLORS = [
    "#e6b800", "#1a7a1a", "#66cc66", "#888888", "#cc4400",
    "#99cc44", "#ff9933", "#d4826a", "#3399ff", "#0044aa",
]

SEG_CLASS_NAMES  = ["Background", "Building", "Road", "Water", "Barren", "Forest", "Agriculture"]
SEG_CLASS_COLORS = ["#1a1a2e", "#e05a5a", "#b0b0b0", "#1e90ff", "#c4a55a", "#2e8b57", "#f5c542"]

PREDICTION_COUNTER = Counter(
    "landuse_predictions_total", "Predictions by class", ["predicted_class"]
)
INFERENCE_DURATION = Histogram(
    "landuse_inference_seconds", "Model forward-pass duration",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
APP_INFO = Info("landuse", "Chicago land-use backend metadata")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, ".."))
model_path = os.environ.get(
    "MODEL_PATH",
    os.path.join(_project_root, "outputs", "resnet18_eurosat.pth"),
)
if not os.path.isfile(model_path):
    model_path = "/app/outputs/resnet18_eurosat.pth"

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device).eval()

# SegFormer-B2 (fine-tuned on LoveDA) — optional, loaded only if weights exist
seg_model_path  = os.environ.get("SEG_MODEL_PATH",
                                  os.path.join(_project_root, "outputs", "segformer_b2_loveda.pth"))
seg_config_path = os.path.join(os.path.dirname(seg_model_path), "segformer_config")

seg_model = None
if os.path.isdir(seg_config_path):
    seg_model = SegformerForSemanticSegmentation.from_pretrained(seg_config_path)
    seg_model.to(device).eval()
    log.info("SegFormer-B2 loaded from %s on %s", seg_config_path, device)
else:
    log.warning("SegFormer config not found at %s — /seg/* endpoints disabled", seg_config_path)

APP_INFO.info({
    "device": str(device),
    "model": "resnet18-eurosat",
    "seg_model": "segformer-b2-loveda" if seg_model else "none",
    "aoi": "chicagoland",
})

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.344, 0.380, 0.408], std=[0.177, 0.150, 0.142]),
])

app = FastAPI(title="Chicago Land Use Change API")
Instrumentator().instrument(app).expose(app)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


def _s3_bytes(key: str) -> bytes:
    if LOCAL_DATA_DIR:
        local_path = os.path.join(LOCAL_DATA_DIR, key)
        if os.path.isfile(local_path):
            with open(local_path, "rb") as f:
                return f.read()
    try:
        return s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("404", "NoSuchKey", "NoSuchBucket"):
            raise HTTPException(404, f"Not yet available — run the ETL first: {key}")
        raise HTTPException(500, str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(device), "seg_model": seg_model is not None}


@app.get("/timeseries")
async def timeseries():
    """Return the full land-use catalog produced by the Airflow ETL."""
    return json.loads(_s3_bytes("catalog/latest.json"))


@app.get("/scenes")
async def scenes():
    """Return the list of classified years available in S3."""
    catalog = json.loads(_s3_bytes("catalog/latest.json"))
    return {"years": [s["year"] for s in catalog["timeseries"]]}


@app.get("/map/{year}.png")
async def map_png(year: int):
    """Serve the colourised land-use classification map for a given year."""
    png = _s3_bytes(f"viz/{year}/map.png")
    return StreamingResponse(io.BytesIO(png), media_type="image/png")


@app.get("/change/{year_a}/{year_b}.png")
async def change_png(year_a: int, year_b: int):
    """
    Serve the change-detection map between any two years.
    Returns the pre-generated PNG if it exists; otherwise builds it on demand
    from the stored classification grids.
    """
    if year_a >= year_b:
        raise HTTPException(400, "year_a must be less than year_b")
    key = f"viz/{year_a}_{year_b}/change.png"
    try:
        png = _s3_bytes(key)
        return StreamingResponse(io.BytesIO(png), media_type="image/png")
    except HTTPException as e:
        if e.status_code != 404:
            raise

    # Build on demand
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def load_grid(yr: int) -> np.ndarray:
        raw = _s3_bytes(f"classifications/{yr}/grid.npy")
        return np.load(io.BytesIO(raw))

    ga, gb = load_grid(year_a), load_grid(year_b)
    rows, cols = min(ga.shape[0], gb.shape[0]), min(ga.shape[1], gb.shape[1])
    ga, gb = ga[:rows, :cols], gb[:rows, :cols]

    changed = ga != gb
    change_pct = float(changed.mean() * 100)

    rgba = np.full((*changed.shape, 4), [15, 17, 26, 255], dtype=np.uint8)
    rgba[changed] = [255, 80, 0, 255]

    fig, ax = plt.subplots(figsize=(10, 15), dpi=100)
    ax.imshow(rgba, interpolation="nearest")
    ax.set_title(
        f"Land Use Change  {year_a} → {year_b}\n"
        f"{change_pct:.1f}% of tiles changed",
        fontsize=12, pad=10,
    )
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    png_bytes = buf.getvalue()

    try:
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=png_bytes, ContentType="image/png")
    except Exception:
        pass

    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


@app.get("/satellite/{year}.png")
async def satellite_png(year: int):
    """
    Downsampled true-color Sentinel-2 RGB composite for a given year,
    sized for Leaflet imageOverlay (~1000 px wide).  Generated on first
    request and cached in S3 so subsequent calls are fast.
    """
    cache_key = f"viz/{year}/satellite.png"
    try:
        png = _s3_bytes(cache_key)
        return StreamingResponse(io.BytesIO(png), media_type="image/png")
    except HTTPException as e:
        if e.status_code != 404:
            raise

    raw = _s3_bytes(f"imagery/{year}/rgb.npy")
    rgb = np.load(io.BytesIO(raw))          # (H, W, 3) uint8

    h, w = rgb.shape[:2]
    scale = max(1, w // 1000)
    small = rgb[::scale, ::scale]           # simple stride subsample, fast

    img = Image.fromarray(small, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    png_bytes = buf.getvalue()

    try:
        s3.put_object(Bucket=S3_BUCKET, Key=cache_key, Body=png_bytes, ContentType="image/png")
    except Exception:
        pass

    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


@app.get("/overlay/{year}.png")
async def overlay_png(year: int):
    """
    Clean pixel-only PNG of the classification grid — no title, no legend,
    no matplotlib borders.  Sized exactly (n_cols, n_rows) pixels so Leaflet
    can stretch it to the AOI bounds without distortion.
    """
    raw = _s3_bytes(f"classifications/{year}/grid.npy")
    grid = np.load(io.BytesIO(raw))

    RIVER_IDX = CLASS_NAMES.index("River")

    color_arr = np.array(
        [[int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16), 255] for c in CLASS_COLORS],
        dtype=np.uint8,
    )
    color_arr[RIVER_IDX, 3] = 0  # transparent — basemap has no river geometry so nothing shows through

    color_map = color_arr[grid]  # (n_rows, n_cols, 4)

    img = Image.fromarray(color_map, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return StreamingResponse(io.BytesIO(buf.getvalue()), media_type="image/png")


@app.get("/seg/overlay/{year}.png")
async def seg_overlay_png(year: int):
    """
    Per-pixel segmentation overlay from SegFormer-B2 (LoveDA 7-class).
    Reads seg/{year}/mask.npy and colorizes it — same mechanism as /overlay/{year}.png
    but at full 10 m/px resolution instead of 640 m tile resolution.
    """
    if seg_model is None:
        raise HTTPException(503, "SegFormer model not loaded — run make train-segformer first")

    raw  = _s3_bytes(f"seg/{year}/mask.npy")
    mask = np.load(io.BytesIO(raw))  # (H, W) uint8

    seg_color_arr = np.array(
        [[int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)] for c in SEG_CLASS_COLORS],
        dtype=np.uint8,
    )
    color_map = seg_color_arr[mask]  # (H, W, 3)

    img = Image.fromarray(color_map, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/grid/{year}")
async def grid_data(year: int):
    """Return the raw classification grid as JSON for the frontend canvas layer."""
    raw = _s3_bytes(f"classifications/{year}/grid.npy")
    grid = np.load(io.BytesIO(raw))
    rows, cols = grid.shape
    return {"rows": rows, "cols": cols, "data": grid.flatten().tolist()}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Classify a single satellite image patch with the EuroSAT ResNet-18."""
    image_bytes = await file.read()
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(422, "Could not decode image — send a valid JPEG or PNG")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        _, pred = torch.max(model(tensor), 1)
    INFERENCE_DURATION.observe(time.perf_counter() - t0)

    cls = CLASS_NAMES[pred.item()]
    PREDICTION_COUNTER.labels(predicted_class=cls).inc()
    return {"predicted_class": cls}
