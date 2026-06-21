"""
Generate synthetic Chicagoland imagery and run both models (ResNet-18 + SegFormer-B2)
to produce all data the backend needs — without requiring Airflow or CDSE credentials.

Outputs land in LOCAL_S3_DIR (default: local_s3/) mirroring the S3 key layout:
  imagery/{year}/rgb.npy
  classifications/{year}/grid.npy
  seg/{year}/mask.npy
  catalog/latest.json

Usage:
  python3 scripts/generate_seg_masks.py                         # all years
  python3 scripts/generate_seg_masks.py --years 2021 2022       # specific years
  python3 scripts/generate_seg_masks.py --out-dir /tmp/local_s3 # custom dir
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
from itertools import product as iproduct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as tvm
from PIL import Image
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── constants (must match backend/main.py and the DAG) ──────────────────────
AOI_BBOX   = (-88.4, 41.45, -87.5, 42.75)
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
SEGFORMER_MEAN   = [0.485, 0.456, 0.406]
SEGFORMER_STD    = [0.229, 0.224, 0.225]

RESNET_MEAN = [0.344, 0.380, 0.408]
RESNET_STD  = [0.177, 0.150, 0.142]
TILE        = 64
SEG_TILE    = 512
SEG_STRIDE  = 256


# ── synthetic imagery ────────────────────────────────────────────────────────
def make_synthetic_rgb(year: int, years: list[int], h: int = 512, w: int = 512) -> np.ndarray:
    """
    Growing urban core (grey) surrounded by agricultural land (green).
    Generated at 1/8 resolution then upscaled so blobs are spatially coherent
    (pixel-by-pixel random noise looks like TV static at full resolution).
    """
    i = years.index(year)
    rng = np.random.default_rng(42 + i)
    SCALE = 8
    lh, lw = max(1, h // SCALE), max(1, w // SCALE)

    low = np.zeros((lh, lw, 3), dtype=np.uint8)
    low[:, :, 0] = rng.integers(40,  90,  (lh, lw), dtype=np.uint8)
    low[:, :, 1] = rng.integers(80,  140, (lh, lw), dtype=np.uint8)
    low[:, :, 2] = rng.integers(30,  70,  (lh, lw), dtype=np.uint8)

    cy, cx = lh // 2, lw // 2
    radius = max(1, (70 + i * 14) // SCALE)
    yy, xx = np.ogrid[:lh, :lw]
    urban = (yy - cy) ** 2 + (xx - cx) ** 2 < radius ** 2
    n = int(urban.sum())
    if n:
        low[urban, 0] = rng.integers(90, 155, n, dtype=np.uint8)
        low[urban, 1] = rng.integers(90, 155, n, dtype=np.uint8)
        low[urban, 2] = rng.integers(90, 155, n, dtype=np.uint8)

    return np.array(Image.fromarray(low).resize((w, h), Image.BILINEAR))


# ── ResNet-18 tile classification ────────────────────────────────────────────
def classify_tiles(rgb: np.ndarray, model: torch.nn.Module,
                   device: torch.device) -> np.ndarray:
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=RESNET_MEAN, std=RESNET_STD),
    ])
    H, W = rgb.shape[:2]
    n_rows, n_cols = H // TILE, W // TILE
    rgb_crop = rgb[:n_rows * TILE, :n_cols * TILE]
    tiles = (rgb_crop
             .reshape(n_rows, TILE, n_cols, TILE, 3)
             .transpose(0, 2, 1, 3, 4)
             .reshape(-1, TILE, TILE, 3))
    BATCH = 64
    preds: list[int] = []
    for start in range(0, len(tiles), BATCH):
        batch = [Image.fromarray(tiles[j]) for j in range(start, min(start + BATCH, len(tiles)))]
        t = torch.stack([preprocess(img) for img in batch]).to(device)
        with torch.no_grad():
            preds.extend(model(t).argmax(1).cpu().tolist())
    return np.array(preds, dtype=np.uint8).reshape(n_rows, n_cols)


# ── SegFormer sliding-window segmentation ────────────────────────────────────
def segment_pixels(rgb: np.ndarray, model: torch.nn.Module,
                   device: torch.device) -> np.ndarray:
    seg_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=SEGFORMER_MEAN, std=SEGFORMER_STD),
    ])
    H, W = rgb.shape[:2]
    ys = list(range(0, max(1, H - SEG_TILE + 1), SEG_STRIDE))
    xs = list(range(0, max(1, W - SEG_TILE + 1), SEG_STRIDE))
    if not ys or ys[-1] + SEG_TILE < H:
        ys.append(max(0, H - SEG_TILE))
    if not xs or xs[-1] + SEG_TILE < W:
        xs.append(max(0, W - SEG_TILE))

    accumulator = np.zeros((H, W, len(SEG_CLASS_NAMES)), dtype=np.float32)
    count_map   = np.zeros((H, W), dtype=np.float32)

    BATCH_WIN = 4
    coords = list(iproduct(ys, xs))
    for batch_start in range(0, len(coords), BATCH_WIN):
        batch_coords = coords[batch_start:batch_start + BATCH_WIN]
        tensors = []
        for y0, x0 in batch_coords:
            patch = rgb[y0:y0 + SEG_TILE, x0:x0 + SEG_TILE]
            if patch.shape[0] < SEG_TILE or patch.shape[1] < SEG_TILE:
                padded = np.zeros((SEG_TILE, SEG_TILE, 3), dtype=np.uint8)
                padded[:patch.shape[0], :patch.shape[1]] = patch
                patch = padded
            tensors.append(seg_transform(Image.fromarray(patch)))

        batch_t = torch.stack(tensors).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(pixel_values=batch_t).logits  # (B, C, H/4, W/4)
        up = F.interpolate(logits, size=(SEG_TILE, SEG_TILE),
                           mode="bilinear", align_corners=False)
        up_np = up.cpu().float().numpy()

        for i, (y0, x0) in enumerate(batch_coords):
            h_sl = min(SEG_TILE, H - y0)
            w_sl = min(SEG_TILE, W - x0)
            accumulator[y0:y0 + h_sl, x0:x0 + w_sl] += \
                up_np[i, :, :h_sl, :w_sl].transpose(1, 2, 0)
            count_map[y0:y0 + h_sl, x0:x0 + w_sl] += 1.0

        if batch_start % (BATCH_WIN * 5) == 0:
            log.info("  seg window %d/%d", batch_start, len(coords))

    return (accumulator / np.maximum(count_map[:, :, np.newaxis], 1.0)
            ).argmax(axis=2).astype(np.uint8)


# ── local-S3 helpers ─────────────────────────────────────────────────────────
def write_local(out_dir: Path, key: str, data: bytes) -> None:
    path = out_dir / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


# ── main ─────────────────────────────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    years   = args.years
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s  |  output: %s  |  years: %s", device, out_dir, years)

    # Load ResNet-18
    resnet = tvm.resnet18(weights=None)
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, 10)
    resnet.load_state_dict(torch.load(args.resnet_path, map_location=device))
    resnet.to(device).eval()
    log.info("ResNet-18 loaded from %s", args.resnet_path)

    # Load SegFormer-B2 — save_pretrained writes model.safetensors into seg_config dir,
    # so from_pretrained loads both architecture and weights in one call.
    seg_model = None
    if os.path.isdir(args.seg_config):
        seg_model = SegformerForSemanticSegmentation.from_pretrained(args.seg_config)
        seg_model.to(device).eval()
        log.info("SegFormer-B2 loaded from %s", args.seg_config)
    else:
        log.warning(
            "SegFormer config not found at %s — skipping pixel segmentation\n"
            "Run 'make train-segformer' first, then re-run this script.",
            args.seg_config,
        )

    year_stats:     list[dict] = []
    seg_year_stats: list[dict] = []
    changes:        list[dict] = []
    prev_grid: np.ndarray | None = None
    prev_year: int | None = None

    for year in sorted(years):
        log.info("── Year %d ──────────────────────────────", year)

        # Imagery
        rgb = make_synthetic_rgb(year, sorted(years))
        write_local(out_dir, f"imagery/{year}/rgb.npy", npy_bytes(rgb))

        # ResNet-18 tile classification
        grid = classify_tiles(rgb, resnet, device)
        write_local(out_dir, f"classifications/{year}/grid.npy", npy_bytes(grid))
        counts = np.bincount(grid.ravel(), minlength=10)
        total  = int(counts.sum())
        class_pct = {CLASS_NAMES[i]: round(float(counts[i] / total * 100), 2) for i in range(10)}
        year_stats.append({"year": year, "total_tiles": total, "classes": class_pct})
        log.info("  tiles classified: %d  top: %s", total,
                 sorted(class_pct.items(), key=lambda x: -x[1])[:3])

        # Change detection vs previous year
        if prev_grid is not None and prev_year is not None:
            rows = min(prev_grid.shape[0], grid.shape[0])
            cols = min(prev_grid.shape[1], grid.shape[1])
            changed   = prev_grid[:rows, :cols] != grid[:rows, :cols]
            change_pct = float(changed.mean() * 100)
            transitions: dict[str, dict[str, int]] = {}
            for fc, tc in zip(prev_grid[:rows, :cols][changed].tolist(),
                               grid[:rows, :cols][changed].tolist()):
                fn, tn = CLASS_NAMES[fc], CLASS_NAMES[tc]
                transitions.setdefault(fn, {})
                transitions[fn][tn] = transitions[fn].get(tn, 0) + 1
            changes.append({
                "from_year": prev_year, "to_year": year,
                "changed_tiles": int(changed.sum()),
                "change_pct": round(change_pct, 2),
                "transitions": transitions,
            })
            log.info("  change vs %d: %.1f%%", prev_year, change_pct)
        prev_grid, prev_year = grid, year

        # SegFormer pixel segmentation
        if seg_model is not None:
            log.info("  running SegFormer segmentation …")
            mask = segment_pixels(rgb, seg_model, device)
            write_local(out_dir, f"seg/{year}/mask.npy", npy_bytes(mask))
            seg_counts = np.bincount(mask.ravel(), minlength=len(SEG_CLASS_NAMES))
            seg_total  = int(seg_counts.sum())
            seg_pct = {SEG_CLASS_NAMES[i]: round(float(seg_counts[i] / seg_total * 100), 2)
                       for i in range(len(SEG_CLASS_NAMES))}
            seg_year_stats.append({"year": year, "pixel_count": seg_total, "seg_classes": seg_pct})
            log.info("  seg classes: %s",
                     sorted(seg_pct.items(), key=lambda x: -x[1])[:3])

    # Catalog
    catalog = {
        "aoi":              {"bbox": list(AOI_BBOX), "name": "Chicagoland"},
        "class_names":      CLASS_NAMES,
        "class_colors":     CLASS_COLORS,
        "timeseries":       year_stats,
        "changes":          changes,
        "seg_class_names":  SEG_CLASS_NAMES,
        "seg_class_colors": SEG_CLASS_COLORS,
        "seg_timeseries":   seg_year_stats,
    }
    write_local(out_dir, "catalog/latest.json",
                json.dumps(catalog, indent=2).encode())
    log.info("Catalog written → %s/catalog/latest.json", out_dir)
    log.info("Done. LOCAL_DATA_DIR=%s", out_dir.resolve())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Chicagoland demo data locally")
    p.add_argument("--years",      type=int, nargs="+",
                   default=list(range(2019, 2025)))
    p.add_argument("--out-dir",    default="local_s3")
    p.add_argument("--resnet-path", default="outputs/resnet18_eurosat.pth")
    p.add_argument("--seg-path",   default="outputs/segformer_b2_loveda.pth")
    p.add_argument("--seg-config", default="outputs/segformer_config")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
