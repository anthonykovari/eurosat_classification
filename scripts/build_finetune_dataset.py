"""
Build a Chicagoland fine-tuning dataset for ResNet-18.

Labels classification-grid cells using Lake Michigan's geographic polygon
(SeaLake) and 2019 pseudo-labels (land classes), then extracts 64×64 patches
from the 2019, 2020, 2021, and 2023 RGB mosaics.

SeaLake patches are drawn only from cells whose centroid falls east of the
western shoreline (inside LAKE_MICHIGAN_POLY), so no land cells bleed in.
2021 and 2023 are included specifically to fix Forest misclassification over
Lake Michigan water.

Output: data/finetune/<ClassName>/<filename>.png  (ImageFolder layout)

Usage:
    python3 scripts/build_finetune_dataset.py
"""

from __future__ import annotations

import io
import random
from collections import defaultdict
from pathlib import Path

import boto3
import numpy as np
from PIL import Image
from shapely.geometry import Point, Polygon

# ── S3 / mosaic config ────────────────────────────────────────────────────────

S3_ENDPOINT = "http://localhost:9010"
S3_CREDS    = {"aws_access_key_id": "minioadmin", "aws_secret_access_key": "minioadmin"}
BUCKET      = "chicago-land-use"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR    = PROJECT_ROOT / "data" / "cache"
OUT_DIR      = PROJECT_ROOT / "data" / "finetune"

# ── Grid / AOI config ─────────────────────────────────────────────────────────

TILE      = 64
N_ROWS    = 222
N_COLS    = 116
MOSAIC_H  = N_ROWS * TILE   # 14208 px
MOSAIC_W  = N_COLS * TILE   # 7424  px
LAT_MAX   = 42.75
LAT_MIN   = 41.45
LON_MIN   = -88.4
LON_MAX   = -87.5
LAT_SPAN  = LAT_MAX - LAT_MIN  # 1.3°
LON_SPAN  = LON_MAX - LON_MIN  # 0.9°

LAND_CAP  = 400   # max patches per non-lake class
SEED      = 42

CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

# ── Lake Michigan western-shore polygon ───────────────────────────────────────
# Traces the IL/WI/IN shoreline from south to north, closes via AOI east edge.
# Coordinates are (lon, lat) for shapely convention.

LAKE_MICHIGAN_POLY = Polygon([
    # Shoreline traced south→north (western boundary of lake within AOI).
    # Coordinates verified against Lake Michigan western shore geography.
    (-87.52, 41.45),   # Hammond/Whiting IN
    (-87.52, 41.63),   # Indiana/Illinois border
    (-87.52, 41.72),   # Calumet area
    (-87.54, 41.78),   # South Chicago
    (-87.57, 41.83),   # South-side Chicago lakefront
    (-87.60, 41.87),   # Chicago (Museum Campus / Lake Shore Dr)
    (-87.63, 41.93),   # Lincoln Park
    (-87.67, 42.02),   # Rogers Park / Evanston south
    (-87.68, 42.05),   # Evanston
    (-87.71, 42.09),   # Wilmette
    (-87.74, 42.13),   # Winnetka
    (-87.80, 42.20),   # Highland Park
    (-87.83, 42.27),   # Lake Bluff / North Chicago
    (-87.84, 42.35),   # Waukegan
    (-87.83, 42.46),   # Zion
    (-87.82, 42.58),   # Kenosha
    (-87.79, 42.75),   # Racine area
    # Close via AOI eastern boundary
    (-87.50, 42.75),
    (-87.50, 41.45),
    (-87.52, 41.45),
])
assert LAKE_MICHIGAN_POLY.is_valid, "Lake polygon is invalid — check coordinates"


def cell_centroid(gr: int, gc: int) -> tuple[float, float]:
    """Return (lat, lon) centroid of classification grid cell (gr, gc)."""
    lat = LAT_MAX - (gr * TILE + TILE // 2) * (LAT_SPAN / MOSAIC_H)
    lon = LON_MIN + (gc * TILE + TILE // 2) * (LON_SPAN / MOSAIC_W)
    return lat, lon


def s3_download(s3, key: str, cache_path: Path) -> np.ndarray:
    if cache_path.exists():
        print(f"  cache hit  {cache_path.name}")
        return np.load(cache_path)
    print(f"  downloading {key} …")
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    data = np.load(io.BytesIO(obj["Body"].read()))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, data)
    print(f"  cached → {cache_path.name}  shape={data.shape}")
    return data


def save_patch(patch: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(patch).save(path, format="PNG")


def main() -> None:
    random.seed(SEED)

    s3 = boto3.client("s3", endpoint_url=S3_ENDPOINT, **S3_CREDS)

    print("Downloading mosaics and 2019 grid …")
    rgb_2019  = s3_download(s3, "imagery/2019/rgb.npy",          CACHE_DIR / "imagery_2019_rgb.npy")
    rgb_2020  = s3_download(s3, "imagery/2020/rgb.npy",          CACHE_DIR / "imagery_2020_rgb.npy")
    rgb_2021  = s3_download(s3, "imagery/2021/rgb.npy",          CACHE_DIR / "imagery_2021_rgb.npy")
    rgb_2023  = s3_download(s3, "imagery/2023/rgb.npy",          CACHE_DIR / "imagery_2023_rgb.npy")
    grid_2019 = s3_download(s3, "classifications/2019/grid.npy", CACHE_DIR / "classifications_2019_grid.npy")

    # Crop mosaics to exact TILE multiple
    rgb_2019 = rgb_2019[:MOSAIC_H, :MOSAIC_W]
    rgb_2020 = rgb_2020[:MOSAIC_H, :MOSAIC_W]
    rgb_2021 = rgb_2021[:MOSAIC_H, :MOSAIC_W]
    rgb_2023 = rgb_2023[:MOSAIC_H, :MOSAIC_W]

    print(f"\nClassifying {N_ROWS}×{N_COLS} grid cells …")
    lake_cells: list[tuple[int, int]] = []
    land_cells: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)

    for gr in range(N_ROWS):
        for gc in range(N_COLS):
            lat, lon = cell_centroid(gr, gc)
            if LAKE_MICHIGAN_POLY.contains(Point(lon, lat)):
                lake_cells.append((gr, gc))
            else:
                label = int(grid_2019[gr, gc])
                land_cells[label].append((gr, gc))

    print(f"  lake cells : {len(lake_cells)}")
    print(f"  land cells : {sum(len(v) for v in land_cells.values())}")

    # ── Write SeaLake patches (all years) ────────────────────────────────────
    # 2021 and 2023 added to fix Forest misclassification over lake water.
    print("\nWriting SeaLake patches (2019 + 2020 + 2021 + 2023) …")
    for gr, gc in lake_cells:
        y0, x0 = gr * TILE, gc * TILE
        for year, rgb in [(2019, rgb_2019), (2020, rgb_2020), (2021, rgb_2021), (2023, rgb_2023)]:
            save_patch(rgb[y0:y0 + TILE, x0:x0 + TILE],
                       OUT_DIR / "SeaLake" / f"sealake_{gr:04d}_{gc:04d}_{year}.png")

    # ── Write land patches (2019 + 2020, capped per class) ───────────────────
    print("Writing land patches (2019 + 2020, pseudo-labels from 2019 grid) …")
    for label, cells in land_cells.items():
        class_name = CLASS_NAMES[label]
        random.shuffle(cells)
        sampled = cells[:LAND_CAP]
        for gr, gc in sampled:
            y0, x0 = gr * TILE, gc * TILE
            save_patch(rgb_2019[y0:y0 + TILE, x0:x0 + TILE],
                       OUT_DIR / class_name / f"{class_name.lower()}_{gr:04d}_{gc:04d}_2019.png")
            save_patch(rgb_2020[y0:y0 + TILE, x0:x0 + TILE],
                       OUT_DIR / class_name / f"{class_name.lower()}_{gr:04d}_{gc:04d}_2020.png")
        print(f"  {class_name:<25} {len(sampled) * 2:>4} patches")

    # ── Verify all 10 classes present ─────────────────────────────────────────
    print("\nClass distribution in data/finetune/:")
    total = 0
    for name in CLASS_NAMES:
        count = len(list((OUT_DIR / name).glob("*.png")))
        assert count > 0, f"No patches found for class {name} — check polygon or grid"
        print(f"  {name:<25} {count:>5}")
        total += count
    print(f"  {'TOTAL':<25} {total:>5}")
    print(f"\nDone. Dataset written to {OUT_DIR}")


if __name__ == "__main__":
    main()
