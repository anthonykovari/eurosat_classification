"""
Chicago Land Use Change Pipeline

For 2019-2024, fetches Sentinel-2 L2A true-colour (B04/B03/B02) imagery for
the greater Chicagoland region (city + far suburbs + Kenosha + Gary) via the
CDSE Sentinel Hub Process API.  Only 3 bands are requested — the server-side
evalscript converts DN to uint8 RGB, so the wire transfer is one small array
per year.

Each year's raster is tiled into 64×64 patches and classified with the
EuroSAT ResNet-18 model, producing annual land-use grids from the same 10
classes the model was trained on.  Consecutive year grids are diffed to build
a change-detection time series, all stored in S3.

Set Airflow Variable 'sprawl_skip_cdse' = 'true' to use synthetic imagery
(no CDSE quota used) — the full classification + change pipeline still runs.
"""

from __future__ import annotations

import io
import json
import logging
import os
from datetime import timedelta

import boto3
import numpy as np
from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable
from airflow.utils.dates import days_ago
from botocore.exceptions import ClientError

log = logging.getLogger(__name__)

S3_BUCKET = Variable.get("sprawl_s3_bucket", default_var="chicago-land-use")

# Chicagoland: city proper + far suburbs + Kenosha (WI) + Gary (IN)
AOI_BBOX = (-88.4, 41.45, -87.5, 42.75)  # lon_min, lat_min, lon_max, lat_max

TARGET_YEARS = list(range(
    int(Variable.get("sprawl_start_year", default_var="2019")),
    int(Variable.get("sprawl_end_year", default_var="2025")),
))

CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]
CLASS_COLORS = [
    "#e6b800",  # AnnualCrop           - amber
    "#1a7a1a",  # Forest               - dark green
    "#66cc66",  # HerbaceousVegetation - light green
    "#888888",  # Highway              - grey
    "#cc4400",  # Industrial           - rust
    "#99cc44",  # Pasture              - yellow-green
    "#ff9933",  # PermanentCrop        - orange
    "#d4826a",  # Residential          - terra cotta
    "#3399ff",  # River                - light blue
    "#0044aa",  # SeaLake              - navy
]

# Only B04/B03/B02 are fetched; the evalscript converts DN → uint8 on the
# Sentinel Hub side so nothing else crosses the wire.
EVALSCRIPT = """
//VERSION=3
// Least-cloudy single scene from the growing season (June-September).
// SIMPLE mosaicking picks the least-cloudy acquisition — ~1 input scene per
// pixel vs ~25 for ORBIT median, so ~25x cheaper in processing units.
// Clip at 3000 DN to match EuroSAT training brightness.
function setup() {
  return {
    input: [{ bands: ["B02", "B03", "B04", "CLM"] }],
    output: { bands: 3, sampleType: "UINT8" },
    mosaicking: "SIMPLE"
  };
}
function evaluatePixel(samples) {
  var s = samples[0];
  var f = 255.0 / 3000.0;
  return [
    Math.min(255, Math.round(s.B04 * f)),
    Math.min(255, Math.round(s.B03 * f)),
    Math.min(255, Math.round(s.B02 * f)),
  ];
}
"""

default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="chicago_land_use_pipeline",
    default_args=default_args,
    description="Chicagoland land-use change: CDSE RGB → EuroSAT ResNet-18 → S3",
    schedule_interval="@yearly",
    start_date=days_ago(1),
    catchup=False,
    tags=["etl", "sentinel2", "land-use", "chicago", "cdse"],
) as dag:

    @task()
    def fetch_imagery() -> dict:
        """
        Download a June–August least-cloud-cover Sentinel-2 composite at 10 m
        for each target year via the CDSE Sentinel Hub Process API.

        The full Chicagoland AOI at 10 m is ~7 400 × 14 400 pixels — far larger
        than the API's 2 500 × 2 500 px per-request limit.  We split the AOI
        into a grid of tiles (each ≤ 2 500 px on either side), request them in
        sequence, then stitch north-to-south / west-to-east into one array.

        Skips years already in S3 unless the stored file is suspiciously small
        (i.e. leftover synthetic data < 1 MB), which forces a re-fetch.
        """
        import math

        if Variable.get("sprawl_skip_cdse", default_var="false").lower() == "true":
            log.info("sprawl_skip_cdse=true — using synthetic imagery")
            return _synthetic_imagery()

        from sentinelhub import (
            BBox, CRS, DataCollection, MimeType,
            SHConfig, SentinelHubRequest, bbox_to_dimensions,
        )

        config = SHConfig(
            sh_client_id=os.environ["CDSE_CLIENT_ID"],
            sh_client_secret=os.environ["CDSE_CLIENT_SECRET"],
            sh_base_url="https://sh.dataspace.copernicus.eu",
            sh_token_url=os.environ.get(
                "CDSE_TOKEN_URL",
                "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
                "protocol/openid-connect/token",
            ),
            download_timeout_seconds=300,  # ORBIT mosaicking is slow server-side
        )
        S2_L2A = DataCollection.SENTINEL2_L2A.define_from(
            "s2l2a_cdse", service_url="https://sh.dataspace.copernicus.eu"
        )

        MAX_PX = 2500
        lon_min, lat_min, lon_max, lat_max = AOI_BBOX
        full_bbox = BBox(bbox=AOI_BBOX, crs=CRS.WGS84)
        full_w, full_h = bbox_to_dimensions(full_bbox, resolution=10)

        n_cols = math.ceil(full_w / MAX_PX)
        n_rows = math.ceil(full_h / MAX_PX)
        col_step = (lon_max - lon_min) / n_cols
        row_step = (lat_max - lat_min) / n_rows

        log.info(
            "AOI at 10 m: %d × %d px → tiling into %d cols × %d rows (%d requests/year)",
            full_w, full_h, n_cols, n_rows, n_cols * n_rows,
        )

        s3 = boto3.client("s3")
        uploaded: dict[str, str] = {}

        for year in TARGET_YEARS:
            key = f"imagery/{year}/rgb.npy"
            try:
                meta = s3.head_object(Bucket=S3_BUCKET, Key=key)
                if meta["ContentLength"] >= 1_000_000:
                    log.info("Year %d already in S3 (%.0f MB) — skipping",
                             year, meta["ContentLength"] / 1e6)
                    uploaded[str(year)] = key
                    continue
                # Synthetic placeholder — delete and re-fetch
                s3.delete_object(Bucket=S3_BUCKET, Key=key)
                log.info("Year %d: removed synthetic placeholder, fetching real imagery", year)
            except ClientError:
                pass

            # Fetch all tiles for this year
            # strips[r] = horizontal strip for tile row r (r=0 is southernmost)
            strips: list[np.ndarray] = []
            for r in range(n_rows):
                row_tiles: list[np.ndarray] = []
                for c in range(n_cols):
                    tile_bbox = BBox(
                        bbox=(
                            lon_min + c * col_step,
                            lat_min + r * row_step,
                            lon_min + (c + 1) * col_step,
                            lat_min + (r + 1) * row_step,
                        ),
                        crs=CRS.WGS84,
                    )
                    tile_w, tile_h = bbox_to_dimensions(tile_bbox, resolution=10)

                    request = SentinelHubRequest(
                        evalscript=EVALSCRIPT,
                        input_data=[SentinelHubRequest.input_data(
                            data_collection=S2_L2A,
                            time_interval=(f"{year}-06-01", f"{year}-09-30"),
                            # no mosaicking_order — ORBIT median is handled in evalscript
                        )],
                        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
                        bbox=tile_bbox,
                        size=(tile_w, tile_h),
                        config=config,
                    )
                    data = request.get_data()
                    if not data or data[0] is None:
                        log.warning("Year %d tile (%d,%d): no data — filling zeros", year, r, c)
                        row_tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
                    else:
                        tile_arr = data[0].astype(np.uint8)
                        row_tiles.append(tile_arr)
                        log.info("Year %d tile (%d,%d) ✓  shape=%s", year, r, c, tile_arr.shape)

                # Crop to uniform height before horizontal concat — Sentinel Hub
                # returns tiles whose pixel height can differ by a few pixels
                # due to geographic projection rounding at different longitudes.
                min_h = min(t.shape[0] for t in row_tiles)
                strips.append(np.concatenate([t[:min_h] for t in row_tiles], axis=1))

            # Stitch: reverse strip order so row-0 of the image = north (lat_max)
            # Crop strips to uniform width for the same reason.
            min_w = min(s.shape[1] for s in strips)
            rgb = np.concatenate([s[:, :min_w] for s in strips[::-1]], axis=0)

            buf = io.BytesIO()
            np.save(buf, rgb)
            s3.put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())
            uploaded[str(year)] = key
            log.info("Year %d complete  shape=%s  %.1f MB", year, rgb.shape, buf.tell() / 1e6)

        return {"uploaded": uploaded, "years": sorted(int(y) for y in uploaded)}

    def _synthetic_imagery() -> dict:
        """
        Synthetic RGB rasters for offline / CI runs.
        Simulates a growing urban core ringed by farmland — good enough to
        exercise the full classification + change pipeline without CDSE quota.
        """
        s3 = boto3.client("s3")
        uploaded: dict[str, str] = {}
        rng = np.random.default_rng(42)
        h, w = 512, 512

        for i, year in enumerate(TARGET_YEARS):
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            # Background: agricultural / herbaceous
            rgb[:, :, 0] = rng.integers(40, 90, (h, w), dtype=np.uint8)
            rgb[:, :, 1] = rng.integers(80, 140, (h, w), dtype=np.uint8)
            rgb[:, :, 2] = rng.integers(30, 70, (h, w), dtype=np.uint8)
            # Growing urban core (grey)
            cy, cx = h // 2, w // 2
            radius = 70 + i * 14
            yy, xx = np.ogrid[:h, :w]
            urban = (yy - cy) ** 2 + (xx - cx) ** 2 < radius ** 2
            n = int(urban.sum())
            rgb[urban, 0] = rng.integers(90, 155, n, dtype=np.uint8)
            rgb[urban, 1] = rng.integers(90, 155, n, dtype=np.uint8)
            rgb[urban, 2] = rng.integers(90, 155, n, dtype=np.uint8)

            buf = io.BytesIO()
            np.save(buf, rgb)
            key = f"imagery/{year}/rgb.npy"
            s3.put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())
            uploaded[str(year)] = key
            log.info("Synthetic imagery %d  shape=(%d,%d,3)", year, h, w)

        return {"uploaded": uploaded, "years": TARGET_YEARS, "synthetic": True}

    @task()
    def classify_tiles(fetch_meta: dict) -> dict:
        """
        Tile each year's RGB raster into 64×64 patches and run the EuroSAT
        ResNet-18 classifier (CPU inference inside the Airflow worker).
        Saves per-year:
          - classifications/{year}/grid.npy  — (n_rows, n_cols) uint8 class indices
          - viz/{year}/map.png               — colourised land-use map PNG
        """
        import torch
        import torchvision.models as tvm
        from torchvision import transforms
        from PIL import Image
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        TILE = 64   # px — matches EuroSAT patch size
        BATCH = 64
        MODEL_PATH = os.environ.get("MODEL_PATH", "/app/outputs/resnet18_eurosat.pth")
        device = torch.device("cpu")  # Airflow worker has no GPU; ~3 min/year on CPU

        model = tvm.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 10)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device).eval()

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.344, 0.380, 0.408], std=[0.177, 0.150, 0.142]),
        ])

        color_arr = np.array([
            [int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)]
            for c in CLASS_COLORS
        ], dtype=np.uint8)

        s3 = boto3.client("s3")
        year_stats: list[dict] = []

        for year in sorted(fetch_meta["years"]):
            obj = s3.get_object(Bucket=S3_BUCKET, Key=f"imagery/{year}/rgb.npy")
            rgb = np.load(io.BytesIO(obj["Body"].read()))  # (H, W, 3)
            H, W = rgb.shape[:2]

            n_rows, n_cols = H // TILE, W // TILE
            rgb_crop = rgb[:n_rows * TILE, :n_cols * TILE]

            # Reshape to (n_tiles, TILE, TILE, 3)
            tiles = (rgb_crop
                     .reshape(n_rows, TILE, n_cols, TILE, 3)
                     .transpose(0, 2, 1, 3, 4)
                     .reshape(-1, TILE, TILE, 3))

            preds: list[int] = []
            for start in range(0, len(tiles), BATCH):
                batch = [
                    Image.fromarray(tiles[j])
                    for j in range(start, min(start + BATCH, len(tiles)))
                ]
                t = torch.stack([preprocess(img) for img in batch]).to(device)
                with torch.no_grad():
                    preds.extend(model(t).argmax(1).cpu().tolist())
                if (start // BATCH) % 50 == 0:
                    log.info("Year %d: classified %d/%d tiles", year, start + BATCH, len(tiles))

            grid = np.array(preds, dtype=np.uint8).reshape(n_rows, n_cols)

            grid_buf = io.BytesIO()
            np.save(grid_buf, grid)
            s3.put_object(Bucket=S3_BUCKET, Key=f"classifications/{year}/grid.npy",
                          Body=grid_buf.getvalue())

            # Colorised classification map
            color_map = color_arr[grid]  # (n_rows, n_cols, 3)
            fig, ax = plt.subplots(figsize=(10, 15), dpi=100)
            ax.imshow(color_map, interpolation="nearest")
            ax.set_title(f"Chicagoland Land Use — {year}", fontsize=14, pad=10)
            ax.axis("off")
            patches = [
                mpatches.Patch(facecolor=np.array(color_arr[i]) / 255, label=CLASS_NAMES[i])
                for i in range(10)
            ]
            ax.legend(handles=patches, loc="lower right", fontsize=7,
                      framealpha=0.85, ncol=2)
            png_buf = io.BytesIO()
            plt.savefig(png_buf, format="png", bbox_inches="tight", dpi=100)
            plt.close(fig)
            s3.put_object(
                Bucket=S3_BUCKET, Key=f"viz/{year}/map.png",
                Body=png_buf.getvalue(), ContentType="image/png",
            )

            counts = np.bincount(grid.ravel(), minlength=10)
            total = int(counts.sum())
            class_pct = {CLASS_NAMES[i]: round(float(counts[i] / total * 100), 2)
                         for i in range(10)}
            year_stats.append({"year": year, "total_tiles": total, "classes": class_pct})
            log.info("Year %d done  tiles=%d  %s", year, total, class_pct)

        return {"year_stats": year_stats}

    @task()
    def detect_changes(classify_meta: dict) -> dict:
        """
        Diff consecutive year grids to find tiles that changed class.
        Saves a change-map PNG per pair: orange = changed, dark = unchanged.
        Also records the full transition matrix (from_class → to_class counts).
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        s3 = boto3.client("s3")
        years = sorted(s["year"] for s in classify_meta["year_stats"])
        changes: list[dict] = []

        def load_grid(yr: int) -> np.ndarray:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=f"classifications/{yr}/grid.npy")
            return np.load(io.BytesIO(obj["Body"].read()))

        for ya, yb in zip(years[:-1], years[1:]):
            ga, gb = load_grid(ya), load_grid(yb)
            rows = min(ga.shape[0], gb.shape[0])
            cols = min(ga.shape[1], gb.shape[1])
            ga, gb = ga[:rows, :cols], gb[:rows, :cols]

            changed = ga != gb
            change_pct = float(changed.mean() * 100)
            n_changed = int(changed.sum())

            transitions: dict[str, dict[str, int]] = {}
            for fc, tc in zip(ga[changed].tolist(), gb[changed].tolist()):
                fname, tname = CLASS_NAMES[fc], CLASS_NAMES[tc]
                transitions.setdefault(fname, {})
                transitions[fname][tname] = transitions[fname].get(tname, 0) + 1

            rgba = np.full((*changed.shape, 4), [15, 17, 26, 255], dtype=np.uint8)
            rgba[changed] = [255, 80, 0, 255]

            fig, ax = plt.subplots(figsize=(10, 15), dpi=100)
            ax.imshow(rgba, interpolation="nearest")
            ax.set_title(
                f"Land Use Change  {ya} → {yb}\n"
                f"{change_pct:.1f}% of tiles changed  ({n_changed:,} tiles)",
                fontsize=12, pad=10,
            )
            ax.axis("off")
            png_buf = io.BytesIO()
            plt.savefig(png_buf, format="png", bbox_inches="tight", dpi=100)
            plt.close(fig)
            s3.put_object(
                Bucket=S3_BUCKET, Key=f"viz/{ya}_{yb}/change.png",
                Body=png_buf.getvalue(), ContentType="image/png",
            )

            changes.append({
                "from_year": ya, "to_year": yb,
                "changed_tiles": n_changed,
                "change_pct": round(change_pct, 2),
                "transitions": transitions,
            })
            log.info("Change %d→%d: %.1f%% tiles changed", ya, yb, change_pct)

        return {"changes": changes}

    @task()
    def register_catalog(classify_meta: dict, change_meta: dict) -> dict:
        """Write the full time-series catalog to S3 for the backend to serve."""
        s3 = boto3.client("s3")
        catalog = {
            "aoi": {"bbox": list(AOI_BBOX), "name": "Chicagoland"},
            "class_names":  CLASS_NAMES,
            "class_colors": CLASS_COLORS,
            "timeseries":   classify_meta["year_stats"],
            "changes":      change_meta["changes"],
        }
        payload = json.dumps(catalog, indent=2).encode()
        s3.put_object(
            Bucket=S3_BUCKET, Key="catalog/latest.json",
            Body=payload, ContentType="application/json",
        )
        log.info("Catalog registered: %d years, %d change pairs",
                 len(classify_meta["year_stats"]), len(change_meta["changes"]))
        return {"status": "success"}

    fetched    = fetch_imagery()
    classified = classify_tiles(fetched)
    changed    = detect_changes(classified)
    register_catalog(classified, changed)
