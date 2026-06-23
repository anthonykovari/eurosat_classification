"""
Classify one year of Sentinel-2 imagery already in S3.

Reads imagery/{year}/rgb.npy, runs EuroSAT ResNet-18 in 64×64 windows,
writes classifications/{year}/grid.npy.  No CDSE calls — safe to retry freely.

Trigger:
  airflow dags trigger chicago_classify_year --conf '{"year": 2019}'
"""

from __future__ import annotations

import io
import json
import logging
import os
import time

import boto3
import numpy as np
from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable
from airflow.utils.dates import days_ago
from botocore.exceptions import ClientError
from datetime import timedelta

log = logging.getLogger(__name__)

S3_BUCKET = Variable.get("sprawl_s3_bucket", default_var="chicago-land-use")

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]
CLASS_COLORS = [
    "#e6b800", "#1a7a1a", "#66cc66", "#888888", "#cc4400",
    "#99cc44", "#ff9933", "#d4826a", "#3399ff", "#0044aa",
]

with DAG(
    dag_id="chicago_classify_year",
    description="Classify one year's imagery with ResNet-18 (trigger with conf={'year': YYYY})",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    default_args={"owner": "ml-team", "retries": 2, "retry_delay": timedelta(minutes=2)},
    tags=["classify", "resnet18", "chicago"],
) as dag:

    @task()
    def classify_tiles(**context) -> dict:
        """
        Tile imagery/{year}/rgb.npy into 64×64 patches, run EuroSAT ResNet-18,
        write classifications/{year}/grid.npy.
        Skips if the grid already exists in S3.
        """
        import mlflow
        import mlflow.pytorch
        import torch
        import torchvision.models as tvm
        from torchvision import transforms
        from PIL import Image

        run_conf = context.get("dag_run").conf or {}
        year = int(run_conf.get("year", __import__("datetime").date.today().year))
        force = bool(run_conf.get("force", False))
        log.info("Classifying year %d (force=%s)", year, force)

        s3 = boto3.client("s3")

        # Skip if grid already exists and is readable, unless force=true.
        grid_key = f"classifications/{year}/grid.npy"
        if not force:
            try:
                resp = s3.get_object(Bucket=S3_BUCKET, Key=grid_key, Range="bytes=0-3")
                size = int(resp["ContentRange"].split("/")[-1])
                if size > 0:
                    log.info("Year %d grid already in S3 (%d bytes) — skipping", year, size)
                    return {"year": year, "skipped": True}
            except ClientError:
                pass

        # Load imagery from S3.
        rgb_key = f"imagery/{year}/rgb.npy"
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=rgb_key)
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                raise RuntimeError(
                    f"imagery/{year}/rgb.npy not in S3 — run chicago_fetch_year first"
                ) from e
            raise
        rgb = np.load(io.BytesIO(obj["Body"].read()))  # (H, W, 3) uint8
        H, W = rgb.shape[:2]
        log.info("Loaded imagery/%d/rgb.npy  shape=%s", year, rgb.shape)

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required — classification must not run on CPU")
        device = torch.device("cuda")
        log.info("Using device: %s  (%s)", device, torch.cuda.get_device_name(0))

        # Load ResNet-18 — try MLflow registry first, fall back to file.
        model_source = "registry"
        mlflow.set_tracking_uri(MLFLOW_URI)
        try:
            model = mlflow.pytorch.load_model(
                "models:/resnet18-eurosat/Production", map_location=device
            )
            model.to(device).eval()
            mv = mlflow.tracking.MlflowClient().get_latest_versions(
                "resnet18-eurosat", stages=["Production"]
            )[0]
            model_version = f"v{mv.version}"
            log.info("Loaded resnet18-eurosat %s from MLflow registry", model_version)
        except Exception as exc:
            log.warning("MLflow registry unavailable (%s) — loading from file", exc)
            MODEL_PATH = os.environ.get("MODEL_PATH", "/app/outputs/resnet18_eurosat.pth")
            model = tvm.resnet18(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, 10)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.to(device).eval()
            model_source, model_version = "file", "unknown"

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.344, 0.380, 0.408], std=[0.177, 0.150, 0.142]),
        ])

        TILE = 64   # px — matches EuroSAT patch size
        BATCH = 64

        n_rows = H // TILE
        n_cols = W // TILE
        rgb_crop = rgb[:n_rows * TILE, :n_cols * TILE]

        tiles = (rgb_crop
                 .reshape(n_rows, TILE, n_cols, TILE, 3)
                 .transpose(0, 2, 1, 3, 4)
                 .reshape(-1, TILE, TILE, 3))

        log.info("Year %d: %d×%d grid = %d tiles", year, n_rows, n_cols, len(tiles))

        preds: list[int] = []
        t0 = time.time()
        for start in range(0, len(tiles), BATCH):
            batch_imgs = [
                Image.fromarray(tiles[j])
                for j in range(start, min(start + BATCH, len(tiles)))
            ]
            t = torch.stack([preprocess(img) for img in batch_imgs]).to(device)
            with torch.no_grad():
                preds.extend(model(t).argmax(1).cpu().tolist())
            if (start // BATCH) % 20 == 0:
                log.info("Year %d: %d/%d tiles classified", year, len(preds), len(tiles))
        inference_seconds = round(time.time() - t0, 2)

        grid = np.array(preds, dtype=np.uint8).reshape(n_rows, n_cols)

        buf = io.BytesIO()
        np.save(buf, grid)
        s3.put_object(Bucket=S3_BUCKET, Key=grid_key, Body=buf.getvalue())

        from datetime import datetime, timezone
        meta = {
            "year": year,
            "model_version": model_version,
            "model_source": model_source,
            "grid_shape": [n_rows, n_cols],
            "classified_at": datetime.now(timezone.utc).isoformat(),
        }
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=f"classifications/{year}/meta.json",
            Body=json.dumps(meta).encode(),
            ContentType="application/json",
        )
        log.info("Wrote classifications/%d/meta.json  model_version=%s", year, model_version)

        counts = np.bincount(grid.ravel(), minlength=10)
        total = int(counts.sum())
        class_pct = {CLASS_NAMES[i]: round(float(counts[i] / total * 100), 2) for i in range(10)}
        log.info(
            "Year %d done  grid=%s  inference=%.1fs  %s",
            year, grid.shape, inference_seconds, class_pct,
        )

        mlflow.set_experiment("chicago-classify")
        with mlflow.start_run(run_name=f"classify-{year}"):
            mlflow.log_params({
                "year":          year,
                "model_version": model_version,
                "model_source":  model_source,
                "n_tiles":       len(tiles),
                "grid_shape":    f"{n_rows}x{n_cols}",
            })
            mlflow.log_metric("inference_seconds", inference_seconds)
            for cls, pct in class_pct.items():
                mlflow.log_metric(f"pct_{cls}", pct)

        return {
            "year": year,
            "grid_shape": list(grid.shape),
            "inference_seconds": inference_seconds,
            "classes": class_pct,
            "skipped": False,
        }

    classify_tiles()
