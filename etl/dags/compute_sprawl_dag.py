"""
Compute urban sprawl statistics across all classified years.

Reads classifications/{year}/grid.npy for every year available in S3,
verifies all years were produced by the same model version, then writes
catalog/sprawl_stats.json with per-class area (km²) and year-over-year deltas.

Trigger manually AFTER all years have been (re-)classified with the same model:
  airflow dags trigger chicago_compute_sprawl

Why a separate DAG: if this ran inside chicago_classify_year, a single-year
reclassify would mix grids from different model versions, making the trend
line measure model improvement rather than real land-use change.
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

# Each ResNet-18 tile = 64 px × 10 m/px = 640 m per side.
CELL_KM2 = (64 * 10 / 1000) ** 2  # 0.4096 km²

with DAG(
    dag_id="chicago_compute_sprawl",
    description="Verify model-version consistency across all years, then publish sprawl stats.",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    default_args={"owner": "ml-team", "retries": 0, "retry_delay": timedelta(minutes=2)},
    tags=["sprawl", "stats", "chicago"],
) as dag:

    @task()
    def check_consistency() -> dict:
        """
        Read meta.json sidecars for every classified year in S3.
        Fails hard if any two years used different model versions — the
        sprawl delta would otherwise measure model drift, not land-use change.

        Falls back to querying MLflow for years that predate the sidecar
        (classified before this change was deployed).
        """
        s3 = boto3.client("s3")

        # Discover all years that have a grid.
        paginator = s3.get_paginator("list_objects_v2")
        years = []
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix="classifications/"):
            for obj in page.get("Contents", []):
                parts = obj["Key"].split("/")
                if len(parts) == 3 and parts[2] == "grid.npy":
                    try:
                        years.append(int(parts[1]))
                    except ValueError:
                        pass
        years = sorted(set(years))
        if not years:
            raise RuntimeError("No classification grids found in S3 — run chicago_classify_year first.")

        log.info("Found classified years: %s", years)

        versions: dict[int, str] = {}
        for year in years:
            # Primary: read the meta.json sidecar written by classify_year_dag.
            try:
                obj = s3.get_object(Bucket=S3_BUCKET, Key=f"classifications/{year}/meta.json")
                meta = json.loads(obj["Body"].read())
                versions[year] = meta.get("model_version", "unknown")
                log.info("Year %d: model_version=%s (from meta.json)", year, versions[year])
                continue
            except s3.exceptions.NoSuchKey:
                pass
            except Exception as exc:
                log.warning("Year %d: could not read meta.json (%s) — trying MLflow", year, exc)

            # Fallback: query MLflow for the latest classify run for this year.
            try:
                import mlflow
                mlflow.set_tracking_uri(MLFLOW_URI)
                client = mlflow.tracking.MlflowClient()
                experiment = client.get_experiment_by_name("chicago-classify")
                if experiment:
                    runs = client.search_runs(
                        experiment_ids=[experiment.experiment_id],
                        filter_string=f"params.year = '{year}'",
                        order_by=["start_time DESC"],
                        max_results=1,
                    )
                    if runs:
                        versions[year] = runs[0].data.params.get("model_version", "unknown")
                        log.info("Year %d: model_version=%s (from MLflow)", year, versions[year])
                        continue
            except Exception as exc:
                log.warning("Year %d: MLflow fallback failed (%s)", year, exc)

            versions[year] = "unknown"
            log.warning("Year %d: model_version unknown — no meta.json and MLflow unavailable", year)

        # Check consistency.
        unique_versions = set(v for v in versions.values() if v != "unknown")
        if len(unique_versions) > 1:
            lines = []
            for year in years:
                v = versions[year]
                marker = "  ← MISMATCH" if v != max(unique_versions, key=lambda x: x) else ""
                lines.append(f"  Year {year}: {v}{marker}")
            raise RuntimeError(
                "Model version mismatch across years — sprawl stats would measure model drift, "
                "not land-use change. Re-run chicago_classify_year for all mismatched years "
                "before triggering this DAG.\n" + "\n".join(lines)
            )

        model_version = next(iter(unique_versions), "unknown")
        log.info("Consistency check passed — all years on model_version=%s", model_version)
        return {"years": years, "model_version": model_version, "versions_by_year": versions}

    @task()
    def compute_stats(consistency: dict) -> None:
        """
        Load all classification grids, compute per-class area in km² for each year,
        and write catalog/sprawl_stats.json. The model_version field in the output
        creates an audit trail: if numbers change after a model upgrade you can
        distinguish real land-use change from improved model accuracy.
        """
        from datetime import datetime, timezone

        s3 = boto3.client("s3")
        years: list[int] = consistency["years"]
        model_version: str = consistency["model_version"]

        grids: dict[int, np.ndarray] = {}
        for year in years:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=f"classifications/{year}/grid.npy")
            grids[year] = np.load(io.BytesIO(obj["Body"].read()))
            log.info("Loaded classifications/%d/grid.npy  shape=%s", year, grids[year].shape)

        areas_km2: dict[str, list[float]] = {}
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            areas_km2[cls_name] = [
                round(float((grids[y] == cls_idx).sum()) * CELL_KM2, 2)
                for y in years
            ]

        deltas_km2: dict[str, float] = {}
        deltas_pct: dict[str, float] = {}
        for cls_name in CLASS_NAMES:
            vals = areas_km2[cls_name]
            delta = round(vals[-1] - vals[0], 2)
            deltas_km2[cls_name] = delta
            deltas_pct[cls_name] = round(delta / vals[0] * 100, 1) if vals[0] > 0 else None

        stats = {
            "model_version": model_version,
            "cell_km2": CELL_KM2,
            "classes": CLASS_NAMES,
            "colors": CLASS_COLORS,
            "years": years,
            "first_year": years[0],
            "last_year": years[-1],
            "areas_km2": areas_km2,
            "deltas_km2": deltas_km2,
            "deltas_pct": deltas_pct,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        payload = json.dumps(stats, indent=2).encode()
        s3.put_object(
            Bucket=S3_BUCKET,
            Key="catalog/sprawl_stats.json",
            Body=payload,
            ContentType="application/json",
        )

        log.info(
            "Wrote catalog/sprawl_stats.json  model=%s  years=%s  "
            "Residential Δ=%.1f km² (%.1f%%)  Forest Δ=%.1f km²",
            model_version,
            years,
            deltas_km2.get("Residential", 0),
            deltas_pct.get("Residential") or 0,
            deltas_km2.get("Forest", 0),
        )

    compute_stats(check_consistency())
