"""
EuroSAT ETL Pipeline

Extracts raw satellite imagery, validates class distribution, applies CLAHE
preprocessing, generates stratified splits, and registers the processed
dataset in S3 for consumption by training jobs.
"""

from __future__ import annotations

import json
import logging
from datetime import timedelta

import os

from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable
from airflow.utils.dates import days_ago

log = logging.getLogger(__name__)

S3_BUCKET = Variable.get("eurosat_s3_bucket", default_var="eurosat-data-lake")
RAW_PREFIX = "data/raw"
PROCESSED_PREFIX = "data/processed"
MANIFEST_PREFIX = "data/manifests"

EUROSAT_CLASSES = {
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake",
}
EUROSAT_RGB_URL = "https://madm.dfki.de/files/sentinel/EuroSAT.zip"

default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="eurosat_etl_pipeline",
    default_args=default_args,
    description="EuroSAT ETL: extract → validate → transform → load to S3",
    schedule_interval="@weekly",
    start_date=days_ago(1),
    catchup=False,
    tags=["etl", "eurosat", "computer-vision"],
) as dag:

    @task()
    def extract(run_date: str) -> dict:
        """Download EuroSAT RGB dataset and stage raw images to S3."""
        import boto3
        import tempfile
        import zipfile
        import urllib.request
        from pathlib import Path

        s3 = boto3.client("s3")
        raw_prefix = f"{RAW_PREFIX}/{run_date}"

        existing = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=raw_prefix, MaxKeys=1)
        if existing.get("KeyCount", 0) > 0:
            log.info("Raw data already at s3://%s/%s — skipping download", S3_BUCKET, raw_prefix)
            return {"s3_raw_prefix": raw_prefix, "skipped": True}

        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "EuroSAT.zip"
            log.info("Downloading EuroSAT RGB from %s", EUROSAT_RGB_URL)
            urllib.request.urlretrieve(EUROSAT_RGB_URL, zip_path)

            extract_dir = Path(tmp) / "extracted"
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)

            uploaded = 0
            for img_path in extract_dir.rglob("*.jpg"):
                class_name = img_path.parent.name
                s3_key = f"{raw_prefix}/{class_name}/{img_path.name}"
                s3.upload_file(str(img_path), S3_BUCKET, s3_key)
                uploaded += 1
                if uploaded % 1000 == 0:
                    log.info("Uploaded %d images...", uploaded)

        log.info("Uploaded %d images to s3://%s/%s", uploaded, S3_BUCKET, raw_prefix)
        return {"s3_raw_prefix": raw_prefix, "image_count": uploaded, "skipped": False}

    @task()
    def validate(extract_meta: dict) -> dict:
        """Assert class completeness and minimum per-class image count."""
        import boto3
        from collections import defaultdict

        s3 = boto3.client("s3")
        raw_prefix = extract_meta["s3_raw_prefix"]
        MIN_PER_CLASS = 1800  # EuroSAT ships ~2000-3000 per class; 10% tolerance

        class_counts: dict[str, int] = defaultdict(int)
        paginator = s3.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=raw_prefix):
            for obj in page.get("Contents", []):
                parts = obj["Key"].split("/")
                if len(parts) >= 2:
                    cls = parts[-2]
                    if cls in EUROSAT_CLASSES:
                        class_counts[cls] += 1

        missing = EUROSAT_CLASSES - set(class_counts.keys())
        if missing:
            raise ValueError(f"Missing classes in raw data: {missing}")

        underpopulated = {c: n for c, n in class_counts.items() if n < MIN_PER_CLASS}
        if underpopulated:
            log.warning("Classes below minimum count: %s", underpopulated)

        total = sum(class_counts.values())
        log.info("Validation passed: %d images across %d classes", total, len(class_counts))
        return {**extract_meta, "class_counts": dict(class_counts), "total_images": total}

    @task()
    def transform(validate_meta: dict, run_date: str) -> dict:
        """
        Apply CLAHE, compute per-channel normalization stats, and write
        stratified train/val/test manifests (70/20/10) to S3.
        """
        import boto3
        import csv
        import io
        import tempfile
        import cv2
        import numpy as np
        from pathlib import Path
        from sklearn.model_selection import train_test_split

        s3 = boto3.client("s3")
        raw_prefix = validate_meta["s3_raw_prefix"]
        processed_prefix = f"{PROCESSED_PREFIX}/{run_date}"
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        all_keys: list[str] = []
        all_labels: list[str] = []
        ch_sum = np.zeros(3, dtype=np.float64)
        ch_sq_sum = np.zeros(3, dtype=np.float64)
        n_images = 0

        paginator = s3.get_paginator("list_objects_v2")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=raw_prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if not key.endswith(".jpg"):
                        continue
                    parts = key.split("/")
                    class_name = parts[-2]

                    local_in = tmp_path / "in.jpg"
                    s3.download_file(S3_BUCKET, key, str(local_in))

                    img = cv2.imread(str(local_in))
                    if img is None:
                        log.warning("Skipping unreadable image: %s", key)
                        continue

                    channels = cv2.split(img)
                    processed = cv2.merge([clahe.apply(c) for c in channels])

                    local_out = tmp_path / "out.jpg"
                    cv2.imwrite(str(local_out), processed, [cv2.IMWRITE_JPEG_QUALITY, 95])

                    out_key = f"{processed_prefix}/images/{class_name}/{parts[-1]}"
                    s3.upload_file(str(local_out), S3_BUCKET, out_key)

                    # Accumulate stats (BGR → RGB channel order)
                    img_f = processed[:, :, ::-1].astype(np.float64) / 255.0
                    ch_sum += img_f.mean(axis=(0, 1))
                    ch_sq_sum += (img_f ** 2).mean(axis=(0, 1))
                    n_images += 1

                    all_keys.append(out_key)
                    all_labels.append(class_name)

            mean = (ch_sum / n_images).tolist()
            std = np.sqrt(ch_sq_sum / n_images - (ch_sum / n_images) ** 2).tolist()

            # Stratified split: 70 train / 20 val / 10 test
            train_k, tmp_k, train_l, tmp_l = train_test_split(
                all_keys, all_labels, test_size=0.30, stratify=all_labels, random_state=42
            )
            val_k, test_k, val_l, test_l = train_test_split(
                tmp_k, tmp_l, test_size=0.333, stratify=tmp_l, random_state=42
            )

            splits = {"train": (train_k, train_l), "val": (val_k, val_l), "test": (test_k, test_l)}
            manifest_keys: dict[str, str] = {}

            for split_name, (keys, labels) in splits.items():
                buf = io.StringIO()
                writer = csv.writer(buf)
                writer.writerow(["s3_key", "label"])
                writer.writerows(zip(keys, labels))
                manifest_key = f"{MANIFEST_PREFIX}/{run_date}/{split_name}.csv"
                s3.put_object(Bucket=S3_BUCKET, Key=manifest_key, Body=buf.getvalue().encode())
                manifest_keys[split_name] = manifest_key
                log.info("Wrote %d rows to %s", len(keys), manifest_key)

            stats = {"mean_rgb": mean, "std_rgb": std, "n_images": n_images}
            stats_key = f"{MANIFEST_PREFIX}/{run_date}/normalization_stats.json"
            s3.put_object(Bucket=S3_BUCKET, Key=stats_key, Body=json.dumps(stats, indent=2).encode())

        return {
            "processed_prefix": processed_prefix,
            "manifest_keys": manifest_keys,
            "stats_key": stats_key,
            "normalization_stats": stats,
            "split_sizes": {k: len(v[0]) for k, v in splits.items()},
        }

    @task()
    def load(transform_meta: dict, run_date: str) -> dict:
        """
        Register the processed dataset in the S3 data catalog and
        update the 'latest' pointer for downstream training jobs.
        """
        import boto3

        s3 = boto3.client("s3")

        catalog_entry = {
            "run_date": run_date,
            "processed_prefix": transform_meta["processed_prefix"],
            "manifests": transform_meta["manifest_keys"],
            "normalization_stats": transform_meta["normalization_stats"],
            "split_sizes": transform_meta["split_sizes"],
            "status": "ready",
        }
        payload = json.dumps(catalog_entry, indent=2).encode()

        versioned_key = f"catalog/datasets/{run_date}.json"
        s3.put_object(Bucket=S3_BUCKET, Key=versioned_key, Body=payload, ContentType="application/json")
        s3.put_object(Bucket=S3_BUCKET, Key="catalog/datasets/latest.json", Body=payload, ContentType="application/json")

        log.info(
            "Dataset %s registered — train: %d | val: %d | test: %d",
            run_date,
            transform_meta["split_sizes"].get("train", 0),
            transform_meta["split_sizes"].get("val", 0),
            transform_meta["split_sizes"].get("test", 0),
        )
        return {"catalog_key": versioned_key, "status": "success"}

    @task()
    def trigger_training(load_meta: dict, run_date: str) -> dict:
        """
        Submit a SageMaker Training Job once the new dataset is registered.
        The job runs asynchronously; monitor it in the SageMaker console.
        """
        import boto3
        import sagemaker
        from sagemaker.pytorch import PyTorch

        data_lake_bucket = Variable.get("eurosat_data_lake_bucket")
        model_registry_bucket = Variable.get("eurosat_model_registry_bucket")
        sagemaker_role_arn = Variable.get("eurosat_sagemaker_role_arn")
        mlflow_uri = Variable.get("eurosat_mlflow_tracking_uri", default_var="")
        aws_region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

        processed_prefix = load_meta.get("processed_prefix", f"data/processed/{run_date}")

        session = sagemaker.Session(
            boto_session=boto3.Session(region_name=aws_region),
            default_bucket=model_registry_bucket,
        )

        estimator = PyTorch(
            entry_point="train.py",
            source_dir="/opt/airflow/dags/../../../scripts",  # mounted via docker-compose volume
            role=sagemaker_role_arn,
            framework_version="2.1",
            py_version="py310",
            instance_type="ml.p3.2xlarge",
            instance_count=1,
            hyperparameters={"epochs": 25, "batch-size": 64, "lr": 1e-3},
            environment={
                "MLFLOW_TRACKING_URI": mlflow_uri,
                "MLFLOW_EXPERIMENT_NAME": "eurosat-resnet18",
            },
            output_path=f"s3://{model_registry_bucket}/training-jobs/",
            sagemaker_session=session,
        )

        estimator.fit(
            inputs={
                "training": f"s3://{data_lake_bucket}/{processed_prefix}/images/",
                "manifests": f"s3://{data_lake_bucket}/data/manifests/{run_date}/",
            },
            job_name=f"eurosat-resnet18-{run_date}",
            wait=False,
        )

        job_name = estimator.latest_training_job.name
        log.info("SageMaker training job submitted: %s", job_name)
        return {"job_name": job_name, "status": "submitted"}

    # DAG wiring
    run_date = "{{ ds }}"
    extracted = extract(run_date)
    validated = validate(extracted)
    transformed = transform(validated, run_date)
    loaded = load(transformed, run_date)
    trigger_training(loaded, run_date)
