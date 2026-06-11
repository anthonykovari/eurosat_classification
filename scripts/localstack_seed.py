"""
Seed LocalStack with the S3 buckets and data needed to run the local stack.

What this does:
  1. Creates the data lake and model registry buckets
  2. Uploads a small subset of local EuroSAT images as the raw dataset
     (avoids downloading 27k images during the Airflow extract task)
  3. Uploads the trained model weights to the model registry

Run once before triggering the Airflow DAG:
  python scripts/localstack_seed.py

The Airflow extract task checks S3 first and skips the internet download
when it finds data already present — that's what makes this work.
"""

import argparse
import json
from datetime import date
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parent.parent
LOCAL_DATA_DIR = ROOT / "data" / "eurosat" / "2750"
LOCAL_MODEL_PATH = ROOT / "outputs" / "resnet18_eurosat.pth"
DATA_LAKE_BUCKET = "eurosat-data-lake"
MODEL_REGISTRY_BUCKET = "eurosat-model-registry"


def make_client(endpoint: str):
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id="test",
        aws_secret_access_key="test",
        region_name="us-east-1",
    )


def create_bucket(s3, name: str) -> None:
    try:
        s3.create_bucket(Bucket=name)
        print(f"  created  s3://{name}")
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            print(f"  exists   s3://{name}")
        else:
            raise


def seed_raw_images(s3, run_date: str, per_class: int) -> int:
    if not LOCAL_DATA_DIR.exists():
        print(f"\nWARNING: {LOCAL_DATA_DIR} not found.")
        print("Run 'python scripts/download_data.py' first, then re-run this script.")
        return 0

    total = 0
    for cls_dir in sorted(LOCAL_DATA_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue
        images = sorted(cls_dir.glob("*.jpg"))[:per_class]
        for img in images:
            key = f"data/raw/{run_date}/{cls_dir.name}/{img.name}"
            s3.upload_file(str(img), DATA_LAKE_BUCKET, key)
            total += 1
        print(f"  {cls_dir.name:<25} {len(images)} images")

    return total


def seed_model(s3) -> None:
    if not LOCAL_MODEL_PATH.exists():
        print(f"  WARNING: {LOCAL_MODEL_PATH} not found — skipping model seed")
        return
    key = "models/resnet18_eurosat.pth"
    s3.upload_file(str(LOCAL_MODEL_PATH), MODEL_REGISTRY_BUCKET, key)
    print(f"  uploaded s3://{MODEL_REGISTRY_BUCKET}/{key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:4566",
                        help="LocalStack S3 endpoint (default: http://localhost:4566)")
    parser.add_argument("--images-per-class", type=int, default=25,
                        help="Images per class to seed (default: 25, × 10 classes = 250 total)")
    args = parser.parse_args()

    s3 = make_client(args.endpoint)
    run_date = str(date.today())

    print(f"LocalStack endpoint : {args.endpoint}")
    print(f"Run date            : {run_date}")
    print(f"Images per class    : {args.images_per_class}\n")

    print("Creating buckets:")
    create_bucket(s3, DATA_LAKE_BUCKET)
    create_bucket(s3, MODEL_REGISTRY_BUCKET)

    print(f"\nSeeding raw images → s3://{DATA_LAKE_BUCKET}/data/raw/{run_date}/")
    count = seed_raw_images(s3, run_date, args.images_per_class)
    print(f"  total: {count} images")

    print(f"\nSeeding model weights → s3://{MODEL_REGISTRY_BUCKET}/")
    seed_model(s3)

    print(f"""
Done. Trigger the Airflow DAG for date {run_date}:

  make etl-trigger
  # or in the Airflow UI: http://localhost:8080
  # DAG: eurosat_etl_pipeline  |  logical date: {run_date}

The extract task will find data already in S3 and skip the internet download.
The trigger_training task will skip automatically (eurosat_skip_sagemaker=true).
""")


if __name__ == "__main__":
    main()
