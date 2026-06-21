"""
Seed LocalStack with the S3 bucket needed by the Chicago land-use pipeline.

What this does:
  1. Creates the chicago-land-use bucket
  2. Optionally uploads a minimal synthetic catalog so the backend /timeseries
     endpoint returns something without running the full Airflow DAG first

Run once before triggering the Airflow DAG:
  python3 scripts/localstack_seed.py

After seeding, trigger the DAG to populate real classification data:
  make etl-trigger
"""

import argparse
import json
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

BUCKET = "chicago-land-use"

CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]
CLASS_COLORS = [
    "#e6b800", "#1a7a1a", "#66cc66", "#888888", "#cc4400",
    "#99cc44", "#ff9933", "#4488cc", "#3399ff", "#0044aa",
]


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


def seed_placeholder_catalog(s3) -> None:
    """
    Write a minimal catalog so the backend has something to serve before
    the first full ETL run completes.
    """
    import random
    random.seed(42)
    timeseries = []
    for year in range(2019, 2025):
        classes = {name: round(random.uniform(2, 30), 2) for name in CLASS_NAMES}
        total = sum(classes.values())
        classes = {k: round(v / total * 100, 2) for k, v in classes.items()}
        timeseries.append({"year": year, "total_tiles": 26100, "classes": classes})

    catalog = {
        "aoi": {"bbox": [-88.4, 41.45, -87.5, 42.75], "name": "Chicagoland"},
        "class_names": CLASS_NAMES,
        "class_colors": CLASS_COLORS,
        "timeseries": timeseries,
        "changes": [],
        "_note": "placeholder — run the chicago_land_use_pipeline DAG for real data",
    }
    s3.put_object(
        Bucket=BUCKET,
        Key="catalog/latest.json",
        Body=json.dumps(catalog, indent=2).encode(),
        ContentType="application/json",
    )
    print(f"  wrote placeholder catalog → s3://{BUCKET}/catalog/latest.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint", default="http://localhost:4566",
        help="LocalStack S3 endpoint (default: http://localhost:4566)",
    )
    parser.add_argument(
        "--no-catalog", action="store_true",
        help="Skip writing the placeholder catalog",
    )
    args = parser.parse_args()

    s3 = make_client(args.endpoint)

    print(f"LocalStack endpoint: {args.endpoint}\n")
    print("Creating bucket:")
    create_bucket(s3, BUCKET)

    if not args.no_catalog:
        print("\nSeeding placeholder catalog:")
        seed_placeholder_catalog(s3)

    print(f"""
Done.  Trigger the Airflow DAG to run the real pipeline:

  make etl-trigger
  # or in the Airflow UI: http://localhost:8080
  # DAG: chicago_land_use_pipeline

Set AIRFLOW_VAR_SPRAWL_SKIP_CDSE=false and supply CDSE credentials
to download live Sentinel-2 imagery instead of synthetic data.
""")


if __name__ == "__main__":
    main()
