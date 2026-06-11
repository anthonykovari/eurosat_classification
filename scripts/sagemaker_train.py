"""
Submit a SageMaker Training Job for EuroSAT ResNet-18.

Reads the latest processed dataset version from the S3 data catalog,
configures a PyTorch estimator pointing at scripts/train.py, and submits
the job. The training script logs metrics + registers the model in MLflow.

Required env vars:
    EUROSAT_DATA_LAKE_BUCKET      — S3 bucket produced by Terraform s3.tf
    EUROSAT_MODEL_REGISTRY_BUCKET — S3 bucket for model artifacts
    SAGEMAKER_ROLE_ARN            — IAM role produced by Terraform iam.tf
    MLFLOW_TRACKING_URI           — MLflow server URL (optional)

Usage:
    python scripts/sagemaker_train.py
    python scripts/sagemaker_train.py --epochs 10 --instance ml.p3.2xlarge
"""

import argparse
import json
import os

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

DATA_LAKE_BUCKET = os.environ["EUROSAT_DATA_LAKE_BUCKET"]
MODEL_REGISTRY_BUCKET = os.environ["EUROSAT_MODEL_REGISTRY_BUCKET"]
SAGEMAKER_ROLE_ARN = os.environ["SAGEMAKER_ROLE_ARN"]
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")


def get_latest_catalog() -> dict:
    s3 = boto3.client("s3", region_name=AWS_REGION)
    obj = s3.get_object(Bucket=DATA_LAKE_BUCKET, Key="catalog/datasets/latest.json")
    return json.loads(obj["Body"].read())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--instance", default="ml.p3.2xlarge", help="SageMaker instance type")
    parser.add_argument("--wait", action="store_true", help="Block until the job completes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    catalog = get_latest_catalog()
    run_date = catalog["run_date"]
    processed_prefix = catalog["processed_prefix"]
    manifests_prefix = f"data/manifests/{run_date}"

    print(f"Dataset version : {run_date}")
    print(f"Training data   : s3://{DATA_LAKE_BUCKET}/{processed_prefix}/images/")
    print(f"Manifests       : s3://{DATA_LAKE_BUCKET}/{manifests_prefix}/")

    session = sagemaker.Session(
        boto_session=boto3.Session(region_name=AWS_REGION),
        default_bucket=MODEL_REGISTRY_BUCKET,
    )

    estimator = PyTorch(
        entry_point="train.py",
        source_dir=str((os.path.dirname(__file__) or ".")),  # packages scripts/ dir
        role=SAGEMAKER_ROLE_ARN,
        framework_version="2.1",
        py_version="py310",
        instance_type=args.instance,
        instance_count=1,
        hyperparameters={
            "epochs": args.epochs,
            "batch-size": args.batch_size,
            "lr": args.lr,
        },
        environment={
            "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
            "MLFLOW_EXPERIMENT_NAME": "eurosat-resnet18",
        },
        output_path=f"s3://{MODEL_REGISTRY_BUCKET}/training-jobs/",
        sagemaker_session=session,
    )

    estimator.fit(
        inputs={
            "training": f"s3://{DATA_LAKE_BUCKET}/{processed_prefix}/images/",
            "manifests": f"s3://{DATA_LAKE_BUCKET}/{manifests_prefix}/",
        },
        job_name=f"eurosat-resnet18-{run_date}",
        wait=args.wait,
        logs=args.wait,
    )

    job_name = estimator.latest_training_job.name
    print(f"Training job submitted: {job_name}")
    if not args.wait:
        print("Job running in background — monitor at: "
              f"https://{AWS_REGION}.console.aws.amazon.com/sagemaker/home?region={AWS_REGION}#/jobs/{job_name}")


if __name__ == "__main__":
    main()
