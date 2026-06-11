"""
Deploy the trained EuroSAT model as a SageMaker real-time inference endpoint.

This script deploys the latest model artifact from S3 to a SageMaker endpoint
using the inference handler in code/inference.py.

Required env vars:
    EUROSAT_MODEL_REGISTRY_BUCKET — S3 bucket for model artifacts (Terraform output)
    SAGEMAKER_ROLE_ARN            — IAM execution role (Terraform output)
    AWS_DEFAULT_REGION            — defaults to us-east-1

Usage:
    python deploy.py
    python deploy.py --model-s3-key models/resnet18_eurosat.pth --instance ml.g4dn.xlarge
"""

import argparse
import os

import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel

MODEL_REGISTRY_BUCKET = os.environ["EUROSAT_MODEL_REGISTRY_BUCKET"]
SAGEMAKER_ROLE_ARN = os.environ["SAGEMAKER_ROLE_ARN"]
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
ENDPOINT_NAME = "eurosat-resnet18"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-s3-key", default="models/resnet18_eurosat.pth")
    parser.add_argument("--instance", default="ml.m5.large")
    parser.add_argument("--update", action="store_true", help="Update an existing endpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_uri = f"s3://{MODEL_REGISTRY_BUCKET}/{args.model_s3_key}"

    session = sagemaker.Session(
        boto_session=boto3.Session(region_name=AWS_REGION),
        default_bucket=MODEL_REGISTRY_BUCKET,
    )

    model = PyTorchModel(
        model_data=model_uri,
        role=SAGEMAKER_ROLE_ARN,
        entry_point="inference.py",
        source_dir="code",
        framework_version="2.1",
        py_version="py310",
        sagemaker_session=session,
    )

    print(f"Deploying {model_uri} → endpoint '{ENDPOINT_NAME}' on {args.instance}")

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=args.instance,
        endpoint_name=ENDPOINT_NAME,
        update_endpoint=args.update,
    )

    print(f"Endpoint '{ENDPOINT_NAME}' is live.")
    print("Test with: predictor.predict(open('image.jpg','rb').read(), initial_args={'ContentType':'image/jpeg'})")


if __name__ == "__main__":
    main()
