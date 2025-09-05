# deploy.py

import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
import os

# -----------------------------
# Config
# -----------------------------
AWS_REGION = "us-east-2"   # change if needed
S3_BUCKET = "myawsbucket-f47ac10b-58cc-4372-a567-0e02b2c3d479"  # replace with your bucket
MODEL_FILE = "outputs/resnet18_eurosat.pth"
MODEL_NAME = "eurosat-resnet18"
ROLE_ARN = "arn:aws:iam::081153154801:role/service-role/AmazonSageMaker-ExecutionRole-20250816T113525"  # replace

# -----------------------------
# Upload model to S3
# -----------------------------
s3_client = boto3.client("s3", region_name=AWS_REGION, verify=False)
model_s3_key = f"models/{os.path.basename(MODEL_FILE)}"
s3_client.upload_file(MODEL_FILE, S3_BUCKET, model_s3_key)
s3_model_uri = f"s3://{S3_BUCKET}/{model_s3_key}"
print(f"Uploaded model to {s3_model_uri}")

# -----------------------------
# Create SageMaker PyTorch model
# -----------------------------
sm_session = sagemaker.Session(boto_session=boto3.Session(region_name=AWS_REGION))
pytorch_model = PyTorchModel(
    entry_point="backend/inference.py",  # your SageMaker-compatible inference.py
    role=ROLE_ARN,
    model_data=s3_model_uri,
    framework_version="2.1.0",           # PyTorch version
    py_version="py310",
    source_dir=".",                      # ensures backend/inference.py is found
    sagemaker_session=sm_session
)

# -----------------------------
# Deploy endpoint
# -----------------------------
predictor = pytorch_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",  # choose instance type
    endpoint_name=MODEL_NAME
)

print(f"SageMaker endpoint '{MODEL_NAME}' deployed successfully.")
print("You can now invoke it with: predictor.predict(image_bytes)")
