import tarfile
import os
import boto3

# === Config ===
model_file = "resnet18_eurosat.pth"   # your trained model
inference_script = "inference.py"     # your custom inference logic
tar_file = "model.tar.gz"
bucket = "myawsbucket-f47ac10b-58cc-4372-a567-0e02b2c3d479"  # your S3 bucket
s3_prefix = "models"

# === Step 1: Create tar.gz containing model + inference script ===
with tarfile.open(tar_file, "w:gz") as tar:
    tar.add(model_file, arcname=os.path.basename(model_file))
    tar.add(inference_script, arcname=os.path.basename(inference_script))

print(f"Created {tar_file} with {model_file} + {inference_script}")

# === Step 2: Upload to S3 ===
s3 = boto3.client('s3')
s3_key = f"{s3_prefix}/{tar_file}"
s3.upload_file(tar_file, bucket, s3_key)

print(f"Uploaded {tar_file} to s3://{bucket}/{s3_key}")

# === Step 3: Use in SageMaker ===
from sagemaker.pytorch import PyTorchModel
import sagemaker

role = sagemaker.get_execution_role()
model_data_s3 = f"s3://{bucket}/{s3_key}"

pytorch_model = PyTorchModel(
    model_data=model_data_s3,
    role=role,
    entry_point=inference_script,  # SageMaker will use this for inference
    framework_version="2.1",
    py_version="py310"
)

predictor = pytorch_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name="resnet18-eurosat-endpoint"
)

print("Deployment complete! Endpoint is live.")
