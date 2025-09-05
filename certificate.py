import boto3

s3 = boto3.client("s3")  # configure credentials if needed
s3.download_file("myawsbucket-f47ac10b-58cc-4372-a567-0e02b2c3d479", "models/resnet18_eurosat.pth", "resnet18_eurosat.pth")
