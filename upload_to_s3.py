import boto3
import os

s3 = boto3.client("s3")
bucket_name = "sagemaker-us-east-2-442042534599"

# Get absolute path
script_path = os.path.abspath("ANA680_Module3.py")

if os.path.exists(script_path):
    s3.upload_file(script_path, bucket_name, "ANA680_Module3.py")
    print("Script uploaded successfully!")
else:
    print(f"Error: File not found at {script_path}")


