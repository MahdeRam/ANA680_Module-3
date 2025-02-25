from sagemaker.sklearn.estimator import SKLearn
import sagemaker

# Define SageMaker session
session = sagemaker.Session()

# Define the SKLearn estimator with the correct S3 paths
sklearn_estimator = SKLearn(
    entry_point="ANA680_Module3.py",  # Ensure the script exists in the source_dir
    source_dir="s3://sagemaker-us-east-2-442042534599/",  # Use the correct directory path
    role="arn:aws:iam::442042534599:role/service-role/AmazonSageMaker-ExecutionRole-20250223T202718",
    instance_count=1,
    instance_type="ml.m5.large",
    framework_version="0.23-1",
    sagemaker_session=session
)

# Start the training job
sklearn_estimator.fit(
    {
        "train": "s3://sagemaker-us-east-2-442042534599/wine_quality.csv"  # Ensure data exists here
    },
    job_name="my-training-job"
)

print("Training job submitted successfully!")


