import boto3
from dotenv import load_dotenv
import os

# Load environment variables from .env file
env_path = os.path.join(".env")
load_dotenv(dotenv_path=env_path)
print(env_path)

# Access the variables
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_DEFAULT_REGION")
print(aws_region)

session = boto3.Session(
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=aws_region
)


""" def get_aws_client():
    return boto3.client(
        "polly",
        raws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region
    ) """


def get_aws_client():
    return session.client("polly")
