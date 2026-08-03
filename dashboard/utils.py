import pandas as pd
from dotenv import load_dotenv
import boto3
from io import BytesIO
import os
import datetime

load_dotenv()


# Date considérée comme "aujourd'hui" par le dashboard : CURRENT_DAY (variable d'env, format
# DD/MM/YYYY) si définie, sinon la date réelle. Permet de figer la démo/les tests sur une date où
# l'on sait que les données (historique + prévisions) existent, sans dépendre d'un pipeline de
# données réellement à jour.
def get_current_day():
    current_day_env = os.environ.get("CURRENT_DAY")
    if current_day_env:
        return datetime.datetime.strptime(current_day_env, "%d/%m/%Y").date()
    return datetime.date.today()


def load_dataset_s3(file):
    """
    Load a dataset from S3 using boto3.

    Args:
        file (str): The path to the file in S3. (directory/file.csv)

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        region_name="eu-west-3",
    )

    response = s3_client.get_object(Bucket=os.environ.get('AWS_BUCKET'), Key=file)
    df = pd.read_csv(BytesIO(response["Body"].read()))
    return df