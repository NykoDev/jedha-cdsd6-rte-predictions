import pandas as pd
from dotenv import load_dotenv
import boto3
from io import BytesIO
import os
import datetime
import streamlit as st

load_dotenv()

# Sur Streamlit Community Cloud, les secrets saisis dans l'UI de l'app sont exposés via
# st.secrets mais ne sont pas injectés dans os.environ. On les recopie ici une fois, à l'import
# du module, pour que os.environ.get(...) (utilisé partout dans ce fichier) fonctionne à
# l'identique en local (.env) et sur Cloud (secrets.toml). Pas d'effet en local : st.secrets est
# alors vide (pas de fichier .streamlit/secrets.toml).
try:
    for _key, _value in st.secrets.items():
        os.environ.setdefault(_key, str(_value))
except Exception:
    pass


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