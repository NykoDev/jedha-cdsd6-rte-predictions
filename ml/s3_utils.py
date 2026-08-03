import os
import pickle
from io import BytesIO

import boto3
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


# Client S3 partagé par les classes de modèle, le script d'ensemble et l'API : centralise la
# convention de credentials/bucket déjà utilisée dans les exemples déposés (dashboard/utils.py,
# ml/model_prophet.py de l'ancien projet), pour ne pas la dupliquer trois fois.
def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name="eu-west-3",
    )


def _bucket() -> str:
    bucket = os.environ.get("AWS_BUCKET")
    if not bucket:
        raise RuntimeError("AWS_BUCKET n'est pas défini dans l'environnement")
    return bucket


# Sauvegarde un modèle (XGBRegressor, Prophet...) en pickle sur S3, sans fichier temporaire local
def upload_pickle(obj, key: str) -> None:
    buffer = BytesIO()
    pickle.dump(obj, buffer)
    buffer.seek(0)
    get_s3_client().put_object(Bucket=_bucket(), Key=key, Body=buffer.getvalue())


def download_pickle(key: str):
    response = get_s3_client().get_object(Bucket=_bucket(), Key=key)
    return pickle.loads(response["Body"].read())


def upload_dataframe_csv(df: pd.DataFrame, key: str) -> None:
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    get_s3_client().put_object(Bucket=_bucket(), Key=key, Body=buffer.getvalue())


# Charge un CSV à une clé S3 fixe
def load_csv(key: str) -> pd.DataFrame:
    obj = get_s3_client().get_object(Bucket=_bucket(), Key=key)
    return pd.read_csv(BytesIO(obj["Body"].read()))


# Fusionne `df` avec le CSV existant à la clé S3 `key` (s'il existe) au lieu de l'écraser :
# les lignes dont `key_column` existe déjà dans le fichier sont remplacées par les nouvelles
# (ex: une prévision rejouée pour une heure déjà prédite), les autres lignes existantes sont
# conservées telles quelles. Comparaison sur les valeurs parsées en datetime UTC (et non les
# chaînes brutes du CSV) pour rester robuste aux différences de format d'affichage.
def upsert_dataframe_csv(df: pd.DataFrame, key: str, key_column: str) -> None:
    client = get_s3_client()
    bucket = _bucket()

    try:
        obj = client.get_object(Bucket=bucket, Key=key)
        existing_df = pd.read_csv(BytesIO(obj["Body"].read()))
    except client.exceptions.NoSuchKey:
        existing_df = None

    if existing_df is not None and key_column in existing_df.columns:
        new_keys = pd.to_datetime(df[key_column], utc=True)
        existing_keys = pd.to_datetime(existing_df[key_column], utc=True)
        existing_df = existing_df[~existing_keys.isin(new_keys)]
        merged_df = pd.concat([existing_df, df], ignore_index=True)
    else:
        merged_df = df.copy()

    sort_order = pd.to_datetime(merged_df[key_column], utc=True).sort_values().index
    merged_df = merged_df.loc[sort_order].reset_index(drop=True)

    upload_dataframe_csv(merged_df, key)
