import os

import boto3
import pandas as pd


# Sauvegarde un dataset en local puis l'envoie sur S3. Générique : ni chemin ni
# préfixe fixés, réutilisable par les 2 pipelines.
class S3Loader:

    def upload(self, df: pd.DataFrame, local_path: str, bucket: str, s3_key: str):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        df.to_csv(local_path, index=False)
        print(f"Saved {len(df)} rows to {local_path}")

        if not bucket:
            raise Exception("Bucket S3 non fourni, impossible d'envoyer sur S3")

        boto3.client("s3").upload_file(local_path, bucket, s3_key)
        print(f"Uploaded to s3://{bucket}/{s3_key}")
