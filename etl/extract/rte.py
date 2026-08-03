import os
from datetime import datetime, timedelta
from time import sleep

import pandas as pd
import requests
import zoneinfo

from etl.config import DATA_DIR

RTE_DATA_DIR = os.path.join(DATA_DIR, "rte")


# Encapsule les API de consommation RTE : consolidated_power_consumption (historique
# finalisé, plage de dates arbitraire) et short_term (donnée récente, type REALISED).
class RTEExtractor():

    def __init__(self):
        self.extract_file_name = ''
        self.data = []  # first json response key to access data
        self.url_oauth_token = "https://digital.iservices.rte-france.com/token/oauth/"
        self.url_api_power_consumption = "https://digital.iservices.rte-france.com/open_api/consolidated_consumption/v1/consolidated_power_consumption"
        self.url_api_short_term_consumption = "https://digital.iservices.rte-france.com/open_api/consumption/v1/short_term"
        # les URLs doivent être définies avant de générer le token, qui en dépend
        self.generate_oauth_token()

    # génère le token OAuth nécessaire à l'authentification sur l'API RTE
    def generate_oauth_token(self):
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": "Basic " + os.getenv("RTE_API_KEY")
        }
        try:
            response_token = requests.post(self.url_oauth_token, headers=headers)
            if response_token.status_code == 200:
                data = response_token.json()
                self.oauth_token = data["access_token"]
                print("Token generated !")
            else:
                raise Exception(f"Token Error: {response_token.status_code}")
        except Exception as e:
            raise(e)

    # récupère la consommation demi-horaire entre 2 dates (format ISO8601, ex: 2025-12-01T00:00:00+01:00),
    # données finalisées, plage arbitraire — utilisé pour l'historique (ETL 1) et le mode backfill (ETL 2)
    def get_historical_consumption(self, start_date: str, end_date: str):
        headers = {
            "Authorization": "Bearer " + self.oauth_token
        }
        params = {
            "start_date": start_date,
            "end_date": end_date
        }
        try:
            response = requests.get(self.url_api_power_consumption, headers=headers, params=params)
            if response.status_code == 200:
                print(f"Data extracted from {start_date} to {end_date}!")
                return response.json()
            else:
                data_error = response.json()
                raise Exception(f"API Error {response.status_code}  : {data_error['error']} + ' : ' + {data_error['error_description']}")
        except Exception as e:
            raise(e)

    # récupère la consommation récente (type REALISED) des `days` derniers jours via
    # l'endpoint short_term, plus adapté que consolidated_power_consumption pour de la
    # donnée très récente — utilisé pour le mode direct de l'ETL 2
    # retourne directement la liste de valeurs (start_date, end_date, value, ...)
    def get_recent_consumption(self, days: int = 30) -> list:
        end_date = datetime.now(tz=zoneinfo.ZoneInfo("Europe/Paris")).replace(minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=days)

        headers = {
            "Authorization": "Bearer " + self.oauth_token
        }
        params = {
            "type": "REALISED",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        }

        try:
            response = requests.get(self.url_api_short_term_consumption, headers=headers, params=params)
            if response.status_code == 200:
                values = response.json()["short_term"][0]["values"]
                print(f"Recent consumption extracted: {len(values)} records ({start_date.isoformat()} -> {end_date.isoformat()})")
                return values
            else:
                data_error = response.json()
                raise Exception(f"API Error {response.status_code} : {data_error.get('error')} : {data_error.get('error_description')}")
        except Exception as e:
            raise(e)

    # récupère une année complète en 2 requêtes (l'API limite la plage par appel),
    # avec un léger chevauchement autour du 30 juin/1er juillet entre les 2 appels
    def get_annual_power_consumption(self, year: int):
        # first 6 months
        start_date = f"{year-1}-12-31T00:00:00Z"
        end_date = f"{year}-07-01T00:00:00Z"
        data1 = self.get_historical_consumption(start_date, end_date)
        data1_values = data1["consolidated_power_consumption"][0]['values']
        print(f"First half data extracted: {len(data1_values)} records")

        sleep(1)

        # second 6 months
        start_date = f"{year}-06-30T00:00:00Z"
        end_date = f"{year+1}-01-01T00:00:00Z"
        data2 = self.get_historical_consumption(start_date, end_date)
        data2_values = data2["consolidated_power_consumption"][0]['values']
        print(f"Second half data extracted: {len(data2_values)} records")

        self.extract_file_name = f"{year}_power_consumption"
        self.data = data1_values + data2_values

    # sauvegarde self.data (trié par instant chronologique réel) au format demandé, dans data/rte/
    def save_data_file(self, format='csv'):
        df = pd.DataFrame(self.data)
        # tri par instant chronologique réel (pas par chaîne de caractères) :
        # autour d'un changement d'heure, "...T02:00:00+01:00" est alphabétiquement
        # avant "...T02:30:00+02:00" bien que ce ne soit pas toujours l'ordre réel
        df.sort_values(by='end_date', key=lambda col: pd.to_datetime(col, utc=True), inplace=True)
        df.reset_index(inplace=True, drop=True)
        if format == 'csv':
            df.to_csv(self.get_data_storage_path('csv'), index=False)
        elif format == 'json':
            df.to_json(self.get_data_storage_path('json'), index=False)
        else:
            raise ValueError(f"Format {format} not supported")

    # construit le chemin du fichier de sortie dans data/rte/, indépendamment
    # du répertoire depuis lequel le script est exécuté
    def get_data_storage_path(self, file_extension: str):
        return os.path.join(RTE_DATA_DIR, self.extract_file_name + "." + file_extension)

    # convertit une date simple (YYYY-MM-DD) en ISO8601 avec le fuseau Europe/Paris
    def format_date(self, date: str, input_format="%Y-%m-%d"):
        dt = datetime.strptime(date, "%Y-%m-%d")
        tz_paris = zoneinfo.ZoneInfo("Europe/Paris")
        dt_tz = dt.replace(hour=0, minute=0, second=0, tzinfo=tz_paris)
        return dt_tz.strftime("%Y-%m-%dT%H:%M:%S%z")
