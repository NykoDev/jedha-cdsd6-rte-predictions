from time import sleep, time

import pandas as pd
import requests


# Encapsule l'API historique Open-Meteo (archive-api.open-meteo.com).
# Un seul appel par ville couvre toute la plage de dates demandée (pas besoin
# de découper en sous-périodes).
class OpenMeteoExtractor():

    ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"
    FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"
    MIN_REQUEST_INTERVAL = 1.1  # secondes entre 2 appels — garde-fou simple contre la limite de l'API

    def __init__(self):
        self._last_request_time = None

    # limitation de débit simple à intervalle fixe : attend si le précédent appel est trop récent
    def _wait_for_rate_limit(self):
        if self._last_request_time is not None:
            elapsed = time() - self._last_request_time
            if elapsed < self.MIN_REQUEST_INTERVAL:
                sleep(self.MIN_REQUEST_INTERVAL - elapsed)

    # récupère la température horaire d'une ville entre 2 dates (format YYYY-MM-DD),
    # données finalisées, utilisé pour l'historique (ETL 1) et le mode backfill (ETL 2)
    # retourne un DataFrame avec les colonnes City, Timestamp, Temperature
    def get_hourly_temperature(self, city: str, latitude: float, longitude: float, start_date: str, end_date: str) -> pd.DataFrame:
        self._wait_for_rate_limit()

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m"
            # pas de paramètre timezone -> l'API répond par défaut en GMT
        }

        try:
            response = requests.get(self.ARCHIVE_API_URL, params=params)
            self._last_request_time = time()

            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame({
                    "City": city,
                    "Timestamp": data["hourly"]["time"],
                    "Temperature": data["hourly"]["temperature_2m"]
                })
                print(f"{city}: {len(df)} records extracted ({start_date} -> {end_date})")
                return df
            elif response.status_code == 429:
                # limite de débit atteinte côté API : on attend puis on retente une fois
                print(f"Rate limited for {city}, waiting 60s and retrying...")
                sleep(60)
                return self.get_hourly_temperature(city, latitude, longitude, start_date, end_date)
            else:
                raise Exception(f"API Error {response.status_code}: {response.text}")
        except Exception as e:
            raise(e)

    # récupère l'historique récent (past_days) + la prévision (forecast_days) d'une
    # ville en un seul appel, via l'endpoint forecast (données pas encore finalisées,
    # contrairement à l'endpoint archive) — utilisé pour le mode direct de l'ETL 2
    # retourne un DataFrame avec les colonnes City, Timestamp, Temperature
    def get_recent_and_forecast_temperature(self, city: str, latitude: float, longitude: float, past_days: int = 30, forecast_days: int = 1) -> pd.DataFrame:
        self._wait_for_rate_limit()

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "temperature_2m",
            "past_days": past_days,
            "forecast_days": forecast_days
            # pas de paramètre timezone -> réponse en GMT, cohérent avec le reste du projet
        }

        try:
            response = requests.get(self.FORECAST_API_URL, params=params)
            self._last_request_time = time()

            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame({
                    "City": city,
                    "Timestamp": data["hourly"]["time"],
                    "Temperature": data["hourly"]["temperature_2m"]
                })
                print(f"{city}: {len(df)} records extracted (past_days={past_days}, forecast_days={forecast_days})")
                return df
            elif response.status_code == 429:
                print(f"Rate limited for {city}, waiting 60s and retrying...")
                sleep(60)
                return self.get_recent_and_forecast_temperature(city, latitude, longitude, past_days, forecast_days)
            else:
                raise Exception(f"API Error {response.status_code}: {response.text}")
        except Exception as e:
            raise(e)
