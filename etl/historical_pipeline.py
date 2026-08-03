import argparse
import os
import zoneinfo
from datetime import date, datetime, time, timedelta, timezone

import pandas as pd

from etl.config import DATA_DIR
from etl.extract.city_pop import CityPopRepository
from etl.extract.open_meteo import OpenMeteoExtractor
from etl.extract.rte import RTEExtractor
from etl.load.s3_loader import S3Loader
from etl.transform.national_temperature import NationalTemperatureAggregator
from etl.transform.training_dataset import HourlyConsumptionAggregator, TrainingDatasetBuilder

# ETL 1 : construit le dataset historique (consommation + météo nationale) sur une
# plage de dates complète, depuis zéro. Sert à l'entraînement du modèle XGBoost et à
# l'affichage des données historiques dans le dashboard.
# Ce script ne fait qu'enchaîner des appels aux classes extract/transform/load,
# aucune logique métier ici.

DEFAULT_START = date(2022, 1, 1)
DEFAULT_END = date(2025, 12, 31)


def local_midnight_utc(d: date) -> datetime:
    return datetime.combine(d, time(0, 0), tzinfo=zoneinfo.ZoneInfo("Europe/Paris")).astimezone(timezone.utc)


def extract_weather(cities: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    extractor = OpenMeteoExtractor()
    dfs = [
        extractor.get_hourly_temperature(row.city, row.latitude, row.longitude, start.isoformat(), end.isoformat())
        for _, row in cities.iterrows()
    ]
    return pd.concat(dfs, ignore_index=True)


def extract_consumption(start: date, end: date) -> pd.DataFrame:
    records = []
    for year in range(start.year, end.year + 1):
        extractor = RTEExtractor()
        extractor.get_annual_power_consumption(year)
        records.extend(extractor.data)
    return pd.DataFrame(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construit le dataset historique (consommation + météo nationale).")
    parser.add_argument("--start", type=str, default=DEFAULT_START.isoformat(), help="Date de début YYYY-MM-DD (défaut 2022-01-01).")
    parser.add_argument("--end", type=str, default=DEFAULT_END.isoformat(), help="Date de fin YYYY-MM-DD incluse (défaut 2025-12-31).")
    parser.add_argument("--bucket", type=str, default=os.getenv("AWS_BUCKET"), help="Bucket S3 cible (défaut : variable d'env AWS_BUCKET).")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    cities = CityPopRepository().load_cities()

    df_weather = extract_weather(cities, start, end)
    df_consumption_raw = extract_consumption(start, end)

    df_national_temp = NationalTemperatureAggregator().transform(df_weather, cities)
    df_hourly_consumption = HourlyConsumptionAggregator().transform(df_consumption_raw)

    # bornes UTC calculées sur l'année locale française (start inclus, end exclu :
    # minuit Paris du lendemain de la date de fin)
    start_utc = local_midnight_utc(start)
    end_utc = local_midnight_utc(end + timedelta(days=1))
    df_train = TrainingDatasetBuilder().build(df_hourly_consumption, df_national_temp, start=start_utc, end=end_utc)

    print(f"Historical dataset: {len(df_train)} rows ({start} -> {end})")

    filename = f"training_dataset_{start.year}_{end.year}.csv"
    local_path = os.path.join(DATA_DIR, filename)
    S3Loader().upload(df_train, local_path, args.bucket, f"historical/{filename}")
