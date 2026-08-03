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

# ETL 2 : construit le dataset "courant" (fenêtre glissante J-28 + météo prévisionnelle)
# pour l'entraînement quotidien du modèle Prophet. Lancement manuel pour la démo.
#
# Mode direct (par défaut, date de référence = aujourd'hui) : endpoints forecast
# (météo, un seul appel couvre historique récent + prévision réelle) et short_term
# (RTE, donnée très récente).
# Mode backfill (--date dans le passé, rétroprévision) : endpoints archive (météo) et
# consolidated (RTE), seuls capables de fournir une donnée finalisée sur une plage de
# dates arbitraire. Pas de vraie prévision dans ce mode (les heures au-delà de J n'ont
# simplement pas de donnée météo, une prévision passée n'ayant pas de sens).
#
# Dataset de sortie unique : température (passé + prévision) mergée à la consommation
# (passé seulement) — consumption vide sur les heures de prévision.

PRODUCTION_DATA_DIR = os.path.join(DATA_DIR, "production")
TRAINING_WINDOW_DAYS = 28
FORECAST_API_DAYS = 2      # marge demandée à l'API (voir build_reference_time) pour
FORECAST_WINDOW_HOURS = 24 # être sûr de couvrir les FORECAST_WINDOW_HOURS réelles depuis la référence


def reference_time_for(run_date: date, is_backfill: bool) -> datetime:
    if is_backfill:
        midnight_paris = datetime.combine(run_date, time(0, 0), tzinfo=zoneinfo.ZoneInfo("Europe/Paris"))
        return midnight_paris.astimezone(timezone.utc)
    return datetime.now(timezone.utc)


def extract_weather(cities: pd.DataFrame, reference_time: datetime, is_backfill: bool) -> pd.DataFrame:
    extractor = OpenMeteoExtractor()
    dfs = []
    if is_backfill:
        start_date = (reference_time - timedelta(days=TRAINING_WINDOW_DAYS)).strftime("%Y-%m-%d")
        end_date = reference_time.strftime("%Y-%m-%d")
        for _, row in cities.iterrows():
            dfs.append(extractor.get_hourly_temperature(row.city, row.latitude, row.longitude, start_date, end_date))
    else:
        for _, row in cities.iterrows():
            dfs.append(extractor.get_recent_and_forecast_temperature(
                row.city, row.latitude, row.longitude, past_days=TRAINING_WINDOW_DAYS, forecast_days=FORECAST_API_DAYS
            ))
    return pd.concat(dfs, ignore_index=True)


def extract_consumption(reference_time: datetime, is_backfill: bool) -> pd.DataFrame:
    extractor = RTEExtractor()
    if is_backfill:
        start_date = extractor.format_date(
            (reference_time - timedelta(days=TRAINING_WINDOW_DAYS)).astimezone(zoneinfo.ZoneInfo("Europe/Paris")).strftime("%Y-%m-%d")
        )
        end_date = extractor.format_date(reference_time.astimezone(zoneinfo.ZoneInfo("Europe/Paris")).strftime("%Y-%m-%d"))
        data = extractor.get_historical_consumption(start_date, end_date)
        values = data["consolidated_power_consumption"][0]["values"]
    else:
        values = extractor.get_recent_consumption(days=TRAINING_WINDOW_DAYS)
    return pd.DataFrame(values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construit le dataset courant (J-28 + prévision) pour l'entraînement quotidien Prophet.")
    parser.add_argument("--date", type=str, default=None, help="Date de référence YYYY-MM-DD (défaut : aujourd'hui). Une date passée active le mode backfill (rétroprévision).")
    parser.add_argument("--bucket", type=str, default=os.getenv("AWS_BUCKET"), help="Bucket S3 cible (défaut : variable d'env AWS_BUCKET).")
    args = parser.parse_args()

    run_date = date.fromisoformat(args.date) if args.date else date.today()
    is_backfill = run_date != date.today()
    reference_time = reference_time_for(run_date, is_backfill)

    cities = CityPopRepository().load_cities()
    df_weather = extract_weather(cities, reference_time, is_backfill)
    df_consumption_raw = extract_consumption(reference_time, is_backfill)

    df_national_temp = NationalTemperatureAggregator().transform(df_weather, cities)
    df_hourly_consumption = HourlyConsumptionAggregator().transform(df_consumption_raw)

    # "left" côté météo : garde les heures de prévision (au-delà de reference_time) même
    # sans consommation connue ; borne basse J-28, pas de borne haute (la météo s'arrête
    # d'elle-même à la fin de la fenêtre extraite : today en backfill, +24h en direct)
    training_start = reference_time - timedelta(days=TRAINING_WINDOW_DAYS)
    df_dataset = TrainingDatasetBuilder().build(df_hourly_consumption, df_national_temp, start=training_start, how="left")

    if not is_backfill:
        forecast_end = reference_time + timedelta(hours=FORECAST_WINDOW_HOURS)
        df_dataset = df_dataset[df_dataset["datetime_utc"] < forecast_end]

    print(f"Current dataset: {len(df_dataset)} rows ({training_start} -> {run_date}"
          + ("" if is_backfill else f" + {FORECAST_WINDOW_HOURS}h forecast") + ")")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"current_dataset_{timestamp}.csv"
    S3Loader().upload(df_dataset, os.path.join(PRODUCTION_DATA_DIR, filename), args.bucket, f"current-data/{filename}")
