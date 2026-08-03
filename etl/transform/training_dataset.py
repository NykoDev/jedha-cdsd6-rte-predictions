import pandas as pd


# Regroupe la consommation RTE demi-horaire en moyenne horaire.
#
# RTE semble allouer un nombre fixe de 48 demi-heures par jour calendaire, ce qui
# génère 2 lignes dupliquées le lendemain du passage à l'heure d'été (jour réel de
# 23h) et fait perdre la 25e heure réelle du jour du passage à l'heure d'hiver (jour
# réel de 25h, ex: 2021-10-31 23:00->00:00 heure locale) — confirmé absent aussi
# directement sur l'API RTE, donc non récupérable en re-demandant les données.
class HourlyConsumptionAggregator:

    # df_raw_half_hourly : colonnes start_date, end_date, value (ISO8601 local, offset +01:00/+02:00)
    def transform(self, df_raw_half_hourly: pd.DataFrame) -> pd.DataFrame:
        df_raw = df_raw_half_hourly.copy()

        # déduplique et trie par instant chronologique réel (pas par ordre alphabétique
        # de la chaîne de date, incorrect autour d'un changement d'heure où les offsets
        # +01:00/+02:00 se mélangent)
        df_raw.drop_duplicates(subset=["start_date"], keep="last", inplace=True)
        df_raw.sort_values(by="start_date", key=lambda col: pd.to_datetime(col, utc=True), inplace=True)
        df_raw.reset_index(inplace=True, drop=True)

        # agrégation horaire : on garde le format de date d'origine (ISO8601 avec offset
        # local +01:00/+02:00) en dérivant l'heure par simple remplacement textuel
        # ":30:" -> ":00:", sans reparser/reformater les dates
        df_raw["start_hour"] = df_raw["start_date"].str.replace(":30:", ":00:")

        # sort=False : df_raw est déjà dans l'ordre chronologique réel, on le conserve
        # tel quel plutôt que de laisser groupby retrier "start_hour" comme une chaîne
        df_hourly = df_raw.groupby("start_hour", sort=False).agg(
            end_date=("end_date", "last"),
            value=("value", "mean")
        ).reset_index().rename(columns={"start_hour": "start_date"})
        df_hourly = df_hourly[["start_date", "end_date", "value"]]

        return self._fill_missing_hours(df_hourly)

    # comble le trou de la 25e heure par la moyenne de l'heure précédente et de
    # l'heure suivante, en repassant par l'instant UTC réel pour détecter les trous
    # (indépendant des changements d'offset local)
    def _fill_missing_hours(self, df_hourly: pd.DataFrame) -> pd.DataFrame:
        utc_index = pd.to_datetime(df_hourly["start_date"], utc=True)
        value_by_utc = pd.Series(df_hourly["value"].to_numpy(), index=utc_index)
        full_range = pd.date_range(utc_index.min(), utc_index.max(), freq="h", tz="UTC")
        missing_utc = full_range.difference(utc_index)

        if len(missing_utc) == 0:
            return df_hourly

        print(f"Filling {len(missing_utc)} missing hour(s) with average of previous/next hour:")
        filled_rows = []
        one_hour = pd.Timedelta(hours=1)
        for ts in missing_utc:
            prev_value = value_by_utc.loc[ts - one_hour]
            next_value = value_by_utc.loc[ts + one_hour]
            avg_value = (prev_value + next_value) / 2
            local_start = ts.tz_convert("Europe/Paris")
            local_end = local_start + one_hour
            print(f"  {ts} (local {local_start}) -> value={avg_value} (prev={prev_value}, next={next_value})")
            filled_rows.append({
                "start_date": local_start.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "end_date": local_end.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "value": avg_value
            })

        df_hourly = pd.concat([df_hourly, pd.DataFrame(filled_rows)], ignore_index=True)
        # remet les ":" dans l'offset (strftime donne "+0100", le reste du fichier utilise "+01:00")
        df_hourly["start_date"] = df_hourly["start_date"].str.replace(r"([+-]\d{2})(\d{2})$", r"\1:\2", regex=True)
        df_hourly["end_date"] = df_hourly["end_date"].str.replace(r"([+-]\d{2})(\d{2})$", r"\1:\2", regex=True)
        df_hourly.sort_values(by="start_date", key=lambda col: pd.to_datetime(col, utc=True), inplace=True)
        df_hourly.reset_index(inplace=True, drop=True)
        return df_hourly


# Fusionne la consommation RTE (heure locale française avec offset saisonnier) et la
# température nationale Open-Meteo (déjà en UTC) sur le même instant réel, en passant
# tout en UTC pour éviter tout décalage silencieux. Réutilisée par les 2 pipelines
# (historique et quotidien), avec une fenêtre de dates optionnelle.
class TrainingDatasetBuilder:

    # df_hourly_consumption : colonnes start_date, end_date, value (sortie de HourlyConsumptionAggregator)
    # df_national_temp : colonnes Timestamp, temperature_weighted_mean, temperature_min, temperature_max, temperature_std
    # start/end : bornes UTC optionnelles (start inclus, end exclu) pour restreindre la fenêtre
    # how : "inner" (défaut, ETL historique) ne garde que les heures où consommation ET météo
    # sont connues ; "left" (ETL quotidien) garde aussi les heures de prévision météo sans
    # consommation connue (consumption = NaN), nécessaire pour la fenêtre de prévision
    def build(self, df_hourly_consumption: pd.DataFrame, df_national_temp: pd.DataFrame,
              start: pd.Timestamp = None, end: pd.Timestamp = None, how: str = "inner") -> pd.DataFrame:
        df_rte = df_hourly_consumption.copy()
        df_rte["datetime_utc"] = pd.to_datetime(df_rte["start_date"], utc=True)

        # Open-Meteo : "2021-01-01T00:00" est une chaîne naïve mais représente déjà de
        # l'UTC (l'API répond en GMT par défaut) -> on la tague simplement en UTC
        df_meteo = df_national_temp.copy()
        df_meteo["datetime_utc"] = pd.to_datetime(df_meteo["Timestamp"], utc=True)

        # côté météo en premier : en mode "left", ce sont ses heures (passé + prévision)
        # qui pilotent le résultat, la consommation n'étant connue que pour le passé
        df_train = df_meteo.merge(df_rte, on="datetime_utc", how=how)

        # heure locale dérivée (rythme de vie français, ex: pic de consommation vers 19h heure locale)
        df_train["local_hour"] = df_train["datetime_utc"].dt.tz_convert("Europe/Paris").dt.hour

        if start is not None:
            df_train = df_train[df_train["datetime_utc"] >= start]
        if end is not None:
            df_train = df_train[df_train["datetime_utc"] < end]

        df_train.rename(columns={"value": "consumption"}, inplace=True)
        df_train = df_train[[
            "datetime_utc", "local_hour", "consumption",
            "temperature_weighted_mean", "temperature_min", "temperature_max", "temperature_std"
        ]]
        df_train.sort_values("datetime_utc", inplace=True)
        df_train.reset_index(inplace=True, drop=True)
        return df_train
