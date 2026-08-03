import pandas as pd


# Agrège la température horaire de plusieurs villes en un dataset national :
#   - moyenne pondérée par la population de chaque ville (Paris pèse plus que Lorient)
#   - min / max bruts parmi les villes
#   - écart-type non pondéré (dispersion brute entre les villes)
class NationalTemperatureAggregator:

    # df_weather : colonnes City, Timestamp, Temperature
    # df_population : colonnes city, population (ou city_id, city, latitude, longitude, population)
    def transform(self, df_weather: pd.DataFrame, df_population: pd.DataFrame) -> pd.DataFrame:
        # 1. ajoute la population de chaque ville sur chaque ligne météo
        df_weather = df_weather.merge(df_population[["city", "population"]], left_on="City", right_on="city", how="left")
        if df_weather["population"].isna().any():
            villes_sans_population = df_weather.loc[df_weather["population"].isna(), "City"].unique()
            raise Exception(f"Population manquante pour les villes : {list(villes_sans_population)}")

        # poids de température utilisé pour la moyenne pondérée (temp * population)
        df_weather["temperature_x_population"] = df_weather["Temperature"] * df_weather["population"]

        # 2. agrégation par heure : moyenne pondérée + min/max/écart-type simples
        df_national = df_weather.groupby("Timestamp").apply(
            lambda g: pd.Series({
                "temperature_weighted_mean": g["temperature_x_population"].sum() / g["population"].sum(),
                "temperature_min": g["Temperature"].min(),
                "temperature_max": g["Temperature"].max(),
                "temperature_std": g["Temperature"].std()
            }),
            include_groups=False
        ).reset_index()

        # 3. tri chronologique + arrondi à 2 décimales (moyenne pondérée et écart-type
        # sont des statistiques calculées avec beaucoup trop de décimales sinon)
        df_national.sort_values("Timestamp", inplace=True)
        df_national.reset_index(inplace=True, drop=True)
        temperature_columns = ["temperature_weighted_mean", "temperature_min", "temperature_max", "temperature_std"]
        df_national[temperature_columns] = df_national[temperature_columns].round(2)

        # avertit si une heure n'a pas toutes les villes (ex: aux bornes d'une plage,
        # certaines villes ont 1h de moins/plus que d'autres) sans bloquer le calcul
        n_cities = df_weather["City"].nunique()
        n_cities_per_hour = df_weather.groupby("Timestamp")["City"].nunique()
        incomplete_hours = n_cities_per_hour[n_cities_per_hour != n_cities]
        if len(incomplete_hours) > 0:
            print(f"WARNING: {len(incomplete_hours)} hour(s) with fewer than {n_cities} cities:")
            print(incomplete_hours.to_string())

        return df_national
