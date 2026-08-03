from itertools import product

import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

from ml import s3_utils

# Reprend la config gagnante "4w_split" de training/train_prophet_sliding_window.ipynb
# (sections 6sexies/7.2/8, MAPE≈2.52% sur les 2 fenêtres de test hiver/été) : fenêtre
# d'entraînement glissante de 4 semaines, saisonnalité hebdomadaire conditionnelle
# weekday/weekend, jours fériés FR, régresseurs météo. Contrairement à XGBoost, Prophet n'a pas
# de modèle unique réutilisable : il est ré-entraîné à chaque prédiction sur les 4 dernières
# semaines disponibles (déjà établi dans les notebooks) — pas de save_model()/load_model() ici,
# ça n'aurait aucune valeur puisque le modèle est de toute façon reconstruit à chaque prévision.


class ProphetConsumptionModel:

    TRAIN_WINDOW = pd.DateOffset(weeks=4)
    REGRESSOR_COLUMNS = ["temperature_weighted_mean", "temperature_min", "temperature_max", "temperature_std"]
    DAILY_DATASET_S3_KEY = "data/daily_dataset.csv"

    # Meilleure combinaison trouvée par tune_4w_hyperparameters("4w_split") (section 7.2)
    DEFAULT_HYPERPARAMETERS = {
        "changepoint_prior_scale": 0.35,
        "seasonality_prior_scale": 1.0,
        "seasonality_mode": "multiplicative",
    }

    def __init__(self, hyperparameters: dict = None):
        self.hyperparameters = dict(hyperparameters) if hyperparameters else dict(self.DEFAULT_HYPERPARAMETERS)
        self.model = None
        self.cutoff = None  # dernier "ds" (UTC naïf) vu à l'entraînement

    # Dataset "quotidien" (historique récent, remis à jour chaque jour), utilisé aussi bien pour
    # ré-entraîner que pour construire les régresseurs de prévision — distinct du dataset
    # historique complet 2022-2025 utilisé par XGBoost pour son entraînement de production.
    def load_daily_dataset(self) -> pd.DataFrame:
        return s3_utils.load_csv(self.DAILY_DATASET_S3_KEY)

    # ds volontairement laissé en UTC naïf (pas de conversion Europe/Paris) : évite les collisions
    # de timestamp dupliqué à l'heure ambiguë du passage à l'heure d'hiver (même choix que dans
    # train_prophet_sliding_window.ipynb)
    @staticmethod
    def _build_prophet_frame(hourly_df: pd.DataFrame) -> pd.DataFrame:
        hourly = hourly_df.copy()
        if not isinstance(hourly.index, pd.DatetimeIndex):
            hourly = hourly.set_index("datetime_utc")
        hourly.index = pd.to_datetime(hourly.index, utc=True)
        hourly = hourly.sort_index()

        return pd.DataFrame({
            "ds": hourly.index.tz_localize(None),
            "y": hourly["consumption"].to_numpy(),
            **{col: hourly[col].to_numpy() for col in ProphetConsumptionModel.REGRESSOR_COLUMNS},
        })

    # Même construction que _build_prophet_frame, mais sans "y" : pour les données de prévision
    # météo (etl/build-daily-production-dataset.py), qui n'ont pas encore de consommation réelle
    @staticmethod
    def _build_forecast_frame(forecast_df: pd.DataFrame) -> pd.DataFrame:
        forecast = forecast_df.copy()
        if not isinstance(forecast.index, pd.DatetimeIndex):
            forecast = forecast.set_index("datetime_utc")
        forecast.index = pd.to_datetime(forecast.index, utc=True)
        forecast = forecast.sort_index()

        return pd.DataFrame({
            "ds": forecast.index.tz_localize(None),
            **{col: forecast[col].to_numpy() for col in ProphetConsumptionModel.REGRESSOR_COLUMNS},
        })

    @staticmethod
    def _add_weekday_flags(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["is_weekday"] = df["ds"].dt.dayofweek < 5
        df["is_weekend"] = ~df["is_weekday"]
        return df

    # Construit et fit un Prophet pour une combinaison d'hyperparamètres donnée — factorisé pour
    # être appelé à la fois par train() (config courante) et finetune() (grille de recherche)
    def _fit_prophet(self, train_slice, changepoint_prior_scale, seasonality_prior_scale, seasonality_mode) -> Prophet:
        model = Prophet(
            growth="linear",
            daily_seasonality=True,
            weekly_seasonality=False,  # remplacée par les 2 saisonnalités conditionnelles ci-dessous
            yearly_seasonality=False,  # fenêtre de 4 semaines : pas assez de recul pour l'estimer
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
        )
        model.add_seasonality(name="weekly_weekday", period=7, fourier_order=3, condition_name="is_weekday")
        model.add_seasonality(name="weekly_weekend", period=7, fourier_order=3, condition_name="is_weekend")
        model.add_country_holidays(country_name="FR")
        for col in self.REGRESSOR_COLUMNS:
            model.add_regressor(col)
        model.fit(train_slice)
        return model

    def train(self, hourly_df: pd.DataFrame) -> Prophet:
        prophet_df = self._add_weekday_flags(self._build_prophet_frame(hourly_df))
        cutoff = prophet_df["ds"].max()
        train_slice = prophet_df[(prophet_df["ds"] > cutoff - self.TRAIN_WINDOW) & (prophet_df["ds"] <= cutoff)]

        self.model = self._fit_prophet(
            train_slice,
            self.hyperparameters.get("changepoint_prior_scale", 0.35),
            self.hyperparameters.get("seasonality_prior_scale", 1.0),
            self.hyperparameters.get("seasonality_mode", "multiplicative"),
        )
        self.cutoff = cutoff
        return self.model

    # Grille sur (changepoint_prior_scale, seasonality_prior_scale, seasonality_mode), reprend
    # l'esprit de tune_4w_hyperparameters (train_prophet_sliding_window.ipynb section 7) mais
    # simplifié : backtest sur les `n_backtest_cutoffs` derniers jours de hourly_df (au lieu des
    # 2 fenêtres fixes hiver/été du notebook, propres au dataset d'entraînement complet).
    def finetune(
        self,
        hourly_df: pd.DataFrame,
        changepoint_prior_scales: list = None,
        seasonality_prior_scales: list = None,
        seasonality_modes: list = None,
        n_backtest_cutoffs: int = 4,
    ) -> dict:
        changepoint_prior_scales = changepoint_prior_scales or [0.05, 0.2, 0.35, 0.5]
        seasonality_prior_scales = seasonality_prior_scales or [0.5, 1.0, 5.0, 10.0]
        seasonality_modes = seasonality_modes or ["additive", "multiplicative"]

        prophet_df = self._add_weekday_flags(self._build_prophet_frame(hourly_df))
        max_ds = prophet_df["ds"].max()
        backtest_cutoffs = [max_ds - pd.Timedelta(24 * i, unit="h") for i in range(n_backtest_cutoffs, 0, -1)]

        best_params, best_mape = None, float("inf")
        for cps, sps, mode in product(changepoint_prior_scales, seasonality_prior_scales, seasonality_modes):
            trial_mapes = []
            for cutoff in backtest_cutoffs:
                train_slice = prophet_df[(prophet_df["ds"] > cutoff - self.TRAIN_WINDOW) & (prophet_df["ds"] <= cutoff)]
                future_slice = prophet_df[(prophet_df["ds"] > cutoff) & (prophet_df["ds"] <= cutoff + pd.Timedelta(24, unit="h"))]
                if future_slice.empty:
                    continue
                model = self._fit_prophet(train_slice, cps, sps, mode)
                forecast = model.predict(future_slice[["ds"] + self.REGRESSOR_COLUMNS + ["is_weekday", "is_weekend"]])
                trial_mapes.append(mean_absolute_percentage_error(future_slice["y"], forecast["yhat"]) * 100)

            if not trial_mapes:
                continue
            mean_mape = sum(trial_mapes) / len(trial_mapes)
            if mean_mape < best_mape:
                best_mape = mean_mape
                best_params = {"changepoint_prior_scale": cps, "seasonality_prior_scale": sps, "seasonality_mode": mode}

        self.hyperparameters = best_params
        return self.hyperparameters

    # `forecast_df` : température prévisionnelle (datetime_utc + REGRESSOR_COLUMNS) sur les 24h
    # suivant la fin de la fenêtre d'entraînement, produite par etl/build-daily-production-dataset.py
    def predict(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None or self.cutoff is None:
            raise RuntimeError(
                "Modèle non entraîné : appeler train() d'abord (Prophet n'a pas de modèle unique "
                "rechargeable, contrairement à XGBoost — il faut ré-entraîner sur les données récentes)."
            )

        future = self._add_weekday_flags(self._build_forecast_frame(forecast_df))
        future = future[(future["ds"] > self.cutoff) & (future["ds"] <= self.cutoff + pd.Timedelta(24, unit="h"))]

        forecast = self.model.predict(future[["ds"] + self.REGRESSOR_COLUMNS + ["is_weekday", "is_weekend"]])
        return pd.DataFrame({
            "target_datetime_utc": future["ds"].dt.tz_localize("UTC").to_numpy(),
            "y_pred": forecast["yhat"].to_numpy(),
        })
