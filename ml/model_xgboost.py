import holidays
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import ParameterSampler
from xgboost import XGBRegressor

from ml import s3_utils

# Reprend telle quelle la feature engineering vérifiée dans training/train_models.ipynb
# (sections 4 à 8) : lags de consommation par timestamp, stats glissantes, features calendaires
# cible + jour férié FR, encodages cycliques. Horizon de prévision fixe à 24h, cohérent avec le
# modèle "xgboost_tuned" retenu dans ce notebook (MAPE test ≈1.77%).


class XGBoostConsumptionModel:

    LAGS = [1, 24, 48, 168]  # 1h, 1 jour, 2 jours, 1 semaine avant t0
    TARGET_HORIZON_H = 24
    MODEL_S3_KEY = "models/xgboost/model.pkl"
    TRAINING_DATA_S3_KEY = "data/historic_dataset_2022_2025.csv"

    # Meilleurs hyperparamètres trouvés par la recherche aléatoire de train_models.ipynb
    # (section 13bis, cellule exécutée : val_mape=2.436%, MAPE test final=1.77%)
    DEFAULT_HYPERPARAMETERS = {
        "max_depth": 5,
        "learning_rate": 0.0629969476735087,
        "subsample": 0.7700623497964979,
        "colsample_bytree": 0.6693458614031088,
        "min_child_weight": 8,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "n_estimators": 903,
        "tree_method": "hist",
        "objective": "reg:squarederror",
        "random_state": 42,
    }

    def __init__(self, hyperparameters: dict = None):
        self.hyperparameters = dict(hyperparameters) if hyperparameters else dict(self.DEFAULT_HYPERPARAMETERS)
        self.model = None
        self.feature_columns = None

    # Dataset historique complet (2022-2025), utilisé pour l'entraînement de production
    # (ml/train_xgboost.py) — distinct des données "production" (training-data/forecast-weather)
    # consommées par predict(), qui elles couvrent une fenêtre récente glissante.
    def load_training_data(self) -> pd.DataFrame:
        return s3_utils.load_csv(self.TRAINING_DATA_S3_KEY)

    @staticmethod
    def _lag_by_timestamp(series: pd.Series, hours: int) -> np.ndarray:
        """Valeur de `series` exactement `hours` heures avant chaque timestamp de son index.
        Lookup par timestamp (pas un .shift() positionnel) pour rester robuste à d'éventuels
        trous dans la série."""
        target_ts = series.index - pd.Timedelta(hours, unit="h")
        return series.reindex(target_ts).to_numpy()

    # Construit la table supervisée (une ligne par origine t0, cible à t0+24h) et la liste des
    # colonnes de features. `forecast_df` (optionnel, colonnes datetime_utc/temperature_weighted_mean)
    # permet de fournir la température prévisionnelle sur les cibles futures en inférence, quand
    # `hourly_df` seul ne couvre pas encore ces heures (température non encore mesurée).
    def _build_features(self, hourly_df: pd.DataFrame, forecast_df: pd.DataFrame = None):
        hourly = hourly_df.copy()
        if not isinstance(hourly.index, pd.DatetimeIndex):
            hourly = hourly.set_index("datetime_utc")
        hourly.index = pd.to_datetime(hourly.index, utc=True)
        hourly = hourly.sort_index()

        for lag_h in self.LAGS:
            hourly[f"consumption_lag_{lag_h}"] = self._lag_by_timestamp(hourly["consumption"], lag_h)

        hourly["rolling_mean_24h"] = hourly["consumption"].rolling("24h").mean()
        hourly["rolling_std_24h"] = hourly["consumption"].rolling("24h").std()
        hourly["rolling_mean_168h"] = hourly["consumption"].rolling("168h").mean()
        hourly["temp_t0"] = hourly["temperature_weighted_mean"]

        origin_feature_columns = [f"consumption_lag_{h}" for h in self.LAGS] + [
            "rolling_mean_24h", "rolling_std_24h", "rolling_mean_168h", "temp_t0",
        ]

        supervised_df = hourly[origin_feature_columns].copy()
        supervised_df.index.name = "origin_datetime_utc"
        supervised_df["target_datetime_utc"] = supervised_df.index + pd.Timedelta(self.TARGET_HORIZON_H, unit="h")

        # Série de température couvrant l'historique réel +, si fournie, la prévision météo :
        # nécessaire pour connaître temp_target sur des cibles qui n'ont pas encore d'actual
        temp_series = hourly["temperature_weighted_mean"]
        if forecast_df is not None:
            forecast = forecast_df.copy()
            if not isinstance(forecast.index, pd.DatetimeIndex):
                forecast = forecast.set_index("datetime_utc")
            forecast.index = pd.to_datetime(forecast.index, utc=True)
            temp_series = pd.concat([temp_series, forecast["temperature_weighted_mean"]])
            temp_series = temp_series[~temp_series.index.duplicated(keep="last")].sort_index()

        supervised_df["target_consumption"] = hourly["consumption"].reindex(supervised_df["target_datetime_utc"]).to_numpy()
        supervised_df["temp_target"] = temp_series.reindex(supervised_df["target_datetime_utc"]).to_numpy()
        supervised_df = supervised_df.reset_index()

        target_local = supervised_df["target_datetime_utc"].dt.tz_convert("Europe/Paris")
        supervised_df["target_local_hour"] = target_local.dt.hour
        supervised_df["target_day_of_week"] = target_local.dt.dayofweek
        supervised_df["target_month"] = target_local.dt.month
        supervised_df["target_is_weekend"] = supervised_df["target_day_of_week"].isin([5, 6]).astype(int)

        fr_holidays = holidays.France(years=range(target_local.dt.year.min() - 1, target_local.dt.year.max() + 2))
        supervised_df["target_is_holiday"] = target_local.dt.date.isin(fr_holidays).astype(int)

        for col, short_name, period in [
            ("target_local_hour", "hour", 24),
            ("target_day_of_week", "dow", 7),
            ("target_month", "month", 12),
        ]:
            angle = 2 * np.pi * supervised_df[col] / period
            supervised_df[f"target_{short_name}_sin"] = np.sin(angle)
            supervised_df[f"target_{short_name}_cos"] = np.cos(angle)

        feature_columns = origin_feature_columns + [
            "temp_target",
            "target_local_hour", "target_day_of_week", "target_month", "target_is_weekend", "target_is_holiday",
            "target_hour_sin", "target_hour_cos",
            "target_dow_sin", "target_dow_cos",
            "target_month_sin", "target_month_cos",
        ]
        return supervised_df, feature_columns

    def train(self, hourly_df: pd.DataFrame) -> XGBRegressor:
        supervised_df, feature_columns = self._build_features(hourly_df)
        model_df = supervised_df.dropna(subset=feature_columns + ["target_consumption"]).reset_index(drop=True)

        X_train, y_train = model_df[feature_columns], model_df["target_consumption"]
        self.model = XGBRegressor(**self.hyperparameters)
        self.model.fit(X_train, y_train)
        self.feature_columns = feature_columns
        return self.model

    # Recherche aléatoire d'hyperparamètres (reprend train_models.ipynb section 13bis) : split
    # temporel train/validation sur les dernières lignes (par défaut les 15% les plus récentes),
    # early stopping XGBoost pour fixer n_estimators automatiquement. Met à jour
    # self.hyperparameters avec la meilleure combinaison trouvée (ne réentraîne pas le modèle
    # final : appeler train() ensuite pour refit sur toutes les données).
    def finetune(
        self,
        hourly_df: pd.DataFrame,
        n_iter: int = 12,
        validation_fraction: float = 0.15,
        early_stopping_rounds: int = 40,
        n_estimators_ceiling: int = 1500,
        random_state: int = 42,
    ) -> dict:
        supervised_df, feature_columns = self._build_features(hourly_df)
        model_df = (
            supervised_df.dropna(subset=feature_columns + ["target_consumption"])
            .sort_values("target_datetime_utc")
            .reset_index(drop=True)
        )

        split_idx = int(len(model_df) * (1 - validation_fraction))
        search_train_df = model_df.iloc[:split_idx]
        search_val_df = model_df.iloc[split_idx:]

        X_fit, y_fit = search_train_df[feature_columns], search_train_df["target_consumption"]
        X_val, y_val = search_val_df[feature_columns], search_val_df["target_consumption"]

        param_distributions = {
            "max_depth": randint(4, 10),
            "learning_rate": loguniform(0.03, 0.2),
            "subsample": uniform(0.6, 0.4),        # -> [0.6, 1.0]
            "colsample_bytree": uniform(0.6, 0.4),  # -> [0.6, 1.0]
            "min_child_weight": randint(1, 15),
            "reg_alpha": [0, 0.01, 0.1, 1],
            "reg_lambda": [0.5, 1, 2, 5, 10],
        }
        param_samples = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=random_state))

        best_config, best_mape = None, float("inf")
        for sampled in param_samples:
            model = XGBRegressor(
                **sampled,
                n_estimators=n_estimators_ceiling,
                tree_method="hist",
                random_state=random_state,
                objective="reg:squarederror",
                eval_metric="mae",
                early_stopping_rounds=early_stopping_rounds,
            )
            model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)

            y_pred_val = model.predict(X_val)
            val_mape = mean_absolute_percentage_error(y_val, y_pred_val) * 100
            n_estimators_used = model.best_iteration + 1

            if val_mape < best_mape:
                best_mape = val_mape
                best_config = {**sampled, "n_estimators": n_estimators_used}

        # cast explicite : ParameterSampler/np mélange int et float64 selon les paramètres tirés,
        # XGBoost lève TypeError si max_depth/min_child_weight/n_estimators ne sont pas des int
        for int_param in ("max_depth", "min_child_weight", "n_estimators"):
            best_config[int_param] = int(best_config[int_param])

        self.hyperparameters = {
            **best_config,
            "tree_method": "hist",
            "random_state": random_state,
            "objective": "reg:squarederror",
        }
        return self.hyperparameters

    # `forecast_df` : température prévisionnelle (datetime_utc, temperature_weighted_mean) sur
    # les heures cibles, produite par etl/build-daily-production-dataset.py. Sans elle, seules
    # les cibles déjà couvertes par `hourly_df` (donc dans le passé) peuvent être prédites.
    def predict(self, hourly_df: pd.DataFrame, forecast_df: pd.DataFrame = None) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Modèle non chargé/entraîné : appeler train() ou load_model() d'abord.")

        supervised_df, feature_columns = self._build_features(hourly_df, forecast_df=forecast_df)
        model_df = supervised_df.dropna(subset=feature_columns).reset_index(drop=True)

        y_pred = self.model.predict(model_df[feature_columns])
        return pd.DataFrame({
            "target_datetime_utc": model_df["target_datetime_utc"],
            "y_pred": y_pred,
        })

    def save_model(self) -> None:
        if self.model is None:
            raise RuntimeError("Aucun modèle entraîné à sauvegarder.")
        s3_utils.upload_pickle(self.model, self.MODEL_S3_KEY)

    def load_model(self) -> None:
        self.model = s3_utils.download_pickle(self.MODEL_S3_KEY)
