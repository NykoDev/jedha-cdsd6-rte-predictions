import pandas as pd

from ml.model_prophet import ProphetConsumptionModel
from ml.model_xgboost import XGBoostConsumptionModel

# Moyenne arrondie des 2 poids appris dans training/train_ensemble.ipynb (protocole croisé
# hiver/été) : 0.82 (fit hiver → test été) et 0.61 (fit été → test hiver). La moyenne pondérée y
# a battu le stacking (qui sur-apprenait sur le peu de données disponibles) — à réajuster si de
# nouvelles données permettent d'affiner ce poids.
ENSEMBLE_WEIGHT_XGB = 0.7


# Ne gère aucune logique propre à un modèle (feature engineering, fit, hyperparamètres) : tout ça
# vit dans XGBoostConsumptionModel / ProphetConsumptionModel. Ce module se contente de charger les
# données de production, d'appeler les 2 classes et de combiner leurs prédictions.
def run_ensemble_prediction() -> pd.DataFrame:
    # data/daily_dataset.csv (clé S3 fixe, cf. ProphetConsumptionModel.DAILY_DATASET_S3_KEY)
    # contient à la fois l'historique récent (consumption renseignée) et les 24h de température
    # prévisionnelle (consumption vide) : on sépare les deux pour alimenter les 2 modèles, qui
    # attendent chacun (historique, prévision météo) séparément.
    prophet_model = ProphetConsumptionModel()
    daily_df = prophet_model.load_daily_dataset()
    history_df = daily_df[daily_df["consumption"].notna()]
    forecast_df = daily_df[daily_df["consumption"].isna()]

    xgb_model = XGBoostConsumptionModel()
    xgb_model.load_model()
    xgb_pred = xgb_model.predict(history_df, forecast_df=forecast_df)

    # Prophet n'a pas de modèle unique rechargeable (fenêtre glissante) : on ré-entraîne à la
    # volée sur les données de production les plus récentes, comme dans les notebooks.
    prophet_model.train(history_df)
    prophet_pred = prophet_model.predict(forecast_df)

    combined = xgb_pred.merge(prophet_pred, on="target_datetime_utc", suffixes=("_xgb", "_prophet"))
    combined["y_pred_ensemble"] = (
        ENSEMBLE_WEIGHT_XGB * combined["y_pred_xgb"] + (1 - ENSEMBLE_WEIGHT_XGB) * combined["y_pred_prophet"]
    )

    return combined[["target_datetime_utc", "y_pred_xgb", "y_pred_prophet", "y_pred_ensemble"]]
