import os
import sys

# Permet `python ml/train_xgboost.py` depuis n'importe quel répertoire : par défaut Python
# n'ajoute à sys.path que le dossier du script (ml/), pas la racine du repo, donc `from ml...`
# échouerait sinon (ModuleNotFoundError: No module named 'ml').
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml.model_xgboost import XGBoostConsumptionModel

# Script de production : charge le dataset historique depuis S3
# (XGBoostConsumptionModel.TRAINING_DATA_S3_KEY), cherche les meilleurs hyperparamètres, réentraîne
# sur toutes les données avec cette config, puis sauvegarde le modèle en pickle sur S3
# (XGBoostConsumptionModel.MODEL_S3_KEY). À relancer périodiquement quand le dataset est mis à jour.


def main():
    model = XGBoostConsumptionModel()

    print(f"Chargement du dataset d'entraînement depuis s3://.../{model.TRAINING_DATA_S3_KEY}")
    hourly_df = model.load_training_data()
    print(f"{len(hourly_df)} lignes chargées")

    print("Recherche d'hyperparamètres (finetune)...")
    best_hyperparameters = model.finetune(hourly_df)
    print(f"Meilleurs hyperparamètres trouvés : {best_hyperparameters}")

    print("Entraînement final sur l'ensemble du dataset...")
    model.train(hourly_df)

    print(f"Sauvegarde du modèle sur s3://.../{model.MODEL_S3_KEY}")
    model.save_model()
    print("Terminé.")


if __name__ == "__main__":
    main()
