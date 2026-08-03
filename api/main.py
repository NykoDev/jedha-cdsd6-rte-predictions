import os
import sys

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

# Permet `python api/main.py` depuis n'importe quel répertoire : par défaut Python n'ajoute à
# sys.path que le dossier du script (api/), pas la racine du repo, donc `from ml...` échouerait
# sinon (ModuleNotFoundError: No module named 'ml'). Sans effet dans le conteneur Docker, qui fixe
# déjà PYTHONPATH=/app, mais nécessaire pour un lancement local direct.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml import s3_utils
from ml.predict_ensemble import run_ensemble_prediction

load_dotenv()

app = FastAPI()

# Clé S3 fixe : les nouvelles prédictions sont ajoutées à la suite du fichier existant, sauf pour
# les heures déjà présentes (même target_datetime_utc), qui sont remplacées — pas un simple
# écrasement (cf. ml/s3_utils.upsert_dataframe_csv)
PREDICTIONS_S3_KEY = "data/previsions.csv"


@app.get("/predict")
async def predict():
    try:
        predictions_df = run_ensemble_prediction()
    except Exception as e:
        return {"status": "error", "step": "prediction", "detail": str(e)}

    try:
        s3_utils.upsert_dataframe_csv(predictions_df, PREDICTIONS_S3_KEY, key_column="target_datetime_utc")
    except Exception as e:
        return {"status": "error", "step": "upload_s3", "detail": str(e)}

    return {"status": "ok", "message": f"{PREDICTIONS_S3_KEY} créé sur S3", "rows": len(predictions_df)}


# Import de ce module sans lancer le serveur (utile pour les tests) : uvicorn ne démarre que si
# le script est exécuté directement, comme prévu par le Dockerfile (`CMD ["python", "api/main.py"]`)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
