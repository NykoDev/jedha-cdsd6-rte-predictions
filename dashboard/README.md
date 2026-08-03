# Dashboard RTE — prévision de consommation électrique

Dashboard Streamlit à 2 pages :
- **Prédiction** : dernière prévision (moyenne pondérée XGBoost/Prophet) vs consommation réelle,
  déclenchable via le bouton "Lancer une prédiction" (appelle l'API FastAPI, `api/main.py`).
- **Historique** : consommation réelle sur toute la période disponible (2022-2025).

Déployé sur [Streamlit Community Cloud](https://streamlit.io/cloud) (app principale :
`dashboard/app.py`, dépendances : `dashboard/requirements.txt`). Nécessite les variables d'env
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_BUCKET` (accès S3) et `API_URL` (URL de l'API
FastAPI déployée), à renseigner via le gestionnaire de secrets de la plateforme.

URL : https://jedha-cdsd6-rte-predictions-esmnwbfkwwknwuud7eo9uh.streamlit.app/
