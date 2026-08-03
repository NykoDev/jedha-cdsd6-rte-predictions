# Prévision de la consommation électrique nationale (RTE)

Projet de certification Jedha — module *Prévisions ML en production*.

Pipeline de bout en bout : collecte de la consommation électrique (API RTE) et de la météo (Open-Meteo), entraînement et comparaison de plusieurs modèles de prévision à horizon 24h, puis mise en production (API + dashboard) d'un modèle ensembliste.

## Architecture

```
RTE API     ──┐
              ├──► etl/historical_pipeline.py ──► S3 ──► training/*.ipynb ──► MLflow (tracking local)
Open-Meteo  ──┘        (dataset 2022-2025)                (XGBoost, Prophet, ensemble)
                                                                  │
                                                                  ▼
                                                  ml/train_xgboost.py ──► S3 (models/xgboost/model.pkl)

RTE + Open-Meteo ──► etl/daily_pipeline.py ──► S3 (dataset courant : J-28 + prévision météo)
                                                                  │
                                                                  ▼
                                              ml/predict_ensemble.py (XGBoost + Prophet)
                                                                  │
                                                                  ▼
                                     api/main.py (FastAPI, EC2) ──► S3 (data/previsions.csv)
                                                                  │
                                                                  ▼
                                        dashboard/ (Streamlit Community Cloud)
```

## Stack

| Couche | Outils |
|---|---|
| Collecte | API RTE (consommation), Open-Meteo (météo), PostgreSQL/NeonDB (référentiel villes) |
| Stockage | AWS S3 |
| Modèles | XGBoost, Prophet (Meta), scikit-learn |
| Tracking d'expériences | MLflow (local, backend SQLite) |
| Service de prédiction | FastAPI (déployé sur EC2) |
| Dashboard | Streamlit (Community Cloud) |

## Structure

```
├── etl/
│   ├── historical_pipeline.py  # Dataset historique complet (2022-2025) → S3
│   ├── daily_pipeline.py       # Dataset courant (J-28 + prévision météo) → S3
│   └── extract/, transform/, load/  # Classes réutilisées par les 2 pipelines
├── training/                   # Notebooks d'analyse/comparaison de modèles (suivis via MLflow)
│   ├── train_models.ipynb              # Baseline, régression linéaire, random forest, XGBoost
│   ├── train_prophet_sliding_window.ipynb  # Prophet, comparaison de fenêtres glissantes
│   └── train_ensemble.ipynb            # XGBoost + Prophet : moyenne pondérée vs stacking
├── ml/                          # Code de production (classes réutilisables, pas de notebook)
│   ├── model_xgboost.py            # Entraînement, fine-tuning, prédiction, sauvegarde S3
│   ├── model_prophet.py            # Idem, réentraîné à chaque prédiction (fenêtre glissante)
│   ├── train_xgboost.py            # Script de prod : dataset S3 → modèle → pickle S3
│   ├── predict_ensemble.py         # Orchestration des 2 modèles + moyenne pondérée
│   └── s3_utils.py                 # Helpers S3 partagés (pickle, CSV, upsert)
├── api/
│   └── main.py                  # FastAPI, endpoint GET /predict
├── dashboard/                   # Dashboard Streamlit (déploiement indépendant, venv séparé)
│   ├── app.py, page_*.py           # Pages "Prédiction" et "Historique"
│   └── requirements.txt
└── data/                        # Datasets locaux utilisés par les notebooks/scripts
```

## Résultats (MAPE, horizon 24h, test sur l'année 2025)

| Modèle | MAPE |
|---|---|
| Baseline naïve (persistance J-1) | 5.10% |
| Régression linéaire | 4.17% |
| Random Forest | 2.10% |
| Prophet (fenêtre glissante 4 semaines, saisonnalité weekday/weekend) | 2.52% |
| XGBoost (tuné) | 1.77% |
| **Ensemble (moyenne pondérée XGBoost/Prophet)** | **1.59 – 1.72%** |

Le stacking (régression linéaire sur les 2 prédictions) a aussi été testé mais sur-apprend sur le peu de données disponibles (voir `train_ensemble.ipynb`) — la moyenne pondérée simple généralise mieux et est celle retenue en production (`ml/predict_ensemble.py`).

## Configuration

`.env` à la racine :

```env
postgres=              # URL de connexion PostgreSQL (NeonDB), table city_pop
RTE_API_KEY=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_BUCKET=            # Bucket S3, utilisé par etl/, ml/, api/ et le dashboard
```

`dashboard/.env` (config séparée, déploiement indépendant) :

```env
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_BUCKET=
API_URL=               # URL de l'API FastAPI déployée (défaut : http://localhost:8000/predict)
CURRENT_DAY=            # Optionnel, format DD/MM/YYYY : fige la date "aujourd'hui" du dashboard
```

## Lancement

```bash
pip install -r requirements.txt
```

1. **ETL** (optionnel si les CSV de `data/` sont déjà présents) : `python -m etl.historical_pipeline`
2. **Analyse et comparaison de modèles** : notebooks dans `training/` (`mlflow ui` pour consulter les runs)
3. **Entraînement de production** : `python ml/train_xgboost.py` (sauvegarde le modèle sur S3)
4. **API** : `python api/main.py` — expose `GET /predict`
5. **Dashboard** (venv séparé, voir `dashboard/requirements.txt`) :
   ```bash
   streamlit run dashboard/app.py
   ```
