FROM python:3.10-slim

# --- Dépendances système ---
# build-essential / gcc / g++ : compilation de Prophet (pystan) et pmdarima
# libgomp1                    : OpenMP requis par XGBoost et Prophet
# libpq-dev                   : psycopg2-binary (connexion PostgreSQL / Neon)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Installation des dépendances Python ---
# On copie uniquement requirements.txt à cette étape pour exploiter le cache Docker (le reste du
# code change plus souvent que les dépendances).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# --- Activation des imports absolus depuis la racine du projet ---
# Nécessaire pour que `from ml.model_xgboost import ...` fonctionne depuis api/main.py
ENV PYTHONPATH=/app

# --- Copie du code source ---
COPY . .

EXPOSE 8000

# main.py se trouve dans le sous-dossier api/
CMD ["python", "api/main.py"]