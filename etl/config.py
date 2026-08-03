import os

from dotenv import load_dotenv
from sqlalchemy import create_engine

# racine du projet (ce fichier est dans etl/, donc un seul niveau au-dessus)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


# connexion NeonDB partagée par tous les modules ayant besoin de city_pop
def get_db_engine():
    return create_engine(os.getenv("postgres"))
