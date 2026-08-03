import pandas as pd
from sqlalchemy import text

from etl.config import get_db_engine


# Charge la liste des villes suivies (city_pop, NeonDB) : centralise une requête
# jusqu'ici dupliquée dans plusieurs scripts.
class CityPopRepository:

    def load_cities(self) -> pd.DataFrame:
        engine = get_db_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT city_id, city, latitude, longitude, population FROM city_pop;"))
            return pd.DataFrame(result.fetchall(), columns=["city_id", "city", "latitude", "longitude", "population"])
