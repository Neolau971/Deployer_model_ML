import pandas as pd
from sqlalchemy import create_engine

DB_USER = "postgres"
DB_PASSWORD = "admin"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "bdd"

def get_engine():
    return create_engine(
        f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

def save_dataframe_to_db(df: pd.DataFrame, table_name: str) -> None:
    engine = get_engine()
    df.to_sql(table_name, con=engine, if_exists="append", index=False, chunksize=1000)