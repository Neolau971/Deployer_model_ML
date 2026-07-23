import pandas as pd
import psycopg
from sqlalchemy import create_engine

DB_USER = "postgres"
DB_PASSWORD = "admin"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "bdd"
TABLE_NAME = "data_central"

def ensure_database():
    conn = psycopg.connect(
        dbname="postgres",
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        autocommit=True,
    )
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM pg_catalog.pg_database WHERE lower(datname) = lower(%s)",
            (DB_NAME,),
        )
        exists = cur.fetchone() is not None
        if not exists:
            cur.execute(f'CREATE DATABASE "{DB_NAME}"')
    conn.close()

def get_engine():
    return create_engine(
        f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

def save_dataframe_to_db(df: pd.DataFrame) -> None:
    ensure_database()
    engine = get_engine()
    df.to_sql(TABLE_NAME, con=engine, if_exists="append", index=False, chunksize=1000)