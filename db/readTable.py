import pandas as pd
from sqlalchemy import create_engine, text

DB_USER = "postgres"
DB_PASSWORD = "admin"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "bdd"


def get_engine():
    return create_engine(
        f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )


def get_predictions_by_request_id(request_id: str) -> list[dict]:
    engine = get_engine()
    query = text("""
        SELECT *
        FROM prediction
        WHERE request_id = :request_id
        ORDER BY row_id
    """)

    with engine.connect() as conn:
        rows = conn.execute(query, {"request_id": request_id}).mappings().all()

    return [dict(row) for row in rows]