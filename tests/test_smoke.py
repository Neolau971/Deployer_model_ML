from pathlib import Path
from io import BytesIO
from types import SimpleNamespace

import pandas as pd
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model.main import load_data_from_upload

BASE_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = BASE_DIR / "data" / "dataCentral.csv"

def make_upload_file():
    csv_text = CSV_PATH.read_text(encoding="utf-8")
    return SimpleNamespace(file=BytesIO(csv_text.encode("utf-8")))


def test_load_data_from_upload_returns_dataframe():
    upload = make_upload_file()
    df = load_data_from_upload(upload)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_expected_columns_present():
    upload = make_upload_file()
    df = load_data_from_upload(upload)
    expected = {
        "a_quitte_l_entreprise",
        "age",
        "heure_supplementaires",
        "niveau_hierarchique_poste",
    }
    assert expected.issubset(df.columns)


def test_target_is_binary():
    upload = make_upload_file()
    df = load_data_from_upload(upload)
    values = set(df["a_quitte_l_entreprise"].dropna().unique())
    assert values.issubset({0, 1})


def test_no_missing_target():
    upload = make_upload_file()
    df = load_data_from_upload(upload)
    assert df["a_quitte_l_entreprise"].notna().all()