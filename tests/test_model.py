from pathlib import Path
from io import BytesIO
from types import SimpleNamespace
import pandas as pd
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model.main import load_data_from_upload


def make_upload_file(csv_text: str):
    return SimpleNamespace(file=BytesIO(csv_text.encode("utf-8")))

CSV_TEST = """age,niveau_hierarchique_poste,heure_supplementaires,satisfaction_globale,nombre_participation_pee,annees_dans_l_entreprise,annees_dans_le_poste_actuel,a_quitte_l_entreprise
30,2,1,4,3,5,2,0
45,3,0,2,1,10,4,1
38,1,1,3,2,6,1,0
41,4,0,1,0,12,5,1
29,2,1,5,4,3,1,0
50,3,0,2,1,15,6,1
36,2,1,4,2,7,2,0
43,4,0,1,0,11,5,1
32,2,1,4,3,4,1,0
47,3,0,2,1,13,6,1
"""


def test_load_data_from_upload_returns_dataframe():
    df = load_data_from_upload(make_upload_file(CSV_TEST))
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_expected_columns_present():
    df = load_data_from_upload(make_upload_file(CSV_TEST))
    expected = {
        "a_quitte_l_entreprise",
        "age",
        "heure_supplementaires",
        "satisfaction_globale",
        "niveau_hierarchique_poste",
    }
    assert expected.issubset(df.columns)


def test_target_is_binary():
    df = load_data_from_upload(make_upload_file(CSV_TEST))
    values = set(df["a_quitte_l_entreprise"].dropna().unique())
    assert values.issubset({0, 1})


def test_no_missing_target():
    df = load_data_from_upload(make_upload_file(CSV_TEST))
    assert df["a_quitte_l_entreprise"].notna().all()


def test_train_test_split_stratified():
    from sklearn.model_selection import train_test_split
    df = load_data_from_upload(make_upload_file(CSV_TEST))
    X = df.drop(columns=["a_quitte_l_entreprise"])
    y = df["a_quitte_l_entreprise"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    assert len(X_train) + len(X_test) == len(df)


def test_essential_numeric_columns_exist():
    df = load_data_from_upload(make_upload_file(CSV_TEST))
    numeric_cols = [
        "age",
        "niveau_hierarchique_poste",
        "nombre_participation_pee",
        "annees_dans_l_entreprise",
        "annees_dans_le_poste_actuel",
    ]
    missing = [c for c in numeric_cols if c not in df.columns]
    assert not missing