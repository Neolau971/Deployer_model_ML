from pathlib import Path

import pandas as pd
import pytest
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model.main import load_data


def test_data_file_exists():
    path = Path("data/dataCentral.csv")
    assert path.exists()


def test_load_data_returns_dataframe():
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_expected_columns_present():
    df = load_data()
    expected = {
        "a_quitte_l_entreprise",
        "age",
        "heure_supplementaires",
        "satisfaction_globale",
        "niveau_hierarchique_poste",
    }
    assert expected.issubset(df.columns)


def test_target_is_binary():
    df = load_data()
    target_values = set(df["a_quitte_l_entreprise"].dropna().unique())
    assert target_values.issubset({0, 1})


def test_no_missing_target():
    df = load_data()
    assert df["a_quitte_l_entreprise"].notna().all()