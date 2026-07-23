from pathlib import Path
from io import BytesIO
from types import SimpleNamespace
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model.main import load_data_from_upload

BASE_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = BASE_DIR / "data" / "dataCentral.csv"

def make_upload_file():
    csv_text = CSV_PATH.read_text(encoding="utf-8")
    return SimpleNamespace(file=BytesIO(csv_text.encode("utf-8")))

def test_train_test_split_stratified():
    from sklearn.model_selection import train_test_split
    df = load_data_from_upload(make_upload_file())
    X = df.drop(columns=["a_quitte_l_entreprise"])
    y = df["a_quitte_l_entreprise"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    assert len(X_train) + len(X_test) == len(df)


def test_essential_numeric_columns_exist():
    df = load_data_from_upload(make_upload_file())
    numeric_cols = [
        "a_quitte_l_entreprise",
        "age",
        "heure_supplementaires",
        "niveau_hierarchique_poste",
        "nombre_participation_pee",
        "annees_dans_l_entreprise",
        "annees_dans_le_poste_actuel",
    ]
    missing = [c for c in numeric_cols if c not in df.columns]
    assert not missing