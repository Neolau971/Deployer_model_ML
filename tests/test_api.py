# tests/test_api.py
import pytest
import pandas as pd
from io import BytesIO
from fastapi.testclient import TestClient
from datetime import datetime
from pathlib import Path
from fastapi import UploadFile

# Import de l'application FastAPI et des chemins réels
from model.main import app, MODEL_PATH, FEATURE_COLUMNS_PATH

BASE_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = BASE_DIR / "data" / "dataCentral.csv"

def make_upload_file(csv_path: Path | None = None):
    """
    Retourne un objet 'files' compatible avec TestClient.post(..., files=...).
    Exemple : {"file": ("dataCentral.csv", BytesIO(b"..."), "text/csv")}
    """
    if csv_path is None:
        csv_path = CSV_PATH

    csv_text = csv_path.read_text(encoding="utf-8")
    csv_bytes = csv_text.encode("utf-8")

    return {
        "file": (csv_path.name, BytesIO(csv_bytes), "text/csv")
    }

@pytest.fixture(scope="module")
def client():
    """
    Fixture pour créer un client de test avec le vrai modèle.
    On suppose que model/model.joblib et model/feature_columns.joblib existent.
    """
    # Vérification optionnelle (pour avoir un message clair si le modèle manque)
    assert MODEL_PATH.exists(), f"Modèle introuvable : {MODEL_PATH}"
    assert FEATURE_COLUMNS_PATH.exists(), f"Fichier de colonnes introuvable : {FEATURE_COLUMNS_PATH}"

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_csv_content():
    # Retourne un bytes, prêt pour BytesIO
    return (
        b"age,niveau_hierarchique_poste,heure_supplementaires,satisfaction_globale,nombre_participation_pee\n"
        b"30,2,10,3.5,5\n"
        b"35,3,15,4.0,3"
    )


@pytest.fixture
def invalid_csv_content():
    return (
        b"age,niveau_hierarchique_poste,heure_supplementaires,satisfaction_globale,nombre_participation_pee\n"
        b"abc,2,10,3.5,5"
    )


class TestPredictEndpoint:
    def test_predict_valid_csv_returns_200(self, client, valid_csv_content):
        files = make_upload_file()
        response = client.post("/predict", files=files)

        assert response.status_code == 200
        data = response.json()
        assert "request_id" in data
        assert "created_at" in data
        assert "predictions" in data
        assert "probabilities" in data
        assert len(data["predictions"]) == 1470
        assert len(data["probabilities"]) == 1470
        assert all(p in [0, 1] for p in data["predictions"])
        assert all(0.0 <= prob <= 1.0 for prob in data["probabilities"])

    def test_predict_invalid_csv_returns_422(self, client, invalid_csv_content):
        files = {"file": ("test.csv", BytesIO(invalid_csv_content), "text/csv")}
        response = client.post("/predict", files=files)

        assert response.status_code == 422
        data = response.json()
        assert isinstance(data["detail"], list)
        assert data["detail"][0]["row"] == 0

    def test_predict_malformed_csv_returns_422(self, client):
        df = pd.read_csv(CSV_PATH, dtype=str)
    
        df.loc[1, 'age'] = "abc"
        
        csv_content = df.to_csv(index=False)
        file_like = BytesIO(csv_content.encode("utf-8"))
        files = {"file": ("modified.csv", file_like, "text/csv")}
        response = client.post("/predict", files=files)

        assert response.status_code == 422

    def test_predict_missing_file_returns_422(self, client):
        response = client.post("/predict")
        assert response.status_code == 422  # FastAPI validation error

    def test_predict_created_at_is_iso8601(self, client, valid_csv_content):
        files = make_upload_file()
        response = client.post("/predict", files=files)

        data = response.json()
        created_at = data["created_at"]
        # Vérifier le format ISO 8601 avec timezone
        datetime.fromisoformat(created_at.replace("Z", "+00:00"))


class TestReadPredictionsEndpoint:
    def test_read_predictions_existing_request_id(self, client, valid_csv_content):
        # D'abord faire une prédiction
        files = make_upload_file()
        predict_response = client.post("/predict", files=files)
        request_id = predict_response.json()["request_id"]

        # Ensuite lire les prédictions
        response = client.get(f"/predictions/{request_id}")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1470
        for row in data:
            assert row["request_id"] == request_id
            assert "row_id" in row
            assert "prediction" in row
            assert "probability" in row
            assert "created_at" in row

    def test_read_predictions_nonexistent_request_id(self, client):
        fake_request_id = "00000000-0000-0000-0000-000000000000"
        response = client.get(f"/predictions/{fake_request_id}")

        assert response.status_code == 404
        assert "Aucune prédiction trouvée" in response.json()["detail"]