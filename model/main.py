from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from io import StringIO
from datetime import datetime, timezone
from uuid import uuid4
import pandas as pd
import joblib
from pydantic import ValidationError

from db.createDB import ensure_database
from db.createTable import save_dataframe_to_db
from db.readTable import get_predictions_by_request_id
from model.schemas.employee import EmployeeInput

TARGET_COL = "a_quitte_l_entreprise"
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"
FEATURE_COLUMNS_PATH = BASE_DIR / "feature_columns.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH.exists() and FEATURE_COLUMNS_PATH.exists():
        app.state.model = joblib.load(MODEL_PATH)
        app.state.feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    else:
        app.state.model = None
        app.state.feature_columns = None
    yield


app = FastAPI(lifespan=lifespan)


def load_data_from_upload(file: UploadFile):
    try:
        content = file.file.read().decode("utf-8")
        df = pd.read_csv(StringIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de lecture CSV: {str(e)}")

    errors = []
    valid_rows = []

    for idx, row in df.iterrows():
        try:
            item = EmployeeInput.model_validate(row.to_dict())
            valid_rows.append(item.model_dump())
        except ValidationError as e:
            errors.append({
                "row": int(idx),
                "errors": e.errors()
            })

    if errors:
        raise HTTPException(status_code=422, detail=errors)

    return pd.DataFrame(valid_rows)


def predict_with_model(request: Request, data: pd.DataFrame):
    model = request.app.state.model
    feature_columns = request.app.state.feature_columns

    if model is None or feature_columns is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")

    if TARGET_COL in data.columns:
        X = data.drop(columns=[TARGET_COL])
    else:
        X = data

    X = pd.get_dummies(X)
    X = X.reindex(columns=feature_columns, fill_value=0)

    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    result_df = pd.DataFrame({
        "row_id": range(len(data)),
        "prediction": y_pred,
        "probability": y_proba
    })

    return result_df


@app.post(
        "/predict",
    description=(
        "Le fichier CSV doit contenir une ligne par employé, avec les colonnes qui "
        "correspondent au modèle `EmployeeInput` (par exemple : `age`, "
        "`niveau_hierarchique_poste`, `heure_supplementaires`, "
        "`satisfaction_globale`, `nombre_participation_pee`, "
        "`a_quitte_l_entreprise`, etc.)."
    ),
    responses={
        200: {
            "description": "Prédictions générées avec succès",
            "content": {
                "application/json": {
                    "example": {
                        "request_id": "550e8400-e29b-41d4-a716-446655440000",
                        "created_at": "2026-07-23T14:00:00Z",
                        "predictions": [0, 1],
                        "probabilities": [0.12, 0.87],
                    }
                }
            },
        },
        422: {"description": "CSV invalide (lignes non conformes à EmployeeInput)"},
        500: {"description": "Modèle non chargé"},
    },
)

async def predict(request: Request, file: UploadFile = File(...)):
    request_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    data = load_data_from_upload(file)
    ensure_database()
    save_dataframe_to_db(data, "data_central")

    results = predict_with_model(request, data)
    results["request_id"] = request_id
    results["created_at"] = created_at

    save_dataframe_to_db(results, "prediction")

    return JSONResponse(content={
    "request_id": request_id,
    "created_at": created_at,
    "predictions": results["prediction"].tolist(),
    "probabilities": results["probability"].tolist()
})
@app.get(
        "/predictions/{request_id}",
            description="Retourne toutes les lignes de prédiction associées à un request_id.",
             responses={
        200: {
            "description": "Prédictions trouvées",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "request_id": "550e8400-e29b-41d4-a716-446655440000",
                            "row_id": 0,
                            "prediction": 1,
                            "probability": 0.87,
                            "created_at": "2026-07-23T14:00:00Z"
                        }
                    ]
                }
            }
        },
        404: {
            "description": "Aucune prédiction trouvée"
        }
    })
async def read_predictions(request_id: str):
    rows = get_predictions_by_request_id(request_id)

    if not rows:
        raise HTTPException(status_code=404, detail="Aucune prédiction trouvée pour ce request_id")

    return JSONResponse(content=rows)