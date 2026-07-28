from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from io import StringIO
import pandas as pd
import joblib
from pydantic import ValidationError

from db.createDB import ensure_database
from db.createTable import save_dataframe_to_db
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

    return {
        "predictions": y_pred.tolist()
    }


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    data = load_data_from_upload(file)
    ensure_database()
    save_dataframe_to_db(data)
    results = predict_with_model(request, data)
    return JSONResponse(content=results)