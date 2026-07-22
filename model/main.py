from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from io import StringIO
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score
from pydantic import ValidationError

from model.schemas.employee import EmployeeInput

model = FastAPI()

TARGET_COL = "a_quitte_l_entreprise"

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


def train_and_evaluate(data: pd.DataFrame):
    if TARGET_COL not in data.columns:
        raise HTTPException(status_code=400, detail=f"Colonne cible '{TARGET_COL}' absente.")

    X = data.drop(columns=[TARGET_COL])
    y = data[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    base_model = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    param_grid = {
        "n_estimators": [200],
        "max_depth": [10],
        "min_samples_leaf": [10]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        verbose=0,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_proba_test = best_model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test >= 0.5).astype(int)

    report = classification_report(y_test, y_pred_test, output_dict=True)
    cm = confusion_matrix(y_test, y_pred_test).tolist()
    precision_cv, recall_cv, _ = precision_recall_curve(y_train, cross_val_predict(
        best_model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1])
    ap_cv = average_precision_score(y_train, cross_val_predict(
        best_model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1])

    return {
        "best_params": grid.best_params_,
        "best_cv_score": grid.best_score_,
        "confusion_matrix": cm,
        "classification_report": report,
        "average_precision_cv": ap_cv,
        "n_rows": len(data),
        "n_features": X.shape[1],
        "predictions": y_pred_test.tolist(),
        "probabilities": y_proba_test.tolist(),
    }

@model.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = load_data_from_upload(file)
    results = train_and_evaluate(data)
    return JSONResponse(content=results)