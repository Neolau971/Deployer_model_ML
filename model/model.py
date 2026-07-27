from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

TARGET_COL = "a_quitte_l_entreprise"

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR.parent / "data" / "dataCentral.csv"
MODEL_PATH = BASE_DIR / "model.joblib"
FEATURE_COLUMNS_PATH = BASE_DIR / "feature_columns.joblib"

data = pd.read_csv(CSV_PATH)

X = data.drop(columns=[TARGET_COL])
y = data[TARGET_COL]

X = pd.get_dummies(X)

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
    n_jobs=-1
)

grid.fit(X_train, y_train)

joblib.dump(grid.best_estimator_, MODEL_PATH)
joblib.dump(X.columns.tolist(), FEATURE_COLUMNS_PATH)
print(f"Modèle sauvegardé dans {MODEL_PATH}")
print(f"Colonnes sauvegardées dans {FEATURE_COLUMNS_PATH}")