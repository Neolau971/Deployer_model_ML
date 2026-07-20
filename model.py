import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict,
    GridSearchCV
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score
)
from sklearn.inspection import permutation_importance
import shap


def load_data(path="dataCentral.csv"):
    return pd.read_csv(path)


def evaluate_model(model, X_train, y_train, X_test, y_test, threshold=0.5):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    y_proba_cv = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    precision_cv, recall_cv, _ = precision_recall_curve(y_train, y_proba_cv)
    ap_cv = average_precision_score(y_train, y_proba_cv)

    plt.figure(figsize=(7, 5))
    plt.plot(recall_cv, precision_cv, label=f"RandomForest CV (AP = {ap_cv:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Courbe Precision-Recall - RandomForestClassifier")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    model.fit(X_train, y_train)

    y_proba_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test >= threshold).astype(int)

    print("=== TEST ===")
    print(confusion_matrix(y_test, y_pred_test))
    print(classification_report(y_test, y_pred_test))

    return model, y_proba_test


def plot_feature_importance(model, X):
    feature_importance = (
        pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_
        })
        .sort_values("importance", ascending=False)
    )

    top_15 = feature_importance.head(15)

    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=top_15,
        x="importance",
        y="feature",
        color="steelblue"
    )
    plt.xlabel("Importance")
    plt.title("Top 15 features - Importances (Random Forest)")
    plt.tight_layout()
    plt.show()

    return feature_importance


def plot_permutation_importance(model, X_test, y_test, X_columns):
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="f1"
    )

    perm_importance = (
        pd.DataFrame({
            "feature": X_columns,
            "importance": result.importances_mean
        })
        .sort_values("importance", ascending=False)
    )

    print(perm_importance.head(15))
    return perm_importance


def shap_analysis(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    shap.plots.beeswarm(shap_values[:, :, 1], max_display=15)
    shap.plots.waterfall(shap_values[0, :, 1], max_display=15)

    shap.plots.scatter(
        shap_values[:, "satisfaction_globale", 1],
        color=shap_values[:, "heure_supplementaires", 1]
    )

    shap.plots.scatter(
        shap_values[:, "nombre_participation_pee", 1],
        color=shap_values[:, "heure_supplementaires", 1]
    )

    shap.plots.scatter(
        shap_values[:, "age", 1],
        color=shap_values[:, "heure_supplementaires", 1]
    )

    shap.plots.scatter(
        shap_values[:, "niveau_hierarchique_poste", 1],
        color=shap_values[:, "heure_supplementaires", 1]
    )


def main():
    data = load_data()

    X = data.drop(columns=["a_quitte_l_entreprise"])
    y = data["a_quitte_l_entreprise"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    param_grid = {
        "n_estimators": [200],
        "max_depth": [10],
        "min_samples_leaf": [10]
    }

    base_model = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        verbose=1,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Meilleurs paramètres :", grid.best_params_)
    print("Meilleur score CV :", grid.best_score_)

    best_model = grid.best_estimator_

    evaluate_model(best_model, X_train, y_train, X_test, y_test, threshold=0.5)
    plot_feature_importance(best_model, X_train)
    plot_permutation_importance(best_model, X_test, y_test, X.columns)
    shap_analysis(best_model, X_test)


if __name__ == "__main__":
    main()