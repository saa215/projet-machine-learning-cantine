
"""Entraîne chaque modèle candidat, l'évalue sur le jeu de test et les compare.

Usage, depuis la racine du projet : python -m monprojet.train
"""

import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from monprojet import config
from monprojet.data import load_data, split
from monprojet.models import candidates


HIGHER_IS_BETTER = {
    "accuracy": True,
    "recall": True,
    "precision": True,
    "f1": True,
    "mae": False,
    "rmse": False,
    "r2": True,
}


def build_preprocessing(X: pd.DataFrame) -> ColumnTransformer:
    """Prépare les colonnes numériques, catégorielles et textuelles."""

    text = [config.TEXT_COLUMN] if config.TEXT_COLUMN else []

    numeric = [
        c for c in X.columns
        if c not in text and pd.api.types.is_numeric_dtype(X[c])
    ]

    categorical = [
        c for c in X.columns
        if c not in text and c not in numeric
    ]

    steps = []

    if numeric:
        steps.append(
            (
                "nombres",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                ),
                numeric,
            )
        )

    if categorical:
        steps.append(
            (
                "categories",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore"),
                ),
                categorical,
            )
        )

    if text:
        steps.append(
            (
                "texte",
                TfidfVectorizer(),
                config.TEXT_COLUMN,
            )
        )

    return ColumnTransformer(steps, sparse_threshold=0)


def evaluate(model, X_test, y_test) -> dict:
    """Évalue le modèle et renvoie les métriques adaptées à la tâche."""

    y_pred = model.predict(X_test)

    if config.TASK == "classification":
        average = "binary" if y_test.nunique() == 2 else "macro"

        scores = {
            "accuracy": accuracy_score(y_test, y_pred),
            "recall": recall_score(
                y_test,
                y_pred,
                average=average,
                zero_division=0,
            ),
            "precision": precision_score(
                y_test,
                y_pred,
                average=average,
                zero_division=0,
            ),
            "f1": f1_score(
                y_test,
                y_pred,
                average=average,
                zero_division=0,
            ),
        }

    elif config.TASK == "regression":
        scores = {
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
        }

    else:
        raise ValueError(
            f"Tâche inconnue : {config.TASK!r}. "
            "Utilisez 'regression' ou 'classification'."
        )

    return {
        nom: round(float(valeur), 3)
        for nom, valeur in scores.items()
    }


def main() -> None:
    """Entraîne les candidats, compare les résultats et sauvegarde le meilleur."""

    # 1. Charger les données et créer les jeux d'entraînement et de test
    df = load_data()
    X_train, X_test, y_train, y_test = split(df)

    # 2. Entraîner et évaluer chaque candidat
    resultats = {}
    modeles_entraines = {}

    for nom, estimateur in candidates().items():
        print(f"\nEntraînement du modèle : {nom}")

        model = Pipeline(
            [
                ("preparation", build_preprocessing(X_train)),
                ("modele", estimateur),
            ]
        )

        model.fit(X_train, y_train)
        scores = evaluate(model, X_test, y_test)

        resultats[nom] = scores
        modeles_entraines[nom] = model

    if not resultats:
        raise ValueError("Aucun modèle candidat n'a été fourni.")

    # 3. Vérifier la métrique principale
    metrique = config.MAIN_METRIC

    if metrique not in HIGHER_IS_BETTER:
        raise ValueError(
            f"Métrique principale inconnue : {metrique!r}"
        )

    for nom, scores in resultats.items():
        if metrique not in scores:
            raise ValueError(
                f"La métrique {metrique!r} est absente "
                f"des résultats du modèle {nom!r}."
            )

    # 4. Comparer et trier les modèles
    tableau = pd.DataFrame(resultats).T

    tableau = tableau.sort_values(
        by=metrique,
        ascending=not HIGHER_IS_BETTER[metrique],
    )

    print("\n" + "=" * 60)
    print("COMPARAISON DES MODÈLES")
    print("=" * 60)
    print(tableau.to_string())

    meilleur_nom = tableau.index[0]
    meilleur_modele = modeles_entraines[meilleur_nom]

    print(f"\nMeilleur modèle : {meilleur_nom}")
    print(f"Métrique principale : {metrique}")
    print(f"Valeur obtenue : {resultats[meilleur_nom][metrique]}")

    # 5. Créer le dossier de sauvegarde
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Sauvegarder le modèle entraîné
    chemin_modele = config.MODELS_DIR / "model.joblib"
    joblib.dump(meilleur_modele, chemin_modele)

    # Sauvegarder les résultats dans un fichier JSON
    rapport = {
        "main_metric": metrique,
        "best": meilleur_nom,
        "results": resultats,
    }

    chemin_metriques = config.MODELS_DIR / "metrics.json"

    with open(chemin_metriques, "w", encoding="utf-8") as fichier:
        json.dump(
            rapport,
            fichier,
            ensure_ascii=False,
            indent=4,
        )

    print("\nSauvegarde terminée.")
    print(f"Modèle : {chemin_modele}")
    print(f"Métriques : {chemin_metriques}")


if __name__ == "__main__":
    main()
