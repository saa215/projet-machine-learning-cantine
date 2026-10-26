
"""Les modèles à comparer."""

from monprojet import config

from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


def candidates() -> dict:
    """Renvoie un dictionnaire {nom: modèle scikit-learn non entraîné}."""

    if config.TASK == "regression":
        return {
            # Baseline : prédit toujours la moyenne de la cible
            "baseline": DummyRegressor(strategy="mean"),

            # Modèles de régression
            "linear_regression": LinearRegression(),

            "ridge": Ridge(),

            "decision_tree": DecisionTreeRegressor(
                random_state=config.SEED
            ),

            "random_forest": RandomForestRegressor(
                n_estimators=100,
                random_state=config.SEED
            ),

            "extra_trees": ExtraTreesRegressor(
                n_estimators=100,
                random_state=config.SEED
            ),
        }

    elif config.TASK == "classification":
        return {
            # Baseline : prédit toujours la classe majoritaire
            "baseline": DummyClassifier(strategy="most_frequent"),

            # Modèles de classification
            "logistic_regression": LogisticRegression(
                max_iter=1000,
                random_state=config.SEED
            ),

            "decision_tree": DecisionTreeClassifier(
                random_state=config.SEED
            ),

            "random_forest": RandomForestClassifier(
                n_estimators=100,
                random_state=config.SEED
            ),

            "extra_trees": ExtraTreesClassifier(
                n_estimators=100,
                random_state=config.SEED
            ),
        }

    else:
        raise ValueError(
            f"Tâche inconnue : {config.TASK!r}. "
            "Utilisez 'regression' ou 'classification'."
        )
