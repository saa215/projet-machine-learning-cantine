
"""Chargement et découpage des données."""

import math
import pandas as pd
from sklearn.model_selection import train_test_split

from monprojet import config


def load_data(path=None) -> pd.DataFrame:
    """Charge les données, vérifie la cible et retire les cibles manquantes."""
    csv_path = config.DATA_PATH if path is None else path
    df = pd.read_csv(csv_path)

    if config.TARGET not in df.columns:
        raise ValueError(
            f"La colonne cible '{config.TARGET}' est absente du fichier CSV."
        )

    df = df.dropna(subset=[config.TARGET]).copy()
    return df


def feature_columns(df: pd.DataFrame) -> list:
    """Renvoie les colonnes d'entrée."""
    if config.FEATURES is not None:
        return list(config.FEATURES)

    excluded = {config.TARGET, config.TIME_COLUMN}
    return [c for c in df.columns if c not in excluded]


def split(df: pd.DataFrame):
    """Sépare les données en ensembles d'entraînement et de test."""
    features = feature_columns(df)

    if config.TARGET not in df.columns:
        raise ValueError(f"Colonne cible absente : {config.TARGET}")

    X = df[features]
    y = df[config.TARGET]

    if config.SPLIT == "random":
        # Stratifier les cibles catégorielles lorsque c'est possible.
        categorical = (
            not pd.api.types.is_numeric_dtype(y)
            or pd.api.types.is_bool_dtype(y)
        )
        stratify = y if categorical else None

        # La stratification exige au moins deux exemples par classe.
        if stratify is not None:
            counts = y.value_counts()
            n_test = math.ceil(len(y) * config.TEST_SIZE)
            n_train = len(y) - n_test

            if (
                counts.min() < 2
                or n_test < len(counts)
                or n_train < len(counts)
            ):
                stratify = None

        return train_test_split(
            X,
            y,
            test_size=config.TEST_SIZE,
            random_state=config.SEED,
            stratify=stratify,
        )

    elif config.SPLIT == "chronologique":
        time_col = config.TIME_COLUMN

        if time_col is None or time_col not in df.columns:
            raise ValueError(
                "Une colonne temporelle valide est nécessaire "
                "pour le découpage chronologique."
            )

        dates = pd.to_datetime(df[time_col], errors="raise")
        order = dates.argsort().to_numpy()

        X_sorted = X.iloc[order]
        y_sorted = y.iloc[order]

        n_test = math.ceil(len(df) * config.TEST_SIZE)
        if n_test <= 0 or n_test >= len(df):
            raise ValueError(
                "TEST_SIZE doit permettre de conserver des données "
                "pour l'entraînement et pour le test."
            )

        return (
            X_sorted.iloc[:-n_test],
            X_sorted.iloc[-n_test:],
            y_sorted.iloc[:-n_test],
            y_sorted.iloc[-n_test:],
        )

    else:
        raise ValueError(
            f"Mode de découpage inconnu : {config.SPLIT}"
        )
