"""Candidate models, training data assembly and the heuristic baseline."""

from typing import Any

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HEURISTIC_FEATURE = "months_since_last_release"
CANDIDATES = ("logistic_regression", "lightgbm")


def build_candidate(kind: str, settings: dict[str, Any], seed: int) -> ClassifierMixin:
    """Create an unfitted candidate model.

    Both candidates weight classes by inverse frequency because abandonment is the
    minority class; probabilities are fixed afterwards by calibration.

    Args:
        kind: ``"logistic_regression"`` or ``"lightgbm"``.
        settings: Hyperparameters from ``params.yaml`` for that kind.
        seed: Random seed.

    Returns:
        An unfitted classifier. Logistic regression is wrapped with median imputation
        and scaling; LightGBM handles missing values itself.

    Raises:
        ValueError: If ``kind`` is not a known candidate.
    """
    if kind == "logistic_regression":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=settings["C"],
                        max_iter=settings["max_iter"],
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )
    if kind == "lightgbm":
        return LGBMClassifier(
            **settings, class_weight="balanced", random_state=seed, subsample_freq=1, verbose=-1
        )
    raise ValueError(f"Unknown model type: {kind!r}")


def join_features_and_labels(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Attach labels to the feature rows of eligible packages.

    Args:
        features: Feature table with ``name`` and ``snapshot`` columns.
        labels: Label table with ``name``, ``snapshot``, ``split`` and ``label``.

    Returns:
        One row per labelled package and snapshot, with features, ``split`` and ``label``.

    Raises:
        ValueError: If a labelled package has no feature row.
    """
    joined = labels.merge(features, on=["name", "snapshot"], how="left", indicator=True)
    missing = joined[joined["_merge"] == "left_only"]
    if not missing.empty:
        example = missing.iloc[0][["name", "snapshot"]].to_dict()
        raise ValueError(f"{len(missing)} labelled rows have no features, e.g. {example}")
    return joined.drop(columns="_merge")


def heuristic_score(frame: pd.DataFrame) -> pd.Series:
    """Score abandonment risk by months since the last release.

    This is the continuous form of the "no release in the past year" baseline: every
    eligible package released in the past year, so the binary form is nearly constant.

    Args:
        frame: Rows with a ``months_since_last_release`` column.

    Returns:
        The heuristic risk score for each row.
    """
    return frame[HEURISTIC_FEATURE].astype(float)
