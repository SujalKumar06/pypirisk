"""Per-prediction explanations as SHAP values in log-odds units.

LightGBM computes exact TreeSHAP values itself (``pred_contrib=True``). For logistic
regression on standardised features, the SHAP value of a feature is its coefficient
times its standardised value (Lundberg and Lee, 2017, linear-model case), so no extra
library is needed for either model.
"""

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin


def shap_values(kind: str, model: ClassifierMixin, features: pd.DataFrame) -> pd.DataFrame:
    """Return each feature's contribution to each prediction (uncalibrated log-odds).

    Args:
        kind: ``"logistic_regression"`` or ``"lightgbm"``.
        model: The fitted candidate model.
        features: Model inputs, columns in training order.

    Returns:
        A frame shaped like ``features`` with one contribution per cell.

    Raises:
        ValueError: If ``kind`` is not a known candidate.
    """
    if kind == "lightgbm":
        values = model.predict(features, pred_contrib=True)[:, :-1]
    elif kind == "logistic_regression":
        scaled = model[:-1].transform(features)
        values = scaled * model[-1].coef_[0]
    else:
        raise ValueError(f"Unknown model type: {kind!r}")
    return pd.DataFrame(np.asarray(values), columns=features.columns, index=features.index)


def top_factors(contributions: pd.Series, values: pd.Series, count: int = 3) -> list[dict]:
    """Pick the features that pushed one prediction most, in either direction.

    Args:
        contributions: SHAP values of one row.
        values: Feature values of the same row.
        count: How many factors to return.

    Returns:
        Up to ``count`` dicts with ``feature``, ``value`` and ``contribution``
        (positive raises risk), largest absolute contribution first.
    """
    order = contributions.abs().sort_values(ascending=False).index[:count]
    return [
        {
            "feature": feature,
            "value": None if pd.isna(values[feature]) else float(values[feature]),
            "contribution": round(float(contributions[feature]), 4),
        }
        for feature in order
    ]


def mean_absolute_shap(shap: pd.DataFrame) -> pd.Series:
    """Global feature importance: mean absolute SHAP value per feature.

    Args:
        shap: Output of :func:`shap_values` over many rows.

    Returns:
        Importance per feature, largest first.
    """
    return shap.abs().mean().sort_values(ascending=False)
