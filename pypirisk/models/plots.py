"""Evaluation and explanation charts logged to MLflow as PNG artifacts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 - the backend must be chosen before import
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay  # noqa: E402


def _save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return path


def feature_importance_chart(importance: pd.Series, path: Path) -> Path:
    """Horizontal bar chart of mean absolute SHAP values.

    Args:
        importance: Importance per feature, largest first.
        path: Output PNG.

    Returns:
        ``path``.
    """
    figure, axis = plt.subplots(figsize=(7, 0.3 * len(importance) + 1))
    importance.iloc[::-1].plot.barh(ax=axis, color="#4C72B0")
    axis.set_xlabel("mean |SHAP value| (log-odds)")
    axis.set_title("Feature importance")
    return _save(figure, path)


def confusion_chart(labels: np.ndarray, flagged: np.ndarray, path: Path) -> Path:
    """Confusion matrix of the packages flagged at the top-k threshold.

    Args:
        labels: True labels.
        flagged: 1 for flagged packages.
        path: Output PNG.

    Returns:
        ``path``.
    """
    figure, axis = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        labels, flagged, display_labels=["active", "abandoned"], ax=axis, colorbar=False
    )
    return _save(figure, path)
