"""Ranking metrics that tolerate small splits with a single class."""

import logging

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger(__name__)


def ranking_metrics(labels: np.ndarray, scores: np.ndarray, prefix: str) -> dict[str, float]:
    """ROC-AUC and PR-AUC, or nothing when only one class is present.

    Args:
        labels: True labels (1 = abandoned).
        scores: Risk scores; higher means more likely abandoned.
        prefix: Metric name prefix, e.g. ``"test_model"``.

    Returns:
        ``<prefix>_roc_auc`` and ``<prefix>_pr_auc``, or an empty dict if undefined.
    """
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        logger.warning("Ranking metrics undefined: one class only", extra={"prefix": prefix})
        return {}
    return {
        f"{prefix}_roc_auc": float(roc_auc_score(labels, scores)),
        f"{prefix}_pr_auc": float(average_precision_score(labels, scores)),
    }


def top_fraction_mask(scores: np.ndarray, fraction: float) -> np.ndarray:
    """Flag the highest-scored ``fraction`` of rows (at least one row).

    Args:
        scores: Risk scores.
        fraction: Share of rows to flag.

    Returns:
        A boolean mask.
    """
    scores = np.asarray(scores)
    count = max(1, int(round(len(scores) * fraction)))
    order = np.argsort(-scores, kind="stable")
    mask = np.zeros(len(scores), dtype=bool)
    mask[order[:count]] = True
    return mask
