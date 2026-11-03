"""Evaluate stage: score the held-out splits and compare with the heuristic baseline.

Run with ``python -m pypirisk.models.evaluate``. Metrics go to ``reports/metrics.json``
(for ``dvc metrics``) and to the training run in MLflow, with the precision-recall and
calibration curves (the same files ``dvc plots`` reads) and a top-k confusion matrix. The
stage also writes the drift reference: the feature distribution of the test snapshot,
the latest population the model was validated on.
"""

import logging
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

from pypirisk.io import read_json, write_json
from pypirisk.models import plots
from pypirisk.models.estimators import heuristic_score
from pypirisk.models.metrics import ranking_metrics, top_fraction_mask
from pypirisk.models.pyfunc import RiskModel
from pypirisk.models.train import Split, load_splits
from pypirisk.monitoring.drift import reference_profile
from pypirisk.stagelog import run_stage

logger = logging.getLogger("pypirisk.models.evaluate")

EVALUATED_SPLITS = ("validation", "test")


def evaluate_split(model: RiskModel, split: Split, name: str) -> dict[str, float]:
    """Score one split with the model and the heuristic.

    Args:
        model: The trained risk model.
        split: The split to score.
        name: Split name, used as the metric prefix.

    Returns:
        Rows, prevalence, and ROC-AUC and PR-AUC for the model and the heuristic.
    """
    if not len(split.labels):
        return {f"{name}_rows": 0.0}
    risk = model.risk(split.features)
    heuristic = heuristic_score(split.rows).to_numpy()
    return {
        f"{name}_rows": float(len(split.labels)),
        f"{name}_prevalence": float(split.labels.mean()),
        **ranking_metrics(split.labels, risk, f"{name}_model"),
        **ranking_metrics(split.labels, heuristic, f"{name}_heuristic"),
    }


def curve_points(xs: np.ndarray, ys: np.ndarray, x: str, y: str, limit: int = 200) -> list[dict]:
    """Thin a curve to at most ``limit`` evenly spaced points for a DVC plot file.

    Args:
        xs: X values.
        ys: Y values.
        x: Name of the x field.
        y: Name of the y field.
        limit: Largest number of points kept.

    Returns:
        ``[{x: ..., y: ...}, ...]`` in the input order.
    """
    keep = (
        np.unique(np.linspace(0, len(xs) - 1, min(limit, len(xs))).astype(int)) if len(xs) else []
    )
    return [{x: round(float(xs[i]), 5), y: round(float(ys[i]), 5)} for i in keep]


def write_plot_data(model: RiskModel, split: Split, directory: Path) -> None:
    """Write the test split's curves as JSON for ``dvc plots show`` and ``dvc plots diff``.

    The files are always written (empty when the split has one class), because DVC
    expects every declared plot output to exist.

    Args:
        model: The trained risk model.
        split: The test split.
        directory: Output directory, ``reports/plots``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    curves = {"pr_model.json": [], "pr_heuristic.json": [], "calibration.json": []}
    if len(np.unique(split.labels)) == 2:
        risk = model.risk(split.features)
        for name, scores in (
            ("pr_model.json", risk),
            ("pr_heuristic.json", heuristic_score(split.rows)),
        ):
            precision, recall, _ = precision_recall_curve(split.labels, scores)
            curves[name] = curve_points(recall, precision, "recall", "precision")
        observed, predicted = calibration_curve(split.labels, risk, n_bins=10, strategy="quantile")
        curves["calibration.json"] = curve_points(predicted, observed, "predicted", "observed")
    for name, points in curves.items():
        write_json(directory / name, points)


def latest_population(params: dict[str, Any]) -> pd.DataFrame:
    """Return the monitored features of every package at the latest labelled snapshot.

    All packages are used, not only the labelled (eligible) ones, because batch scoring
    also covers every package; comparing like with like keeps the drift baseline honest.

    Args:
        params: The full parameter dictionary.

    Returns:
        One row per package with the model's feature columns.
    """
    features = pd.read_parquet(params["train"]["features_path"])
    latest = params["snapshots"]["test"][-1]
    return features.loc[features["snapshot"] == latest, params["train"]["feature_columns"]]


def evaluate(params: dict[str, Any]) -> dict[str, float]:
    """Evaluate the trained model on the held-out splits and log the results.

    Args:
        params: The full parameter dictionary.

    Returns:
        All metrics, keyed by ``<split>_<scorer>_<metric>``.
    """
    run_info = read_json(Path(params["train"]["run_info_path"]))
    model = mlflow.pyfunc.load_model(run_info["model_uri"]).unwrap_python_model()
    splits = load_splits(params)
    top_k = params["evaluate"]["top_k_fraction"]

    metrics: dict[str, float] = {}
    for name in EVALUATED_SPLITS:
        metrics.update(evaluate_split(model, splits[name], name))
    reference_path = Path(params["train"]["reference_path"])
    write_json(reference_path, reference_profile(latest_population(params)))
    plots_dir = Path(params["evaluate"]["plots_dir"])
    write_plot_data(model, splits["test"], plots_dir)
    test = splits["test"]
    with mlflow.start_run(run_id=run_info["run_id"]):
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(plots_dir), artifact_path="evaluation")
        if len(np.unique(test.labels)) == 2:
            flagged = top_fraction_mask(model.risk(test.features), top_k).astype(int)
            with tempfile.TemporaryDirectory() as tmp:
                chart = plots.confusion_chart(
                    test.labels, flagged, Path(tmp) / "test_confusion.png"
                )
                mlflow.log_artifact(str(chart), artifact_path="evaluation")
        mlflow.log_artifact(str(reference_path), artifact_path="drift")
    return metrics


def main_body(params: dict[str, Any]) -> None:
    """Evaluate and write ``reports/metrics.json``.

    Args:
        params: The full parameter dictionary.
    """
    metrics = evaluate(params)
    write_json(Path(params["evaluate"]["metrics_path"]), metrics)
    logger.info("Evaluated model", extra=metrics)


if __name__ == "__main__":
    run_stage("evaluate", main_body)
