"""Train stage: fit the candidates, pick one, calibrate it and log everything to MLflow.

Run with ``python -m pypirisk.models.train``. The tracking server comes from
``MLFLOW_TRACKING_URI``. Besides MLflow's usual parameters and metrics, every run logs
the Git commit, the data hashes, label prevalence, crawl coverage, feature statistics
and feature importance.
"""

import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm
import mlflow
import numpy as np
import pandas as pd
import sklearn
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

import pypirisk
from pypirisk.io import write_json
from pypirisk.models import plots
from pypirisk.models.estimators import build_candidate, join_features_and_labels
from pypirisk.models.explain import mean_absolute_shap, shap_values
from pypirisk.models.metrics import ranking_metrics
from pypirisk.models.pyfunc import RiskModel
from pypirisk.stagelog import run_stage
from pypirisk.tracking import lineage_tags

logger = logging.getLogger("pypirisk.models.train")


@dataclass
class Split:
    """Feature matrix and labels of one split.

    Attributes:
        rows: The full labelled rows.
        features: Model inputs in training column order.
        labels: 1 = abandoned.
    """

    rows: pd.DataFrame
    features: pd.DataFrame
    labels: np.ndarray


def load_splits(params: dict[str, Any]) -> dict[str, Split]:
    """Load features and labels and split them by snapshot.

    Args:
        params: The full parameter dictionary.

    Returns:
        ``train``, ``validation`` and ``test`` splits.

    Raises:
        ValueError: If the training split lacks one of the classes.
    """
    stage = params["train"]
    columns = stage["feature_columns"]
    data = join_features_and_labels(
        pd.read_parquet(stage["features_path"]), pd.read_parquet(stage["labels_path"])
    )
    splits = {}
    for name in ("train", "validation", "test"):
        rows = data[data["split"] == name].reset_index(drop=True)
        splits[name] = Split(rows, rows[columns].astype(float), rows["label"].to_numpy())
    if len(np.unique(splits["train"].labels)) < 2:
        raise ValueError("The training split needs both abandoned and active packages")
    return splits


def select_candidate(scores: dict[str, float | None], order: list[str]) -> tuple[str, str]:
    """Pick the candidate with the best validation score.

    Args:
        scores: Validation score per candidate (``None`` when undefined).
        order: Candidates in ``params.yaml`` order, the fallback when no score exists.

    Returns:
        The chosen candidate and the reason, for the MLflow tags.
    """
    defined = {kind: score for kind, score in scores.items() if score is not None}
    if not defined:
        return order[0], "validation metric undefined (one class); first candidate used"
    best = max(defined, key=defined.get)
    return best, f"best validation score {defined[best]:.4f}"


def calibrate(model: ClassifierMixin, split: Split, method: str) -> tuple[ClassifierMixin, str]:
    """Calibrate a fitted model on the validation split.

    Args:
        model: The fitted candidate.
        split: Validation split.
        method: ``"isotonic"`` or ``"sigmoid"``.

    Returns:
        The calibrated classifier and a status for the MLflow tags. If the validation
        split has one class, calibration is impossible and the model is used as is.
    """
    if len(np.unique(split.labels)) < 2:
        logger.warning("Calibration skipped: the validation split has one class")
        return FrozenEstimator(model), "skipped: one class in validation"
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=method)
    return calibrated.fit(split.features, split.labels), method


def _coverage_metrics() -> dict[str, float]:
    path = Path("reports/ingest.json")
    if not path.exists():
        return {}
    coverage = json.loads(path.read_text())["coverage"]
    return {f"coverage_{key}": float(value) for key, value in coverage.items()}


def _split_metrics(splits: dict[str, Split]) -> dict[str, float]:
    metrics = {}
    for name, split in splits.items():
        metrics[f"{name}_rows"] = float(len(split.labels))
        metrics[f"{name}_prevalence"] = float(split.labels.mean()) if len(split.labels) else 0.0
    return metrics


def fit_candidates(
    splits: dict[str, Split], params: dict[str, Any]
) -> dict[str, tuple[ClassifierMixin, dict[str, float]]]:
    """Fit every candidate on train and score it on validation, one nested run each.

    Args:
        splits: Train, validation and test splits.
        params: The full parameter dictionary.

    Returns:
        Fitted model and validation metrics per candidate.
    """
    stage = params["train"]
    results = {}
    for kind in stage["candidates"]:
        settings = stage[kind]
        model = build_candidate(kind, settings, stage["random_seed"]).fit(
            splits["train"].features, splits["train"].labels
        )
        scores = model.predict_proba(splits["validation"].features)[:, 1]
        metrics = ranking_metrics(splits["validation"].labels, scores, "validation")
        with mlflow.start_run(run_name=kind, nested=True):
            mlflow.log_params({f"{kind}.{key}": value for key, value in settings.items()})
            mlflow.log_metrics(metrics)
        results[kind] = (model, metrics)
    return results


def _log_feature_reports(kind: str, model: ClassifierMixin, splits: dict[str, Split]) -> None:
    # The files only need to exist long enough to be uploaded to MLflow.
    with tempfile.TemporaryDirectory() as tmp:
        _write_and_log_feature_reports(kind, model, splits, Path(tmp))


def _write_and_log_feature_reports(
    kind: str, model: ClassifierMixin, splits: dict[str, Split], workdir: Path
) -> None:
    train = splits["train"].features
    stats = train.describe().T.assign(missing_share=train.isna().mean())
    stats.to_csv(workdir / "feature_stats.csv")
    sample = splits["validation"].features if len(splits["validation"].labels) else train
    importance = mean_absolute_shap(shap_values(kind, model, sample))
    importance.rename("mean_abs_shap").to_csv(workdir / "feature_importance.csv")
    plots.feature_importance_chart(importance, workdir / "feature_importance.png")
    for name in ("feature_stats.csv", "feature_importance.csv", "feature_importance.png"):
        mlflow.log_artifact(str(workdir / name), artifact_path="features")
    mlflow.log_metrics({f"missing_share.{c}": float(v) for c, v in train.isna().mean().items()})


def model_signature(columns: list[str]) -> ModelSignature:
    """Declare the served model's input and output schema.

    Every feature is an optional float: activity and issue features are legitimately
    missing for packages without a cloned repository or issue history, and an inferred
    schema would wrongly mark them required.

    Args:
        columns: Feature columns in training order.

    Returns:
        The signature: optional doubles in, ``risk`` (double) and ``top_factors``
        (JSON string) out.
    """
    return ModelSignature(
        inputs=Schema([ColSpec("double", column, required=False) for column in columns]),
        outputs=Schema([ColSpec("double", "risk"), ColSpec("string", "top_factors")]),
    )


def train_and_log(params: dict[str, Any]) -> dict[str, str]:
    """Train, select, calibrate and log the model with full lineage.

    Args:
        params: The full parameter dictionary.

    Returns:
        ``run_id``, ``model_uri``, ``experiment_name`` and the chosen ``candidate``.
    """
    stage = params["train"]
    columns = stage["feature_columns"]
    splits = load_splits(params)

    mlflow.set_experiment(stage["experiment_name"])
    with mlflow.start_run(run_name=f"train-{params['profile']}") as run:
        mlflow.set_tags(
            {
                **lineage_tags(
                    {"features": Path(stage["features_path"]), "labels": Path(stage["labels_path"])}
                ),
                "profile": params["profile"],
            }
        )
        mlflow.log_params(
            {
                "feature_columns": ",".join(columns),
                "candidates": ",".join(stage["candidates"]),
                "calibration_method": stage["calibration_method"],
                "train_snapshots": ",".join(params["snapshots"]["train"]),
                "outcome_months": params["labels"]["outcome_months"],
                "max_commits_in_outcome": params["labels"]["max_commits_in_outcome"],
                "random_seed": stage["random_seed"],
            }
        )
        mlflow.log_metrics({**_split_metrics(splits), **_coverage_metrics()})

        candidates = fit_candidates(splits, params)
        metric = stage["selection_metric"]
        kind, reason = select_candidate(
            {name: metrics.get(metric) for name, (_, metrics) in candidates.items()},
            stage["candidates"],
        )
        model = candidates[kind][0]
        calibrated, calibration = calibrate(
            model, splits["validation"], stage["calibration_method"]
        )
        mlflow.set_tags({"candidate": kind, "selection": reason, "calibration": calibration})
        _log_feature_reports(kind, model, splits)

        risk_model = RiskModel(kind, model, calibrated, columns)
        example = splits["train"].features.head(5)
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=risk_model,
            # Ship the package with the model so any server can unpickle RiskModel.
            code_paths=[str(Path(pypirisk.__file__).parent)],
            signature=model_signature(columns),
            input_example=example,
            pip_requirements=[
                f"mlflow=={mlflow.__version__}",
                f"scikit-learn=={sklearn.__version__}",
                f"lightgbm=={lightgbm.__version__}",
                f"pandas=={pd.__version__}",
                f"numpy=={np.__version__}",
            ],
        )
    return {
        "run_id": run.info.run_id,
        "model_uri": model_info.model_uri,
        "experiment_name": stage["experiment_name"],
        "candidate": kind,
    }


def train(params: dict[str, Any]) -> None:
    """Train the model and write the run reference used by later stages.

    Args:
        params: The full parameter dictionary.
    """
    run_info = train_and_log(params)
    write_json(Path(params["train"]["run_info_path"]), run_info)
    logger.info("Trained model", extra=run_info)


if __name__ == "__main__":
    run_stage("train", train)
