"""Register stage: champion/challenger gate, promotion and post-deploy check.

Run with ``python -m pypirisk.models.register``. Every trained model is registered (for
the audit trail), but the serving alias moves only if the model:

1. beats the heuristic's test PR-AUC by ``min_gain_over_heuristic``, and
2. is not worse than the current champion's logged test PR-AUC by more than
   ``max_drop_vs_champion``.

The very first model is promoted without a gate so the platform has something to serve.
After promotion the model is loaded through the alias and must produce valid scores;
otherwise the alias is rolled back and the stage fails.
"""

import json
import logging
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from pypirisk.io import read_json, write_json
from pypirisk.models.registry import champion_version, promote, rollback
from pypirisk.stagelog import run_stage

logger = logging.getLogger("pypirisk.models.register")


class DeployCheckError(RuntimeError):
    """Raised when the promoted model fails the post-deploy check."""


def champion_test_pr_auc(client: MlflowClient, champion) -> float | None:
    """Return the test PR-AUC the current champion logged when it was evaluated.

    Args:
        client: MLflow client.
        champion: The champion's model version, or ``None``.

    Returns:
        The logged ``test_model_pr_auc``, or ``None`` if there is no champion or metric.
    """
    if champion is None:
        return None
    return client.get_run(champion.run_id).data.metrics.get("test_model_pr_auc")


def decide(
    metrics: dict[str, float],
    champion_score: float | None,
    has_champion: bool,
    settings: dict[str, Any],
) -> tuple[bool, str]:
    """Apply the promotion gate.

    Args:
        metrics: This model's evaluation metrics.
        champion_score: The champion's logged test PR-AUC, if known.
        has_champion: Whether a champion exists.
        settings: The ``register`` section of ``params.yaml``.

    Returns:
        Whether to promote, and why.
    """
    if not has_champion:
        return True, "bootstrap: no champion yet"
    new = metrics.get("test_model_pr_auc")
    heuristic = metrics.get("test_heuristic_pr_auc")
    if new is None:
        return False, "test PR-AUC undefined (one class in test split)"
    if heuristic is not None and new < heuristic + settings["min_gain_over_heuristic"]:
        return False, f"test PR-AUC {new:.4f} does not beat the heuristic's {heuristic:.4f}"
    if champion_score is not None and new < champion_score - settings["max_drop_vs_champion"]:
        return False, f"test PR-AUC {new:.4f} is worse than the champion's {champion_score:.4f}"
    return True, f"test PR-AUC {new:.4f} passes the gate"


def deploy_check(model_uri: str, rows: pd.DataFrame) -> None:
    """Load a model as the servers would and check that its output is usable.

    Args:
        model_uri: URI to load, normally ``models:/<name>@<alias>``.
        rows: Sample of serving features.

    Raises:
        DeployCheckError: If loading or scoring fails, or scores are not probabilities.
    """
    try:
        output = mlflow.pyfunc.load_model(model_uri).predict(rows)
    except Exception as error:  # noqa: BLE001 - any failure here means "do not serve"
        raise DeployCheckError(f"{model_uri} failed to score: {error}") from error
    risk = np.asarray(output["risk"], dtype=float)
    if len(risk) != len(rows) or not np.all(np.isfinite(risk)) or risk.min() < 0 or risk.max() > 1:
        raise DeployCheckError(f"{model_uri} returned invalid risk scores")
    json.loads(output["top_factors"].iloc[0])


def serving_sample(params: dict[str, Any]) -> pd.DataFrame:
    """Return feature rows from the scoring snapshot for the deploy check.

    Args:
        params: The full parameter dictionary.

    Returns:
        Up to ``deploy_check_rows`` rows of model inputs.
    """
    features = pd.read_parquet(params["train"]["features_path"])
    rows = features[features["snapshot"] == params["snapshots"]["scoring"]]
    sample = rows[params["train"]["feature_columns"]].head(params["register"]["deploy_check_rows"])
    return sample.astype(float)


def register(params: dict[str, Any]) -> dict[str, Any]:
    """Register the trained model, gate it, promote it and verify the deployment.

    Args:
        params: The full parameter dictionary.

    Returns:
        The registry entry written to ``reports/registry.json``.

    Raises:
        DeployCheckError: If the promoted model fails the check (after rolling back).
    """
    settings = params["register"]
    name, alias = settings["model_name"], settings["alias"]
    run_info = read_json(Path(params["train"]["run_info_path"]))
    metrics = read_json(Path(params["evaluate"]["metrics_path"]))
    client = MlflowClient()

    champion = champion_version(client, name, alias)
    has_champion = champion is not None
    champion_score = champion_test_pr_auc(client, champion)
    version = mlflow.register_model(run_info["model_uri"], name).version
    promoted, reason = decide(metrics, champion_score, has_champion, settings)
    client.set_model_version_tag(name, version, "gate", reason)
    client.set_model_version_tag(name, version, "profile", params["profile"])

    previous = None
    if promoted:
        previous = promote(client, name, alias, version)
        try:
            deploy_check(f"models:/{name}@{alias}", serving_sample(params))
        except DeployCheckError as error:
            if previous is not None:
                rollback(client, name, alias, str(error))
            raise
    champion = champion_version(client, name, alias)
    return {
        "model_name": name,
        "alias": alias,
        "version": str(version),
        "promoted": promoted,
        "reason": reason,
        "replaced_version": previous,
        "champion_version": str(champion.version) if champion else None,
        "champion_test_pr_auc_before": champion_score,
        "run_id": run_info["run_id"],
    }


def main_body(params: dict[str, Any]) -> None:
    """Register the model and write ``reports/registry.json``.

    Args:
        params: The full parameter dictionary.
    """
    entry = register(params)
    write_json(Path(params["register"]["registry_path"]), entry)
    logger.info("Registered model", extra=entry)


if __name__ == "__main__":
    run_stage("register", main_body)
