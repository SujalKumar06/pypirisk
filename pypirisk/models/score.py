"""Score stage (batch scoring): ``python -m pypirisk.models.score``.

Scores every package at the scoring snapshot with the current champion and writes the
files the API serves from ``data/serving``:

| File | Contents |
| --- | --- |
| ``scores.parquet`` | risk, grade, top factors and key features per package |
| ``features.parquet`` | model inputs, for the live serving path |
| ``edges.parquet`` | the scoring snapshot's dependency graph, for project scans |
| ``reference.json`` | the training distribution, for drift checks |
| ``model.json`` | which model version produced the scores |
"""

import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow import MlflowClient

from pypirisk.grades import grade
from pypirisk.io import read_json, write_json
from pypirisk.models.registry import champion_version
from pypirisk.monitoring.drift import feature_drift
from pypirisk.stagelog import run_stage

logger = logging.getLogger("pypirisk.models.score")

CONTEXT_COLUMNS = [
    "months_since_last_release",
    "releases_in_lookback",
    "commits_in_lookback",
    "bus_factor",
    "dependents",
    "pagerank",
    "repo_url",
]


class NoChampionError(RuntimeError):
    """Raised when there is no promoted model to score with."""


def score_rows(
    model: mlflow.pyfunc.PyFuncModel,
    rows: pd.DataFrame,
    columns: list[str],
    bands: dict[str, float],
) -> pd.DataFrame:
    """Score feature rows and attach grades and context columns.

    Args:
        model: The champion model.
        rows: Scoring-snapshot feature rows.
        columns: The model's feature columns.
        bands: Grade bands.

    Returns:
        One row per package with ``risk``, ``grade``, ``top_factors`` and context.
    """
    output = model.predict(rows[columns].astype(float))
    scores = rows[["name", "snapshot", *CONTEXT_COLUMNS]].reset_index(drop=True)
    scores["risk"] = output["risk"].to_numpy()
    scores["grade"] = [grade(risk, bands) for risk in scores["risk"]]
    scores["top_factors"] = output["top_factors"].to_numpy()
    return scores


def score(params: dict[str, Any]) -> None:
    """Score the scoring snapshot with the champion and write the serving files.

    Args:
        params: The full parameter dictionary.

    Raises:
        NoChampionError: If no model has been promoted.
    """
    settings = params["register"]
    name, alias = settings["model_name"], settings["alias"]
    champion = champion_version(MlflowClient(), name, alias)
    if champion is None:
        raise NoChampionError(f"{name}@{alias} does not exist; nothing to score with")
    model = mlflow.pyfunc.load_model(f"models:/{name}@{alias}")

    columns = params["train"]["feature_columns"]
    snapshot = params["snapshots"]["scoring"]
    features = pd.read_parquet(params["train"]["features_path"])
    rows = features[features["snapshot"] == snapshot].reset_index(drop=True)
    scores = score_rows(model, rows, columns, params["score"]["grades"])

    output = Path(params["score"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(output / "scores.parquet", index=False)
    rows[["name", "snapshot", *columns]].to_parquet(output / "features.parquet", index=False)
    edges = pd.read_parquet(Path(params["graph"]["output_dir"]) / "edges.parquet")
    edges[edges["snapshot"] == snapshot].to_parquet(output / "edges.parquet", index=False)
    shutil.copyfile(params["train"]["reference_path"], output / "reference.json")

    drift = feature_drift(read_json(Path(params["train"]["reference_path"])), rows)
    write_json(
        output / "model.json",
        {
            "model_name": name,
            "alias": alias,
            "version": champion.version,
            "run_id": champion.run_id,
            "snapshot": snapshot,
            "scored_at": datetime.now(UTC).isoformat(),
            "packages": len(scores),
            "max_feature_drift": max(drift.values(), default=0.0),
        },
    )
    logger.info("Scored packages", extra={"packages": len(scores), "version": champion.version})


if __name__ == "__main__":
    run_stage("score", score)
