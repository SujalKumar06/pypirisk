"""Rollback demo: deploy a broken model and watch it roll back automatically.

``python -m pypirisk.ops.broken_model_demo`` registers a model that returns NaN risk,
promotes it to ``@champion`` (skipping the metric gate, as a bad manual deploy would),
then runs the same post-deploy check as the register stage. The check fails and the
alias moves back to the previous champion. Needs an existing champion.
"""

import argparse
import logging
import sys
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

from pypirisk.config import DEFAULT_PARAMS_PATH, load_params
from pypirisk.logs import configure_logging
from pypirisk.models.register import DeployCheckError, deploy_check, serving_sample
from pypirisk.models.registry import champion_version, promote, rollback
from pypirisk.models.train import model_signature

logger = logging.getLogger("pypirisk.ops.broken_model_demo")


class BrokenModel(mlflow.pyfunc.PythonModel):
    """A model that loads fine but returns NaN, like a bad preprocessing change."""

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        """Return NaN risk for every row."""
        return pd.DataFrame(
            {"risk": [float("nan")] * len(model_input), "top_factors": ["[]"] * len(model_input)}
        )


def run_demo(params: dict) -> dict:
    """Register, promote and check a broken model; roll back when the check fails.

    Args:
        params: The full parameter dictionary.

    Returns:
        The champion before, the broken version and the champion after.

    Raises:
        RuntimeError: If there is no champion to fall back to.
    """
    name, alias = params["register"]["model_name"], params["register"]["alias"]
    client = MlflowClient()
    before = champion_version(client, name, alias)
    if before is None:
        raise RuntimeError("Run the pipeline first: the demo needs a champion to fall back to")

    mlflow.set_experiment(params["train"]["experiment_name"])
    with mlflow.start_run(run_name="broken-model-demo"):
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=BrokenModel(),
            signature=model_signature(params["train"]["feature_columns"]),
        )
    broken = mlflow.register_model(info.model_uri, name).version
    client.set_model_version_tag(name, broken, "gate", "demo: promoted without the gate")
    promote(client, name, alias, broken)
    try:
        deploy_check(f"models:/{name}@{alias}", serving_sample(params))
    except DeployCheckError as error:
        rollback(client, name, alias, str(error))
    after = champion_version(client, name, alias)
    return {
        "champion_before": before.version,
        "broken_version": broken,
        "champion_after": after.version,
    }


def main() -> None:
    """Run the demo and exit non-zero if the rollback did not restore the champion."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    args = parser.parse_args()
    configure_logging()
    outcome = run_demo(load_params(args.params))
    logger.info("Rollback demo finished", extra=outcome)
    if outcome["champion_after"] != outcome["champion_before"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
