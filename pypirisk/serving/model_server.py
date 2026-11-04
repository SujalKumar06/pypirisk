"""MLflow model server with a Prometheus ``/metrics`` endpoint.

``mlflow models serve`` has no Prometheus endpoint, and every service must expose one.
This module builds the same FastAPI app ``mlflow models serve`` uses
(``mlflow.pyfunc.scoring_server.init``), so ``/invocations``, ``/ping``, ``/health`` and
``/version`` behave exactly as in MLflow, and adds request metrics and ``/metrics``.

As in Lecture 10, the server loads ``models:/<name>@<alias>`` when it starts; after a
promotion or a rollback moves the alias, restart it (``docker compose restart
model-server``) to serve the new version.

Configuration (environment variables):

- ``MLFLOW_TRACKING_URI``: tracking and registry server.
- ``MODEL_NAME`` / ``MODEL_ALIAS``: what to serve (default ``pypirisk-abandonment@champion``).
- ``MODEL_SERVER_HOST`` / ``MODEL_SERVER_PORT``: bind address (default ``0.0.0.0:8080``).
- ``MODEL_LOAD_RETRY_SECONDS``: wait between attempts while no model is registered yet.
"""

import logging
import os
import time

import mlflow
import uvicorn
from fastapi import FastAPI
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import PyFuncModel, scoring_server
from prometheus_client import Gauge

from pypirisk.logs import configure_logging
from pypirisk.monitoring.http import instrument_app

logger = logging.getLogger("pypirisk.serving.model_server")

SERVICE_NAME = "model-server"

MODEL_INFO = Gauge(
    "model_server_model_info",
    "The model currently served; the value is always 1.",
    ["model_uri", "run_id"],
)


def load_model_when_available(model_uri: str, retry_seconds: float) -> PyFuncModel:
    """Load a model, retrying until it exists in the registry.

    On a fresh stack the model server starts before the pipeline has registered any
    model, so it waits instead of crashing in a restart loop.

    Args:
        model_uri: MLflow model URI to load.
        retry_seconds: Pause between attempts.

    Returns:
        The loaded pyfunc model.
    """
    while True:
        try:
            return mlflow.pyfunc.load_model(model_uri)
        except (MlflowException, OSError) as error:
            logger.warning(
                "Model not available yet, retrying",
                extra={"model_uri": model_uri, "error": str(error), "retry_seconds": retry_seconds},
            )
            time.sleep(retry_seconds)


def create_app(model: PyFuncModel, model_uri: str) -> FastAPI:
    """Build the instrumented scoring app for ``model``.

    Args:
        model: The loaded pyfunc model.
        model_uri: URI the model was loaded from, exported as a metric label.

    Returns:
        MLflow's scoring FastAPI app with request metrics and ``/metrics``.
    """
    app = scoring_server.init(model)
    instrument_app(app, SERVICE_NAME)
    MODEL_INFO.labels(model_uri, model.metadata.run_id or "unknown").set(1)
    return app


def main() -> None:
    """Load the configured model and serve it with uvicorn."""
    configure_logging()
    name = os.environ.get("MODEL_NAME", "pypirisk-abandonment")
    alias = os.environ.get("MODEL_ALIAS", "champion")
    model_uri = f"models:/{name}@{alias}"
    model = load_model_when_available(
        model_uri, float(os.environ.get("MODEL_LOAD_RETRY_SECONDS", "15"))
    )
    logger.info("Model loaded", extra={"model_uri": model_uri, "run_id": model.metadata.run_id})
    uvicorn.run(
        create_app(model, model_uri),
        host=os.environ.get("MODEL_SERVER_HOST", "0.0.0.0"),
        port=int(os.environ.get("MODEL_SERVER_PORT", "8080")),
        log_config=None,
    )


if __name__ == "__main__":
    main()
