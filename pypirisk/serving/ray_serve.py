"""Second serving path: the champion model on Ray Serve.

Serves the same registered model with the same contract as the MLflow model server
(``POST /invocations`` with ``dataframe_records``, ``GET /health``, ``GET /metrics``), so
the API can switch backends with one environment variable. Ray Serve adds replica
autoscaling and request batching that MLflow Serve lacks.

Run with ``python -m pypirisk.serving.ray_serve``. Configuration: ``MLFLOW_TRACKING_URI``,
``MODEL_NAME``, ``MODEL_ALIAS``, ``RAY_SERVE_PORT`` (default 8000), ``RAY_SERVE_REPLICAS``
(default 1).
"""

import json
import logging
import os
import signal
import threading

import mlflow
import pandas as pd
import ray
from fastapi import FastAPI, HTTPException, Request
from ray import serve

from pypirisk.logs import configure_logging
from pypirisk.monitoring.http import instrument_app

logger = logging.getLogger("pypirisk.serving.ray_serve")

SERVICE_NAME = "ray-serve"

app = FastAPI(title="pypirisk Ray Serve")
instrument_app(app, SERVICE_NAME)


def records_to_frame(payload: dict) -> pd.DataFrame:
    """Accept the MLflow scoring formats ``dataframe_records`` and ``dataframe_split``.

    Args:
        payload: The decoded request body.

    Returns:
        The input rows.

    Raises:
        ValueError: If neither format is present.
    """
    if "dataframe_records" in payload:
        return pd.DataFrame(payload["dataframe_records"])
    if "dataframe_split" in payload:
        split = payload["dataframe_split"]
        return pd.DataFrame(split["data"], columns=split["columns"])
    raise ValueError("Body must contain dataframe_records or dataframe_split")


@serve.deployment(num_replicas=int(os.environ.get("RAY_SERVE_REPLICAS", "1")))
@serve.ingress(app)
class RiskDeployment:
    """Ray Serve deployment holding the champion model."""

    def __init__(self, model_uri: str) -> None:
        """Load the model once per replica.

        Args:
            model_uri: Registry URI, e.g. ``models:/pypirisk-abandonment@champion``.
        """
        self.model_uri = model_uri
        self.model = mlflow.pyfunc.load_model(model_uri)

    @app.get("/health")
    def health(self) -> dict:
        """Report that the replica has a model loaded."""
        return {"status": "ok", "model_uri": self.model_uri}

    @app.post("/invocations")
    async def invocations(self, request: Request) -> dict:
        """Score rows exactly like MLflow's ``/invocations``.

        Raises:
            HTTPException: 400 for a malformed body.
        """
        try:
            frame = records_to_frame(await request.json())
            output = self.model.predict(frame)
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"predictions": output.to_dict(orient="records")}


def main() -> None:
    """Start Ray, deploy the champion and block until the container is stopped."""
    configure_logging()
    name = os.environ.get("MODEL_NAME", "pypirisk-abandonment")
    alias = os.environ.get("MODEL_ALIAS", "champion")
    ray.init(include_dashboard=False, _temp_dir=os.environ.get("RAY_TMPDIR") or None)
    serve.start(
        http_options={"host": "0.0.0.0", "port": int(os.environ.get("RAY_SERVE_PORT", "8000"))}
    )
    serve.run(RiskDeployment.bind(f"models:/{name}@{alias}"), route_prefix="/")
    logger.info("Ray Serve deployment running", extra={"model": f"{name}@{alias}"})
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    stop.wait()
    serve.shutdown()


if __name__ == "__main__":
    main()
