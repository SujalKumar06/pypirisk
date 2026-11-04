"""HTTP client for a model server (MLflow Serve or Ray Serve; same contract)."""

import json
import math
import time

import httpx
from prometheus_client import Histogram

MODEL_SERVER_LATENCY = Histogram(
    "api_model_server_request_duration_seconds",
    "Time the API waits for a model server.",
    ["backend", "outcome"],
)


class ModelServerError(RuntimeError):
    """Raised when a model server cannot be reached or returns a bad response."""


class ModelServerClient:
    """Calls ``POST /invocations`` and ``GET /health`` on a model server."""

    def __init__(self, base_url: str, timeout: float, backend: str = "mlflow") -> None:
        """Create a client.

        Args:
            base_url: Model server root URL, without a trailing slash.
            timeout: Request timeout in seconds.
            backend: Label for metrics, e.g. ``"mlflow"`` or ``"ray"``.
        """
        self.backend = backend
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def predict(self, features: dict[str, float | None]) -> tuple[float, list[dict]]:
        """Score one package live.

        Args:
            features: One row of model inputs (``None`` for missing values).

        Returns:
            The probability of abandonment and its top factors.

        Raises:
            ModelServerError: On connection errors, error statuses or an unexpected body.
        """
        started = time.perf_counter()
        outcome = "error"
        try:
            response = self._client.post(
                "/invocations", json={"dataframe_records": [_json_safe(features)]}
            )
            response.raise_for_status()
            prediction = response.json()["predictions"][0]
            risk = float(prediction["risk"])
            factors = json.loads(prediction["top_factors"])
            outcome = "ok"
            return risk, factors
        except httpx.HTTPError as error:
            raise ModelServerError(f"Model server request failed: {error}") from error
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ModelServerError(f"Unexpected model server response: {error}") from error
        finally:
            MODEL_SERVER_LATENCY.labels(self.backend, outcome).observe(
                time.perf_counter() - started
            )

    def is_healthy(self) -> bool:
        """Tell whether the model server answers its health check.

        Returns:
            ``True`` if ``GET /health`` returns 200.
        """
        try:
            return self._client.get("/health").status_code == 200
        except httpx.HTTPError:
            return False


def _json_safe(features: dict[str, float | None]) -> dict[str, float | None]:
    return {
        key: None if value is None or (isinstance(value, float) and math.isnan(value)) else value
        for key, value in features.items()
    }
