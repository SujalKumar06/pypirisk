"""Client the frontend uses to reach the backend API over REST."""

import re
from typing import Any

import httpx
from prometheus_client import Counter, Histogram

# PEP 508 project name: letters, digits, '.', '_' and '-', starting and ending alphanumeric.
# AI-generated (Claude Code). Prompt (condensed): "A regex that accepts a valid PEP 508
# project name and nothing else."
_PACKAGE_NAME = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?$")

API_CALLS = Counter(
    "frontend_api_calls_total", "Calls from the frontend to the API.", ["endpoint", "outcome"]
)
API_LATENCY = Histogram(
    "frontend_api_call_duration_seconds", "Time spent waiting for the API.", ["endpoint"]
)


class ApiError(RuntimeError):
    """Raised with a user-facing message when an API call does not succeed."""


def is_valid_package_name(name: str) -> bool:
    """Tell whether ``name`` is a syntactically valid PyPI project name.

    Args:
        name: User input.

    Returns:
        ``True`` if the name follows PEP 508.
    """
    return bool(_PACKAGE_NAME.match(name))


class ApiClient:
    """Calls the backend API; the base URL always comes from configuration."""

    def __init__(self, base_url: str, timeout: float) -> None:
        """Create a client.

        Args:
            base_url: API root URL.
            timeout: Request timeout in seconds.
        """
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def _request(self, endpoint: str, method: str, path: str, **kwargs: Any) -> Any:
        outcome = "error"
        try:
            with API_LATENCY.labels(endpoint).time():
                response = self._client.request(method, path, **kwargs)
            if response.status_code >= 400:
                outcome = str(response.status_code)
                detail = _detail(response)
                raise ApiError(detail)
            outcome = "ok"
            return response.json()
        except httpx.HTTPError as error:
            raise ApiError("The risk service is unavailable. Please try again later.") from error
        finally:
            API_CALLS.labels(endpoint, outcome).inc()

    def risk(self, package: str, live: bool = False, backend: str = "mlflow") -> dict:
        """Fetch one package's risk.

        Args:
            package: A valid package name.
            live: Score now through a model server instead of the batch score.
            backend: ``"mlflow"`` or ``"ray"`` for live scoring.

        Returns:
            The risk response.

        Raises:
            ApiError: If the package is unknown or the API fails.
        """
        params = {"live": str(live).lower(), "backend": backend}
        return self._request("risk", "GET", f"/packages/{package}/risk", params=params)

    def package(self, package: str) -> dict:
        """Fetch a package's risk with context and graph neighbours.

        Args:
            package: A valid package name.

        Returns:
            The package detail.

        Raises:
            ApiError: If the package is unknown or the API fails.
        """
        return self._request("package", "GET", f"/packages/{package}")

    def scan(self, requirements: str) -> dict:
        """Scan a project's requirements.

        Args:
            requirements: ``requirements.txt`` contents.

        Returns:
            The scan result.

        Raises:
            ApiError: If the file is rejected or the API fails.
        """
        return self._request("scan", "POST", "/projects/scan", json={"requirements": requirements})

    def watch(self, limit: int, min_dependents: int) -> list[dict]:
        """Fetch the Ecosystem Watch list.

        Args:
            limit: Rows to return.
            min_dependents: Only packages with at least this many dependents.

        Returns:
            Watch entries, highest priority first.

        Raises:
            ApiError: If the API fails.
        """
        params = {"limit": limit, "min_dependents": min_dependents}
        return self._request("watch", "GET", "/ecosystem/watch", params=params)

    def get(self, endpoint: str, path: str, **params: Any) -> Any:
        """GET a console endpoint (``/pipeline/...``, ``/models``).

        Args:
            endpoint: Metric label for the call.
            path: API path.
            **params: Query parameters; ``None`` values are left out.

        Returns:
            The decoded response.

        Raises:
            ApiError: If the API fails.
        """
        present = {key: value for key, value in params.items() if value is not None}
        return self._request(endpoint, "GET", path, params=present)


def _detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, list):
        detail = "; ".join(item.get("msg", str(item)) for item in detail)
    return str(detail or f"The service answered {response.status_code}.")
