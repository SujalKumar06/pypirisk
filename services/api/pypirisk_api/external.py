"""Clients for the services the API consults: the MLflow registry and PyPI."""

import logging
from datetime import UTC, datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class UpstreamError(RuntimeError):
    """Raised when a service the API depends on fails or is not configured."""


class RegistryClient:
    """Read model versions and aliases through MLflow's REST API.

    Using REST keeps the API image free of the MLflow package.
    """

    def __init__(self, base_url: str | None, timeout: float) -> None:
        """Create a client.

        Args:
            base_url: MLflow tracking server URL, or ``None`` if not configured.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url
        self.timeout = timeout

    def _get(self, endpoint: str, params: dict) -> dict:
        if not self.base_url:
            raise UpstreamError("MLFLOW_URL is not configured")
        try:
            response = httpx.get(
                f"{self.base_url}/api/2.0/mlflow/{endpoint}", params=params, timeout=self.timeout
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise UpstreamError(f"MLflow request failed: {error}") from error
        return response.json()

    def versions(self, name: str) -> list[dict[str, Any]]:
        """List a registered model's versions with their aliases and tags.

        Args:
            name: Registered model name.

        Returns:
            Versions, newest first.

        Raises:
            UpstreamError: If MLflow is not configured or the request fails.
        """
        versions = self._get(
            "model-versions/search", {"filter": f"name='{name}'", "max_results": 100}
        ).get("model_versions", [])
        # The version search omits aliases; the registered model lists them.
        model = self._get("registered-models/get", {"name": name}).get("registered_model", {})
        aliases: dict[str, list[str]] = {}
        for entry in model.get("aliases", []):
            aliases.setdefault(str(entry["version"]), []).append(entry["alias"])
        rows = [
            {
                "version": version["version"],
                "run_id": version.get("run_id"),
                "created_at": datetime.fromtimestamp(
                    int(version["creation_timestamp"]) / 1000, UTC
                ).isoformat(),
                "aliases": aliases.get(str(version["version"]), []),
                "tags": {tag["key"]: tag["value"] for tag in version.get("tags", [])},
            }
            for version in versions
        ]
        return sorted(rows, key=lambda row: int(row["version"]), reverse=True)


def heuristic_fallback(pypi_url: str, package: str, timeout: float) -> float | None:
    """Months since a package's last release, straight from PyPI.

    Used for packages outside the scored universe: a live crawl of commits and issues
    cannot finish within a request, but this one call can.

    Args:
        pypi_url: PyPI JSON API root.
        package: Package name.
        timeout: Request timeout in seconds.

    Returns:
        Months since the last upload, or ``None`` if PyPI cannot answer.
    """
    try:
        response = httpx.get(f"{pypi_url}/{package}/json", timeout=timeout)
        response.raise_for_status()
        uploads = [
            file["upload_time_iso_8601"]
            for files in response.json()["releases"].values()
            for file in files
        ]
    except (httpx.HTTPError, KeyError, ValueError) as error:
        logger.warning("Heuristic fallback failed", extra={"package": package, "error": str(error)})
        return None
    if not uploads:
        return None
    last = datetime.fromisoformat(max(uploads).replace("Z", "+00:00"))
    return (datetime.now(UTC) - last).days / (365.25 / 12)
