"""Smoke test against a running Docker Compose stack: ``pytest -m e2e``.

Service URLs default to the host ports in ``docker-compose.yml`` and can be overridden
with environment variables.
"""

import os

import httpx
import pytest

pytestmark = pytest.mark.e2e

API = os.environ.get("API_URL", "http://localhost:8000")
HEALTH_URLS = {
    "api": f"{API}/health",
    "model-server": os.environ.get("MODEL_SERVER_URL", "http://localhost:5001") + "/health",
    "ray-serve": os.environ.get("RAY_SERVE_URL", "http://localhost:8002") + "/health",
    "mlflow": os.environ.get("MLFLOW_URL", "http://localhost:5000") + "/health",
    # Streamlit's own health endpoint; the frontend's /metrics side server starts with the
    # first browser session.
    "frontend": os.environ.get("FRONTEND_URL", "http://localhost:8501") + "/_stcore/health",
    "prometheus": os.environ.get("PROMETHEUS_URL", "http://localhost:9090") + "/-/healthy",
    "alertmanager": os.environ.get("ALERTMANAGER_URL", "http://localhost:9093") + "/-/healthy",
    "grafana": os.environ.get("GRAFANA_URL", "http://localhost:3000") + "/api/health",
}


@pytest.mark.parametrize("service", sorted(HEALTH_URLS))
def test_service_is_healthy(service):
    assert httpx.get(HEALTH_URLS[service], timeout=10).status_code == 200


@pytest.mark.parametrize("backend", ["mlflow", "ray"])
def test_live_prediction_goes_through_each_serving_path(backend):
    response = httpx.get(
        f"{API}/packages/requests/risk", params={"live": "true", "backend": backend}, timeout=30
    )
    assert response.status_code == 200
    assert 0.0 <= response.json()["risk"] <= 1.0


def test_project_scan_returns_a_grade():
    response = httpx.post(f"{API}/projects/scan", json={"requirements": "requests\n"}, timeout=30)
    assert response.status_code == 200
    assert response.json()["grade"] in "ABCDF"
