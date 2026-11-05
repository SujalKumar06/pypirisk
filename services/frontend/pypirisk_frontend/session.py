"""Per-process resources shared by the pages."""

import os

import streamlit as st

from pypirisk_frontend.api_client import ApiClient


@st.cache_resource
def api_client() -> ApiClient:
    """Create the API client once per process; the URL comes only from ``API_URL``."""
    return ApiClient(os.environ["API_URL"], float(os.environ.get("API_TIMEOUT_SECONDS", "30")))


def public_links() -> dict[str, str]:
    """Browser-facing links to the operations tools, from environment variables.

    Returns:
        Tool name mapped to URL, for the tools that are configured.
    """
    names = {
        "MLflow": "MLFLOW_PUBLIC_URL",
        "Grafana": "GRAFANA_PUBLIC_URL",
        "Prometheus": "PROMETHEUS_PUBLIC_URL",
        "Alertmanager": "ALERTMANAGER_PUBLIC_URL",
    }
    return {name: os.environ[var] for name, var in names.items() if os.environ.get(var)}
