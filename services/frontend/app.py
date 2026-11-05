"""Streamlit frontend: four pages that reach the backend only through ``API_URL``.

Run with ``streamlit run services/frontend/app.py``.
"""

import os

import streamlit as st

from pypirisk.logs import configure_logging
from pypirisk_frontend.ops_server import start_ops_server


@st.cache_resource
def ops_server():
    """Start the ``/health`` and ``/metrics`` server once per process, not once per rerun."""
    configure_logging()
    return start_ops_server(int(os.environ.get("OPS_PORT", "9101")))


def main() -> None:
    """Configure the app and route to the selected page."""
    ops_server()
    st.set_page_config(page_title="pypirisk", page_icon="📦", layout="wide")
    if not os.environ.get("API_URL"):
        st.error("API_URL is not configured.")
        return
    pages = [
        st.Page("pages/package_lookup.py", title="Package Lookup", icon="🔎", default=True),
        st.Page("pages/project_scan.py", title="Project Scan", icon="📋"),
        st.Page("pages/ecosystem_watch.py", title="Ecosystem Watch", icon="🌐"),
        st.Page("pages/pipeline_console.py", title="Pipeline Console", icon="🛠️"),
    ]
    st.navigation(pages).run()


main()
