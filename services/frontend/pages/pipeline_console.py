"""Pipeline Management Console: DAG status, runs and errors, throughput, models, actions."""

from datetime import datetime

import pandas as pd
import streamlit as st

from pypirisk_frontend.api_client import ApiError
from pypirisk_frontend.session import api_client, public_links

STATUS_COLOURS = {"success": "#2e7d32", "failed": "#c62828", None: "#9e9e9e"}

st.title("Pipeline Console")
links = public_links()
if links:
    st.write(" · ".join(f"[{name}]({url})" for name, url in links.items()))

client = api_client()


def dag_dot(dag: dict) -> str:
    """Draw the DVC stages coloured by the status of their last run."""
    lines = [
        "digraph G {",
        "rankdir=LR;",
        'node [shape=box, style="rounded,filled", fontcolor="white"];',
    ]
    for stage in dag["stages"]:
        status = (stage.get("last_run") or {}).get("status")
        lines.append(f'"{stage["name"]}" [fillcolor="{STATUS_COLOURS.get(status, "#9e9e9e")}"];')
    lines.extend(f'"{edge["from"]}" -> "{edge["to"]}";' for edge in dag["edges"])
    return "\n".join([*lines, "}"])


try:
    dag = client.get("dag", "/pipeline/dag")
    reports = client.get("reports", "/pipeline/reports")
except ApiError as error:
    st.error(str(error))
    st.stop()

st.subheader("DVC pipeline")
st.graphviz_chart(dag_dot(dag))
st.caption("Green: last run succeeded · red: last run failed · grey: not run yet")
stage_rows = []
for stage in dag["stages"]:
    run = stage.get("last_run") or {}
    stage_rows.append(
        {
            "stage": stage["name"],
            "last status": run.get("status", "never run"),
            "started": run.get("started_at", ""),
            "duration (s)": run.get("duration_seconds"),
            "profile": run.get("profile", ""),
        }
    )
st.dataframe(pd.DataFrame(stage_rows), hide_index=True, width="stretch")

st.subheader("Throughput and data coverage")
ingest = reports.get("ingest") or {}
validation = (reports.get("validation") or {}).get("summary", {})
crawl = ingest.get("crawl", {})
cols = st.columns(4)
cols[0].metric("Packages crawled", ingest.get("coverage", {}).get("packages", "–"))
cols[1].metric("Crawl throughput (packages/min)", crawl.get("packages_per_minute", "–"))
cols[2].metric("Validation throughput (records/s)", validation.get("records_per_second", "–"))
cols[3].metric(
    "Rows rejected", f"{validation.get('invalid_fraction', 0):.2%}" if validation else "–"
)
if ingest.get("coverage"):
    st.json(ingest["coverage"], expanded=False)

st.subheader("Runs")
show_failed = st.toggle("Only failed runs")
try:
    runs = client.get("runs", "/pipeline/runs", limit=100, status="failed" if show_failed else None)
except ApiError as error:
    st.error(str(error))
    runs = []
if runs:
    st.dataframe(
        pd.DataFrame(runs)[["started_at", "stage", "status", "duration_seconds", "profile"]],
        hide_index=True,
        width="stretch",
    )
    for run in [run for run in runs if run["status"] == "failed"][:10]:
        with st.expander(f"❌ {run['stage']} at {run['started_at']}: {run.get('error', '')[:120]}"):
            st.code(run.get("traceback", "no traceback recorded"))
else:
    st.info("No runs recorded yet.")

st.subheader("Models")
metrics = reports.get("metrics") or {}
registry = reports.get("registry") or {}
cols = st.columns(4)
cols[0].metric(
    "Test ROC-AUC (model)",
    f"{metrics['test_model_roc_auc']:.3f}" if "test_model_roc_auc" in metrics else "–",
)
cols[1].metric(
    "Test PR-AUC (model)",
    f"{metrics['test_model_pr_auc']:.3f}" if "test_model_pr_auc" in metrics else "–",
)
cols[2].metric(
    "Test PR-AUC (heuristic)",
    f"{metrics['test_heuristic_pr_auc']:.3f}" if "test_heuristic_pr_auc" in metrics else "–",
)
cols[3].metric("Champion version", registry.get("champion_version", "–"))
if registry:
    st.caption(f"Last registration: version {registry.get('version')}, {registry.get('reason')}")
try:
    models = client.get("models", "/models")
    versions = pd.DataFrame(
        [
            {
                "version": version["version"],
                "aliases": ", ".join(version["aliases"]),
                "created": version["created_at"],
                "gate": version["tags"].get("gate", ""),
                "rolled back": version["tags"].get("rolled_back", ""),
            }
            for version in models["versions"]
        ]
    )
    st.dataframe(versions, hide_index=True, width="stretch")
except ApiError as error:
    st.warning(f"Model registry unavailable: {error}")

st.subheader("How to act")
st.markdown(
    "- **Retrain:** run `dvc repro` (the DVC pipeline: lint, tests, then the stages that "
    "changed).\n"
    "- **Roll back:** in MLflow, move the `champion` alias to the previous version, then "
    "`docker compose restart model-server`.\n"
    "- A model that fails the post-deploy check is rolled back automatically by the "
    "`register` stage."
)
st.caption(f"Page refreshed {datetime.now():%H:%M:%S}")
