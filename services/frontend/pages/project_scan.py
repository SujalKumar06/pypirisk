"""Project Scan page: upload requirements.txt and see the project's dependency risk."""

import pandas as pd
import streamlit as st

from pypirisk_frontend.api_client import ApiError
from pypirisk_frontend.display import dot_graph
from pypirisk_frontend.session import api_client

MAX_UPLOAD_BYTES = 100_000
MAX_GRAPH_NODES = 80

st.title("Project Scan")
st.write(
    "Upload a `requirements.txt` (or paste package names, one per line). The scan follows "
    "the dependency graph and shows which dependencies put the project at risk."
)

upload = st.file_uploader("requirements.txt", type=["txt"])
pasted = st.text_area("…or paste requirements", height=120, placeholder="requests\nflask>=3\n")

text = ""
if upload is not None:
    raw = upload.getvalue()
    if len(raw) > MAX_UPLOAD_BYTES:
        st.error(f"The file is larger than {MAX_UPLOAD_BYTES // 1000} KB.")
        st.stop()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        st.error("The file is not UTF-8 text.")
        st.stop()
elif pasted.strip():
    text = pasted

if text and st.button("Scan project", type="primary"):
    try:
        result = api_client().scan(text)
    except ApiError as error:
        st.error(str(error))
        st.stop()

    a, b, c = st.columns(3)
    a.metric(
        "Project grade",
        result["grade"],
        help="Weakest link: the riskiest dependency, discounted by depth",
    )
    b.metric("Expected dependencies abandoned", f"{result['expected_abandoned']:.1f}")
    c.metric("Dependencies reached", result["dependencies_total"])
    for error in result["parse_errors"]:
        st.warning(f"Skipped {error}")

    st.subheader("Riskiest dependency paths")
    rows = [
        {
            "package": item["package"],
            "grade": item["grade"],
            "risk": round(item["risk"], 3),
            "depth": item["depth"],
            "path": " → ".join(item["path"]),
        }
        for item in result["contributions"][:15]
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    if result["unscored"]:
        st.subheader("Not scored yet")
        st.caption(
            "These are outside the scored package set. "
            "The heuristic shows how long since their last release."
        )
        st.dataframe(pd.DataFrame(result["unscored"]), hide_index=True)

    edges = [tuple(edge) for edge in result["edges"]]
    nodes = {name for edge in edges for name in edge} | set(result["declared"])
    if len(nodes) <= MAX_GRAPH_NODES:
        st.subheader("Dependency graph")
        grades = {item["package"]: item["grade"] for item in result["contributions"]}
        st.graphviz_chart(dot_graph(edges, grades, set(result["declared"])))
    else:
        st.caption(f"The graph has {len(nodes)} packages; too many to draw clearly.")
