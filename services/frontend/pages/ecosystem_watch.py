"""Ecosystem Watch page: widely used packages that most need support."""

import pandas as pd
import streamlit as st

from pypirisk_frontend.api_client import ApiError
from pypirisk_frontend.display import describe_factor
from pypirisk_frontend.session import api_client

st.title("Ecosystem Watch")
st.write(
    "Packages ranked by support priority: abandonment risk weighted by how central the "
    "package is in the dependency graph. A high rank means many projects would be "
    "affected, and maintainers may welcome help."
)

left, right = st.columns(2)
limit = left.slider("Packages to show", 10, 200, 50, step=10)
min_dependents = right.number_input("Minimum dependents", min_value=0, value=0, step=1)

try:
    entries = api_client().watch(limit, int(min_dependents))
except ApiError as error:
    st.error(str(error))
    st.stop()

if not entries:
    st.info("No packages match.")
    st.stop()

table = pd.DataFrame(
    [
        {
            "package": entry["package"],
            "grade": entry["grade"],
            "risk": round(entry["risk"], 3),
            "dependents": int(entry["dependents"]),
            "priority": round(entry["priority"], 3),
            "main reason": describe_factor(entry["top_factors"][0]) if entry["top_factors"] else "",
        }
        for entry in entries
    ]
)
st.bar_chart(table.head(20).set_index("package")["priority"])
st.dataframe(table, hide_index=True, width="stretch")
