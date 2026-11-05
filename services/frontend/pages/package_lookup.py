"""Package Lookup page: one package's risk, its reasons and its neighbours."""

import streamlit as st

from pypirisk_frontend.api_client import ApiError, is_valid_package_name
from pypirisk_frontend.display import GRADE_COLOURS, describe_factor, number
from pypirisk_frontend.session import api_client

st.title("Package Lookup")
st.write(
    "How likely is a PyPI package to stop being maintained in the next 18 months? "
    "This is a sustainability signal to help decide where support is needed, "
    "not a judgement of its maintainers."
)

left, right = st.columns([3, 2])
package = left.text_input("Package name", placeholder="e.g. requests").strip()
live = right.toggle(
    "Score live through a model server", help="Otherwise the latest batch score is shown."
)
backend = right.radio("Model server", ["mlflow", "ray"], horizontal=True, disabled=not live)

if package and not is_valid_package_name(package):
    st.warning("That is not a valid PyPI package name.")
elif package:
    try:
        risk = api_client().risk(package, live=live, backend=backend)
        detail = api_client().package(package)
    except ApiError as error:
        st.error(str(error))
    else:
        colour = GRADE_COLOURS.get(risk["grade"], "#757575")
        a, b, c = st.columns(3)
        a.metric("Risk of abandonment (18 months)", f"{risk['risk']:.0%}")
        b.markdown(
            f"### Grade <span style='color:{colour}'>{risk['grade']}</span>", unsafe_allow_html=True
        )
        c.caption(
            f"Source: {risk['source']} ({risk['backend']}), model version "
            f"{risk['model_version']}, data as of {risk['snapshot']}"
        )
        st.subheader("Why")
        for factor in risk["top_factors"]:
            st.write("• " + describe_factor(factor))

        st.subheader("Context")
        context = st.columns(4)
        context[0].metric(
            "Months since last release", number(detail["months_since_last_release"], ".1f")
        )
        context[1].metric("Commits in the past year", number(detail["commits_in_lookback"], ".0f"))
        context[2].metric("Bus factor", number(detail["bus_factor"], ".0f"))
        context[3].metric("Packages depending on it", number(detail["dependents"], ".0f"))
        if detail["repo_url"]:
            st.write(f"Repository: {detail['repo_url']}")
        deps, dependents = st.columns(2)
        deps.write(
            "**Depends on:** " + (", ".join(detail["dependencies"]) or "nothing in the scored set")
        )
        dependents.write(
            "**Depended on by:** "
            + (", ".join(detail["dependents_in_universe"][:30]) or "nothing in the scored set")
        )
