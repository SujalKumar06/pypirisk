"""Plain-language labels and small formatting helpers for the pages."""

FEATURE_LABELS = {
    "releases_in_lookback": "releases in the past year",
    "months_since_last_release": "months since the last release",
    "commits_in_lookback": "commits in the past year",
    "months_since_last_commit": "months since the last commit",
    "bus_factor": "bus factor (authors making half the commits)",
    "median_days_to_close": "median days to close an issue",
    "open_issue_share": "share of the year's issues still open",
    "dependents": "packages that depend on it",
    "pagerank": "importance in the dependency graph (PageRank)",
    "kcore": "depth in the dependency core (k-core)",
    "upstream_stale_share": "share of its dependencies with no release in a year",
}

GRADE_COLOURS = {"A": "#2e7d32", "B": "#7cb342", "C": "#f9a825", "D": "#ef6c00", "F": "#c62828"}


def describe_factor(factor: dict) -> str:
    """Turn one SHAP factor into a sentence.

    Args:
        factor: ``feature``, ``value`` and ``contribution``.

    Returns:
        E.g. ``"months since the last release = 26.1 raises the risk"``.
    """
    label = FEATURE_LABELS.get(factor["feature"], factor["feature"])
    value = "unknown" if factor["value"] is None else f"{factor['value']:.3g}"
    direction = "raises" if factor["contribution"] > 0 else "lowers"
    return f"{label} = {value} {direction} the risk"


def dot_graph(edges: list[tuple[str, str]], grades: dict[str, str], highlight: set[str]) -> str:
    """Build a Graphviz DOT graph coloured by grade.

    Args:
        edges: ``(package, dependency)`` pairs.
        grades: Letter grade per package (missing = unscored).
        highlight: Packages drawn with a bold border.

    Returns:
        DOT source for ``st.graphviz_chart``.
    """
    nodes = sorted({name for edge in edges for name in edge} | highlight)
    lines = ["digraph G {", "rankdir=LR;", 'node [shape=box, style="rounded,filled", fontsize=10];']
    for name in nodes:
        colour = GRADE_COLOURS.get(grades.get(name, ""), "#bdbdbd")
        width = 3 if name in highlight else 1
        lines.append(f'"{name}" [fillcolor="{colour}", fontcolor="white", penwidth={width}];')
    lines.extend(f'"{source}" -> "{target}";' for source, target in edges)
    lines.append("}")
    return "\n".join(lines)


def number(value: float | None, spec: str) -> str:
    """Format a number for display, or a dash when it is unknown.

    Args:
        value: The value, or ``None``.
        spec: Format spec, e.g. ``".1f"``.

    Returns:
        The formatted value.
    """
    return "–" if value is None else format(value, spec)
