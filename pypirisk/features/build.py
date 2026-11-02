"""Assemble the feature table: one row per package per snapshot."""

import pandas as pd

from pypirisk.features import groups
from pypirisk.timeutils import add_months

ACTIVITY_COUNTS = ["commits_in_lookback", "bus_factor"]
ACTIVITY_COLUMNS = [*ACTIVITY_COUNTS, "months_since_last_commit"]
ISSUE_COLUMNS = ["median_days_to_close", "open_issue_share"]
NETWORK_COLUMNS = ["dependents", "pagerank", "kcore", "upstream_stale_share"]
PACKAGE_COLUMNS = [
    "repo_url",
    "has_commit_data",
    "has_issue_data",
    "issues_complete_since",
    "monorepo",
]


def snapshot_features(
    tables: dict[str, pd.DataFrame],
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    snapshot: str,
    lookback_months: int,
) -> pd.DataFrame:
    """Build every feature group for one snapshot.

    Activity features are NaN when the repository could not be cloned, and issue
    features are NaN when the issue history does not reach back to the lookback start,
    so missing data is never mistaken for zero activity.

    Args:
        tables: Validated tables (``packages``, ``releases``, ``commits``, ``issues``).
        nodes: That snapshot's network metrics, indexed by name.
        edges: That snapshot's dependency edges.
        snapshot: ISO snapshot date.
        lookback_months: Length of the lookback window.

    Returns:
        One row per package that had a release before the snapshot.
    """
    moment = pd.Timestamp(snapshot, tz="UTC")
    lookback_start = add_months(moment, -lookback_months)
    releases = groups.release_features(tables["releases"], moment, lookback_start)
    commits = groups.human_events(tables["commits"])
    issues = groups.human_events(tables["issues"])
    issues = issues[~issues["is_pr"].astype(bool)]

    table = releases.join(tables["packages"].set_index("name")[PACKAGE_COLUMNS], how="left")
    table = table.join(
        groups.activity_and_maintainer_features(commits, moment, lookback_start), on="repo_url"
    )
    table = table.join(groups.issue_features(issues, moment, lookback_start), on="repo_url")
    table = table.join(nodes[["dependents", "pagerank", "kcore"]])
    table = table.join(groups.upstream_features(edges, releases))

    has_commits = table["has_commit_data"].fillna(False).astype(bool)
    table.loc[has_commits, ACTIVITY_COUNTS] = table.loc[has_commits, ACTIVITY_COUNTS].fillna(0)
    table = groups.nan_where(table, ACTIVITY_COLUMNS, ~has_commits)

    complete_since = table["issues_complete_since"]
    issues_known = table["has_issue_data"].fillna(False).astype(bool) & (
        complete_since.isna() | (complete_since <= lookback_start)
    )
    table = groups.nan_where(table, ISSUE_COLUMNS, ~issues_known)

    table[NETWORK_COLUMNS] = table[NETWORK_COLUMNS].fillna(0)
    table["monorepo"] = table["monorepo"].fillna(False).astype(int)
    table["has_issue_data"] = issues_known.astype(int)
    table["has_commit_data"] = has_commits.astype(int)
    table = table.drop(columns=["issues_complete_since"]).rename_axis("name").reset_index()
    return table.assign(snapshot=snapshot)


def build_feature_table(
    tables: dict[str, pd.DataFrame],
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    snapshots: list[str],
    lookback_months: int,
) -> pd.DataFrame:
    """Build the feature table for all snapshots.

    Args:
        tables: Validated tables.
        nodes: Node metrics for all snapshots (``snapshot`` column).
        edges: Edges for all snapshots (``snapshot`` column).
        snapshots: ISO snapshot dates.
        lookback_months: Length of the lookback window.

    Returns:
        Features for every package and snapshot, sorted by snapshot then name.
    """
    frames = [
        snapshot_features(
            tables,
            nodes[nodes["snapshot"] == snapshot].set_index("name"),
            edges[edges["snapshot"] == snapshot],
            snapshot,
            lookback_months,
        )
        for snapshot in snapshots
    ]
    return pd.concat(frames, ignore_index=True).sort_values(["snapshot", "name"], ignore_index=True)
