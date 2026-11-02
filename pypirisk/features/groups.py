"""The five feature groups, each computed for one snapshot.

Every function receives the snapshot time and the start of the lookback window and
uses only events strictly before the snapshot. The leakage tests check this by adding
future events and asserting the features do not change.
"""

import numpy as np
import pandas as pd

DAYS_PER_MONTH = 365.25 / 12


def _months(delta: pd.Series) -> pd.Series:
    return delta.dt.total_seconds() / 86400 / DAYS_PER_MONTH


def release_features(
    releases: pd.DataFrame, snapshot: pd.Timestamp, lookback_start: pd.Timestamp
) -> pd.DataFrame:
    """Release history: how recently and how often the package released.

    Args:
        releases: ``name``, ``uploaded_at``.
        snapshot: The snapshot time.
        lookback_start: Start of the lookback window.

    Returns:
        ``months_since_last_release`` and ``releases_in_lookback`` for every package with
        a release before the snapshot, indexed by ``name``.
    """
    past = releases[releases["uploaded_at"] < snapshot]
    last = past.groupby("name")["uploaded_at"].max()
    recent = past[past["uploaded_at"] >= lookback_start].groupby("name").size()
    return pd.DataFrame(
        {
            "months_since_last_release": _months(snapshot - last),
            "releases_in_lookback": recent.reindex(last.index, fill_value=0),
        }
    )


def _bus_factor(counts: pd.Series) -> int:
    shares = counts.sort_values(ascending=False).cumsum() / counts.sum()
    return int((shares < 0.5).sum() + 1)


def activity_and_maintainer_features(
    commits: pd.DataFrame, snapshot: pd.Timestamp, lookback_start: pd.Timestamp
) -> pd.DataFrame:
    """Development activity and maintainer concentration from human commits.

    The bus factor is the smallest number of authors who made at least half of the
    lookback window's commits.

    Args:
        commits: Human commits: ``repo_url``, ``authored_at``, ``author_id``.
        snapshot: The snapshot time.
        lookback_start: Start of the lookback window.

    Returns:
        ``commits_in_lookback``, ``months_since_last_commit`` and ``bus_factor``, indexed
        by ``repo_url``, for repositories with a commit before the snapshot.
    """
    past = commits[commits["authored_at"] < snapshot]
    window = past[past["authored_at"] >= lookback_start]
    per_author = window.groupby(["repo_url", "author_id"]).size()
    features = pd.DataFrame(
        {
            "months_since_last_commit": _months(
                snapshot - past.groupby("repo_url")["authored_at"].max()
            )
        }
    )
    features["commits_in_lookback"] = window.groupby("repo_url").size()
    features["bus_factor"] = per_author.groupby(level="repo_url").apply(_bus_factor)
    counts = ["commits_in_lookback", "bus_factor"]
    features[counts] = features[counts].fillna(0)
    return features


def issue_features(
    issues: pd.DataFrame, snapshot: pd.Timestamp, lookback_start: pd.Timestamp
) -> pd.DataFrame:
    """Issue responsiveness from human-filed issues (pull requests excluded).

    An issue counts as open at the snapshot if it was not closed before it, even if it
    was closed later.

    Args:
        issues: ``repo_url``, ``created_at``, ``closed_at`` (human issues only).
        snapshot: The snapshot time.
        lookback_start: Start of the lookback window.

    Returns:
        ``median_days_to_close`` (issues closed in the window) and ``open_issue_share``
        (issues opened in the window and still open), indexed by ``repo_url``.
    """
    opened = issues[(issues["created_at"] >= lookback_start) & (issues["created_at"] < snapshot)]
    closed = issues[(issues["closed_at"] >= lookback_start) & (issues["closed_at"] < snapshot)]
    still_open = opened["closed_at"].isna() | (opened["closed_at"] >= snapshot)
    days_to_close = (closed["closed_at"] - closed["created_at"]).dt.total_seconds() / 86400
    return pd.DataFrame(
        {
            "median_days_to_close": days_to_close.groupby(closed["repo_url"]).median(),
            "open_issue_share": still_open.groupby(opened["repo_url"]).mean(),
        }
    )


def upstream_features(edges: pd.DataFrame, releases_now: pd.DataFrame) -> pd.DataFrame:
    """Upstream risk: the share of a package's direct dependencies that stopped releasing.

    This uses the heuristic signal of the same snapshot, never model scores, so the
    feature does not depend on the model it feeds (no circularity).

    Args:
        edges: One snapshot's ``source -> target`` edges.
        releases_now: That snapshot's :func:`release_features`.

    Returns:
        ``upstream_stale_share`` (dependencies with no release in the lookback), indexed
        by package.
    """
    upstream = edges.merge(releases_now, left_on="target", right_index=True)
    stale = upstream["releases_in_lookback"] == 0
    return stale.groupby(upstream["source"]).mean().to_frame("upstream_stale_share")


def human_events(table: pd.DataFrame) -> pd.DataFrame:
    """Drop rows created by bots.

    Args:
        table: A commit or issue table with an ``is_bot`` column.

    Returns:
        The rows made by people.
    """
    return table[~table["is_bot"].astype(bool)]


def nan_where(frame: pd.DataFrame, columns: list[str], mask: pd.Series) -> pd.DataFrame:
    """Blank out ``columns`` on rows where ``mask`` is true.

    Args:
        frame: The feature table.
        columns: Columns to blank.
        mask: Rows whose values are unknown.

    Returns:
        The table with unknown values set to NaN.
    """
    frame.loc[mask.to_numpy(), columns] = np.nan
    return frame
