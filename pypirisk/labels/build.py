"""Decide eligibility and the abandonment label for each package and snapshot."""

from dataclasses import dataclass

import pandas as pd

from pypirisk.features.groups import human_events
from pypirisk.labels.windows import SPLIT_ORDER
from pypirisk.timeutils import add_months


@dataclass(frozen=True)
class LabelSpec:
    """Thresholds that define eligibility and abandonment.

    Attributes:
        lookback_months: Window before the snapshot used to judge "actively maintained".
        outcome_months: Window after the snapshot in which abandonment is observed.
        max_commits_in_outcome: Most human commits an abandoned package may have.
        min_releases_for_eligibility: Releases needed in the lookback window.
    """

    lookback_months: int
    outcome_months: int
    max_commits_in_outcome: int
    min_releases_for_eligibility: int


def _count_between(times: pd.Series, keys: pd.Series, start, end) -> pd.Series:
    inside = (times >= start) & (times < end)
    return inside.groupby(keys).sum()


def snapshot_labels(
    packages: pd.DataFrame,
    releases: pd.DataFrame,
    commits: pd.DataFrame,
    snapshot: str,
    spec: LabelSpec,
) -> pd.DataFrame:
    """Label every package that was actively maintained at ``snapshot``.

    Eligible: at least ``min_releases_for_eligibility`` releases in the lookback window
    and not archived before the snapshot. Abandoned: archived before the outcome window
    ends, or no release and at most ``max_commits_in_outcome`` human commits in it.
    Packages without commit data are labelled on releases alone (``label_basis``).

    Args:
        packages: ``name``, ``repo_url``, ``archived_at``, ``has_commit_data``.
        releases: ``name``, ``uploaded_at``.
        commits: Human commits: ``repo_url``, ``authored_at``.
        snapshot: ISO snapshot date.
        spec: Label thresholds.

    Returns:
        ``name``, ``snapshot``, ``label`` (1 = abandoned) and ``label_basis``.
    """
    moment = pd.Timestamp(snapshot, tz="UTC")
    lookback_start = add_months(moment, -spec.lookback_months)
    outcome_end = add_months(moment, spec.outcome_months)

    recent = _count_between(releases["uploaded_at"], releases["name"], lookback_start, moment)
    later = _count_between(releases["uploaded_at"], releases["name"], moment, outcome_end)
    commit_counts = _count_between(commits["authored_at"], commits["repo_url"], moment, outcome_end)

    table = packages.set_index("name")
    table = table.assign(
        recent_releases=recent.reindex(table.index, fill_value=0),
        outcome_releases=later.reindex(table.index, fill_value=0),
        outcome_commits=table["repo_url"].map(commit_counts).fillna(0),
    )
    archived = table["archived_at"]
    eligible = (table["recent_releases"] >= spec.min_releases_for_eligibility) & ~(
        archived.notna() & (archived < moment)
    )
    table = table[eligible]
    archived = table["archived_at"]
    quiet = (table["outcome_releases"] == 0) & (
        table["outcome_commits"] <= spec.max_commits_in_outcome
    )
    abandoned = (archived.notna() & (archived < outcome_end)) | quiet
    basis = table["has_commit_data"].map({True: "releases_and_commits", False: "releases_only"})
    return pd.DataFrame(
        {
            "name": table.index,
            "snapshot": snapshot,
            "label": abandoned.astype(int).to_numpy(),
            "label_basis": basis.to_numpy(),
        }
    )


def build_label_table(
    tables: dict[str, pd.DataFrame], splits: dict[str, list[str]], spec: LabelSpec
) -> pd.DataFrame:
    """Label every eligible package at every labelled snapshot.

    Args:
        tables: Validated ``packages``, ``releases`` and ``commits`` tables.
        splits: The ``snapshots`` section of ``params.yaml``.
        spec: Label thresholds.

    Returns:
        ``name``, ``snapshot``, ``split``, ``label``, ``label_basis``, sorted by
        snapshot then name.
    """
    commits = human_events(tables["commits"])
    frames = [
        snapshot_labels(tables["packages"], tables["releases"], commits, snapshot, spec).assign(
            split=split
        )
        for split in SPLIT_ORDER
        for snapshot in splits[split]
    ]
    columns = ["name", "snapshot", "split", "label", "label_basis"]
    table = pd.concat(frames, ignore_index=True)[columns]
    return table.sort_values(["snapshot", "name"], ignore_index=True)
