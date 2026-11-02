"""No feature, graph edge or drift reference may use data dated after its snapshot."""

import pandas as pd

from conftest import commits_frame, issues_frame, packages_frame, releases_frame
from pypirisk.features.build import build_feature_table
from pypirisk.graph.build import build_graphs
from pypirisk.labels.windows import check_split_windows

SNAPSHOTS = ["2019-01-01", "2021-01-01"]
REPO = "https://github.com/o/a"


def tables(with_future: bool) -> dict:
    releases = [("a", "1", "2018-03-01"), ("a", "2", "2020-05-01"), ("b", "1", "2017-01-01")]
    commits = [(REPO, "2018-06-01", "alice", False), (REPO, "2020-07-01", "bob", False)]
    issues = [
        (REPO, "2020-02-01", "2021-02-01", False, False),
        (REPO, "2018-02-01", None, False, False),
    ]
    requires = [("a", "1", "b"), ("a", "2", "b")]
    if with_future:
        releases += [
            ("a", "3", "2021-01-01T00:00:00"),
            ("a", "4", "2023-01-01"),
            ("c", "1", "2022-01-01"),
        ]
        commits += [
            (REPO, "2021-01-01T00:00:00", "carol", False),
            (REPO, "2024-01-01", "dave", False),
        ]
        issues += [(REPO, "2021-06-01", None, False, False)]
        requires += [("a", "3", "c"), ("a", "4", "c")]
    return {
        "packages": packages_frame({"name": "a", "repo_url": REPO}, {"name": "b"}, {"name": "c"}),
        "releases": releases_frame(releases),
        "commits": commits_frame(commits),
        "issues": issues_frame(issues),
        "requires": pd.DataFrame(requires, columns=["name", "version", "dependency"]),
    }


def features(with_future: bool) -> pd.DataFrame:
    data = tables(with_future)
    edges, nodes = build_graphs(data["releases"], data["requires"], SNAPSHOTS)
    return build_feature_table(data, nodes, edges, SNAPSHOTS, lookback_months=12)


def test_events_at_or_after_the_snapshot_never_change_any_feature():
    pd.testing.assert_frame_equal(features(with_future=False), features(with_future=True))


def test_snapshot_graphs_ignore_dependencies_added_later():
    past, future = tables(False), tables(True)
    edges_past, _ = build_graphs(past["releases"], past["requires"], SNAPSHOTS)
    edges_future, _ = build_graphs(future["releases"], future["requires"], SNAPSHOTS)
    pd.testing.assert_frame_equal(edges_past, edges_future)


def test_configured_splits_do_not_overlap_in_time(params):
    check_split_windows(params["snapshots"], params["labels"]["outcome_months"])


def test_coverage_flags_measured_today_are_not_model_inputs(params):
    leaky = {"has_commit_data", "has_issue_data", "dependents_in_universe"}
    assert not leaky & set(params["train"]["feature_columns"])


def test_drift_reference_comes_from_a_labelled_snapshot_not_the_scoring_one(params):
    assert params["snapshots"]["test"][-1] < params["snapshots"]["scoring"]
