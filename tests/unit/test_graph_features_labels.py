import numpy as np
import pandas as pd
import pytest

from conftest import commits_frame, issues_frame, packages_frame, releases_frame, ts
from pypirisk.features import groups
from pypirisk.features.activity import is_bot_author
from pypirisk.features.build import snapshot_features
from pypirisk.graph.build import current_versions, network_metrics, snapshot_edges
from pypirisk.labels.build import snapshot_labels
from pypirisk.labels.windows import check_matured, check_split_windows, every_snapshot

SNAPSHOT = ts("2021-01-01")
LOOKBACK = ts("2020-01-01")
REPO = "https://github.com/o/r"


def test_current_versions_pick_the_newest_release_before_the_snapshot():
    releases = releases_frame(
        [("a", "1", "2019-01-01"), ("a", "2", "2020-06-01"), ("a", "3", "2021-06-01")]
    )
    assert current_versions(releases, SNAPSHOT).to_dict("records") == [
        {"name": "a", "version": "2"}
    ]


def test_snapshot_edges_use_the_dependencies_of_the_release_current_then():
    releases = releases_frame(
        [
            ("a", "1", "2019-01-01"),
            ("a", "2", "2022-01-01"),
            ("b", "1", "2018-01-01"),
            ("c", "1", "2018-01-01"),
        ]
    )
    requires = pd.DataFrame({"name": ["a", "a"], "version": ["1", "2"], "dependency": ["b", "c"]})
    assert snapshot_edges(releases, requires, SNAPSHOT).to_dict("records") == [
        {"source": "a", "target": "b"}
    ]


def test_network_metrics_put_leaf_pagerank_at_one():
    edges = pd.DataFrame({"source": ["a", "c"], "target": ["b", "b"]})
    metrics = network_metrics(["a", "b", "c"], edges).set_index("name")
    assert metrics.loc["a", "pagerank"] == pytest.approx(1.0)
    assert metrics.loc["b", "pagerank"] > 1.0
    assert metrics.loc["b", "dependents"] == 2


def test_release_features_only_count_releases_before_the_snapshot():
    releases = releases_frame(
        [("a", "1", "2018-01-01"), ("a", "2", "2020-07-01"), ("a", "3", "2021-02-01")]
    )
    features = groups.release_features(releases, SNAPSHOT, LOOKBACK).loc["a"]
    assert features["releases_in_lookback"] == 1
    assert features["months_since_last_release"] == pytest.approx(6, abs=0.1)


def test_bus_factor_is_the_fewest_authors_making_half_the_commits():
    commits = commits_frame(
        [(REPO, "2020-03-01", "alice", False)] * 4
        + [(REPO, "2020-04-01", "bob", False)] * 3
        + [(REPO, "2020-05-01", "carol", False)] * 3
    )
    features = groups.activity_and_maintainer_features(commits, SNAPSHOT, LOOKBACK).loc[REPO]
    assert features["bus_factor"] == 2
    assert features["commits_in_lookback"] == 10


def test_issue_still_open_at_the_snapshot_counts_as_open_even_if_closed_later():
    issues = issues_frame(
        [
            (REPO, "2020-06-01", "2020-06-11", False, False),
            (REPO, "2020-07-01", "2021-03-01", False, False),
        ]
    )
    features = groups.issue_features(issues, SNAPSHOT, LOOKBACK).loc[REPO]
    assert features["open_issue_share"] == 0.5
    assert features["median_days_to_close"] == 10


def test_upstream_features_flag_dependencies_without_recent_releases():
    releases_now = pd.DataFrame(
        {"releases_in_lookback": [0, 3], "months_since_last_release": [30.0, 2.0]},
        index=["old", "fresh"],
    )
    edges = pd.DataFrame({"source": ["app", "app"], "target": ["old", "fresh"]})
    upstream = groups.upstream_features(edges, releases_now).loc["app"]
    assert upstream["upstream_stale_share"] == 0.5


def tables_for(has_commit_data=True, complete_since="2017-01-01"):
    return {
        "packages": packages_frame(
            {
                "name": "a",
                "repo_url": REPO,
                "has_commit_data": has_commit_data,
                "issues_complete_since": complete_since,
            }
        ),
        "releases": releases_frame([("a", "1", "2020-06-01")]),
        "commits": commits_frame([(REPO, "2020-08-01", "alice", False)] if has_commit_data else []),
        "issues": issues_frame([(REPO, "2020-08-01", None, False, False)]),
    }


def feature_row(tables):
    nodes = pd.DataFrame(
        index=pd.Index(["a"], name="name"),
        data={"dependents": [0], "dependencies": [0], "pagerank": [1.0], "kcore": [0]},
    )
    edges = pd.DataFrame(columns=["source", "target"])
    return snapshot_features(tables, nodes, edges, "2021-01-01", 12).iloc[0]


def test_missing_commit_data_gives_nan_activity_not_zero():
    assert np.isnan(feature_row(tables_for(has_commit_data=False))["commits_in_lookback"])
    assert feature_row(tables_for())["commits_in_lookback"] == 1


def test_issue_history_that_does_not_reach_the_lookback_gives_nan_issue_features():
    assert np.isnan(feature_row(tables_for(complete_since="2020-06-01"))["open_issue_share"])
    assert feature_row(tables_for())["open_issue_share"] == 1.0


def test_bot_authors_are_detected_case_insensitively():
    assert is_bot_author("Dependabot[bot]", ["dependabot"])
    assert not is_bot_author("alice", ["[bot]"])


def label_tables(releases, commits=(), archived_at=None):
    return (
        packages_frame({"name": "a", "repo_url": REPO, "archived_at": archived_at}),
        releases_frame([("a", str(i), date) for i, date in enumerate(releases)]),
        commits_frame([(REPO, date, "x", False) for date in commits]),
    )


def label(spec, releases, commits=(), archived_at=None):
    table = snapshot_labels(*label_tables(releases, commits, archived_at), "2021-01-01", spec)
    return None if table.empty else int(table.iloc[0]["label"])


def test_quiet_package_is_abandoned(label_spec):
    assert label(label_spec, ["2020-06-01"], commits=["2021-03-01", "2021-04-01"]) == 1


def test_a_release_or_enough_commits_in_the_window_means_active(label_spec):
    assert label(label_spec, ["2020-06-01", "2022-06-01"]) == 0
    assert (
        label(label_spec, ["2020-06-01"], commits=["2021-02-01", "2021-03-01", "2021-04-01"]) == 0
    )


def test_activity_after_the_outcome_window_does_not_count(label_spec):
    assert label(label_spec, ["2020-06-01", "2022-07-02"]) == 1


def test_archiving_in_the_window_means_abandoned_and_before_it_means_ineligible(label_spec):
    assert label(label_spec, ["2020-06-01", "2021-06-01"], archived_at="2022-01-01") == 1
    assert label(label_spec, ["2020-06-01"], archived_at="2020-12-01") is None


def test_inactive_packages_are_not_eligible(label_spec):
    assert label(label_spec, ["2019-06-01"]) is None


def test_split_windows_reject_overlapping_outcomes():
    check_split_windows(
        {"train": ["2020-01-01"], "validation": ["2023-01-01"], "test": ["2025-01-01"]}, 18
    )
    with pytest.raises(ValueError, match="after split 'validation'"):
        check_split_windows(
            {"train": ["2021-01-01"], "validation": ["2022-01-01"], "test": ["2025-01-01"]}, 18
        )


def test_unmatured_snapshots_are_rejected():
    with pytest.raises(ValueError, match="has not matured"):
        check_matured({"train": [], "validation": [], "test": ["2025-06-01"]}, 18, ts("2026-11-01"))


def test_every_snapshot_appends_the_scoring_snapshot():
    splits = {
        "train": ["2020-01-01"],
        "validation": ["2023-01-01"],
        "test": ["2025-01-01"],
        "scoring": "2026-11-01",
    }
    assert every_snapshot(splits)[-1] == "2026-11-01"
