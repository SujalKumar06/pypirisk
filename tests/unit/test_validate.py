import pandas as pd

from conftest import commits_frame, issues_frame, packages_frame, releases_frame, ts
from pypirisk.validate import checks

NOW = ts("2026-11-02")


def test_releases_of_unknown_packages_or_impossible_dates_are_rejected():
    releases = releases_frame(
        [
            ("a", "1", "2020-01-01"),
            ("ghost", "1", "2020-01-01"),
            ("a", "2", "1990-01-01"),
            ("a", "3", None),
        ]
    )
    assert checks.check_releases(releases, {"a"}, NOW).tolist() == [
        "",
        "unknown package",
        "upload time out of range",
        "missing upload time",
    ]


def test_commits_from_misconfigured_clocks_are_rejected():
    commits = commits_frame(
        [
            ("r", "1970-01-01", "x", False),
            ("r", "2035-01-01", "x", False),
            ("r", "2024-01-01", "x", False),
        ]
    )
    assert checks.check_commits(commits, NOW).tolist() == ["commit date out of range"] * 2 + [""]


def test_issues_closed_before_creation_are_rejected():
    issues = issues_frame(
        [("r", "2024-02-01", "2024-01-01", False, False), ("r", "2024-01-01", None, False, False)]
    )
    assert checks.check_issues(issues, NOW).tolist() == ["closed before created", ""]


def test_requires_without_a_dependency_are_rejected():
    requires = pd.DataFrame({"name": ["a", "a"], "version": ["1", "1"], "dependency": ["b", ""]})
    assert checks.check_requires(requires, {"a"}).tolist() == ["", "missing dependency"]


def test_duplicate_package_names_are_rejected():
    packages = packages_frame({"name": "a"}, {"name": "a"}, {"name": "b"})
    assert checks.check_packages(packages).tolist() == ["duplicate name", "duplicate name", ""]
