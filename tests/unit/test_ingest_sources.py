from datetime import UTC, datetime

import pytest

from pypirisk.ingest.cache import ResponseCache
from pypirisk.ingest.sources import ecosystems, git_history, github_graphql, pypi
from pypirisk.ingest.sources.repos import Repository, candidate_urls, parse_repository
from pypirisk.ingest.universe import versions_current_at

RAW_PACKAGE = {
    "info": {
        "name": "Demo_Pkg",
        "version": "2.0",
        "requires_dist": ["idna>=2"],
        "project_urls": {"Source": "https://github.com/Example/Demo.git"},
        "home_page": None,
    },
    "releases": {
        "2.0": [
            {"upload_time_iso_8601": "2021-05-02T00:00:00Z"},
            {"upload_time_iso_8601": "2021-05-01T00:00:00Z"},
        ],
        "1.0": [{"upload_time_iso_8601": "2020-01-01T00:00:00Z"}],
        "0.9": [],
    },
}


def test_trim_keeps_earliest_upload_per_release_oldest_first():
    trimmed = pypi.trim_package(RAW_PACKAGE)
    assert [r["version"] for r in trimmed["releases"]] == ["1.0", "2.0"]
    assert trimmed["releases"][1]["uploaded_at"] == "2021-05-01T00:00:00Z"


def test_required_packages_skip_optional_and_invalid_entries():
    entries = ["Typing_Extensions>=4; python_version < '3.11'", "pytest; extra == 'test'", "!!bad"]
    assert pypi.required_packages(entries) == ["typing-extensions"]


def test_fetch_package_replays_the_cache_offline(tmp_path):
    cache = ResponseCache(tmp_path, offline=False)
    cache.put("pypi", "demo", pypi.trim_package(RAW_PACKAGE))
    offline = ResponseCache(tmp_path, offline=True)
    assert pypi.fetch_package(None, offline, "https://pypi.org/pypi", "demo")["version"] == "2.0"


def test_missing_package_is_cached_as_missing(tmp_path):
    class NotFound:
        def get_json(self, host, url, params=None):
            return None

    cache = ResponseCache(tmp_path, offline=False)
    assert pypi.fetch_package(NotFound(), cache, "base", "ghost") is None
    assert (
        pypi.fetch_package(NotFound(), ResponseCache(tmp_path, offline=True), "base", "ghost")
        is None
    )


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://github.com/Psf/Requests.git", "https://github.com/psf/requests"),
        ("https://github.com/psf/requests/tree/main/docs", "https://github.com/psf/requests"),
        ("git+https://gitlab.com/group/proj", "https://gitlab.com/group/proj"),
        ("https://github.com/sponsors/hynek", None),
        ("https://example.org/docs", None),
        (None, None),
    ],
)
def test_parse_repository_normalises_known_hosts(url, expected):
    repository = parse_repository(url)
    assert (repository.url if repository else None) == expected


def test_candidate_urls_put_source_links_first():
    urls = {"Funding": "https://github.com/sponsors/x", "Source Code": "https://github.com/a/b"}
    assert candidate_urls(urls, "https://home.page")[:2] == [
        "https://github.com/a/b",
        "https://home.page",
    ]


def test_versions_current_at_picks_the_newest_release_before_each_snapshot():
    releases = [
        {"version": "1.0", "uploaded_at": "2019-06-01T00:00:00Z"},
        {"version": "2.0", "uploaded_at": "2020-06-01T00:00:00Z"},
    ]
    snapshots = [
        datetime(2019, 1, 1, tzinfo=UTC),
        datetime(2020, 1, 1, tzinfo=UTC),
        datetime(2021, 1, 1, tzinfo=UTC),
    ]
    assert versions_current_at(releases, snapshots) == {"1.0", "2.0"}


def test_author_id_is_stable_case_insensitive_and_not_the_email():
    assert git_history.author_id("Alice@Example.org ") == git_history.author_id("alice@example.org")
    assert "alice" not in git_history.author_id("alice@example.org")


def test_parse_log_flags_bots_and_skips_malformed_lines():
    output = (
        "2024-01-01T10:00:00+01:00\talice@x.org\tAlice\n"
        "garbage line\n"
        "2024-01-02T10:00:00+00:00\t49699333+dependabot[bot]@users.noreply.github.com\tdependabot[bot]\n"
    )
    rows = git_history.parse_log(output, ["[bot]"])
    assert len(rows) == 2
    assert rows[0][0] == "2024-01-01T10:00:00+01:00"
    assert [row[2] for row in rows] == [False, True]


def test_compact_issue_keeps_dates_kind_and_bot_flag():
    issue = {
        "created_at": "2024-01-01",
        "closed_at": None,
        "pull_request": True,
        "user": "renovate[bot]",
    }
    assert ecosystems.compact_issue(issue, ["[bot]"]) == ["2024-01-01", None, True, True]


class PagedClient:
    def __init__(self, pages):
        self.pages = pages
        self.calls = 0

    def get_json(self, host, url, params=None):
        self.calls += 1
        return self.pages[params["page"] - 1] if params["page"] <= len(self.pages) else []


def issue(created):
    return {"created_at": created, "closed_at": None, "pull_request": False, "user": "a"}


def test_fetch_issues_stops_once_history_start_is_reached(tmp_path):
    client = PagedClient([[issue("2024-01-01"), issue("2016-05-01")], [issue("2015-01-01")] * 2])
    result = ecosystems.fetch_issues(
        client,
        ResponseCache(tmp_path, False),
        "u",
        Repository("github.com", "a", "b"),
        "2017-01-01",
        2,
        10,
        [],
    )
    assert client.calls == 1
    assert result["truncated"] is False
    assert len(result["issues"]) == 2


def test_fetch_issues_marks_truncation_at_the_page_cap(tmp_path):
    client = PagedClient(
        [[issue("2024-02-01"), issue("2024-01-01")], [issue("2023-06-01"), issue("2023-01-01")]]
    )
    result = ecosystems.fetch_issues(
        client,
        ResponseCache(tmp_path, False),
        "u",
        Repository("github.com", "a", "b"),
        "2017-01-01",
        2,
        2,
        [],
    )
    assert result["truncated"] is True
    assert result["complete_since"] == "2023-01-01"


def test_fetch_issues_skips_hosts_ecosystems_does_not_index(tmp_path):
    repo = Repository("bitbucket.org", "a", "b")
    assert (
        ecosystems.fetch_issues(None, ResponseCache(tmp_path, False), "u", repo, "2017", 1, 1, [])
        is None
    )


def test_graphql_archived_at_only_for_archived_repositories():
    archived = {"data": {"repository": {"isArchived": True, "archivedAt": "2023-01-01T00:00:00Z"}}}
    active = {"data": {"repository": {"isArchived": False, "archivedAt": None}}}
    assert github_graphql.parse_archived_at(archived) == "2023-01-01T00:00:00Z"
    assert github_graphql.parse_archived_at(active) is None
    assert github_graphql.parse_archived_at({"data": {"repository": None}}) is None


def test_fetch_issues_rejects_error_objects_instead_of_crashing_the_crawl(tmp_path):
    class ErrorClient:
        def get_json(self, host, url, params=None):
            return {"error": "repository not synced"}

    with pytest.raises(ValueError, match="unexpected issues response"):
        ecosystems.fetch_issues(
            ErrorClient(),
            ResponseCache(tmp_path, False),
            "u",
            Repository("github.com", "a", "b"),
            "2017",
            10,
            2,
            [],
        )
