import pandas as pd

from pypirisk.ingest.cache import ResponseCache, cache_key
from pypirisk.ingest.tables import archive_date, build_tables, write_event_tables


def package(name, repo=("github.com", "org", "repo"), found=True):
    return {
        "name": name,
        "found": found,
        "latest_version": "1.0",
        "releases": [{"version": "1.0", "uploaded_at": "2020-01-01T00:00:00Z"}],
        "requires": {"1.0": ["dep", name], "0.9": None},
        "repository": dict(zip(("host", "owner", "name"), repo, strict=True)) if repo else None,
        "errors": [],
        "requests": 2,
        "cache_hits": 1,
    }


def repository(url, archived=False, exact=None):
    return {
        "url": url,
        "has_commits": True,
        "metadata": {"archived": archived, "pushed_at": "2022-03-01T00:00:00Z"},
        "issues": {"complete_since": "2017-01-01"},
        "archived_at_exact": exact,
        "errors": [],
        "requests": 3,
        "cache_hits": 0,
    }


def test_archive_date_prefers_the_exact_github_date():
    assert archive_date(repository("u", archived=True, exact="2023-01-01")) == ("2023-01-01", True)
    assert archive_date(repository("u", archived=True)) == ("2022-03-01T00:00:00Z", False)
    assert archive_date(repository("u")) == (None, False)


def test_shared_repositories_are_flagged_as_monorepos():
    url = "https://github.com/org/repo"
    tables = build_tables(
        [package("a"), package("b"), package("c", repo=None)], [repository(url)], seeds=["a"]
    )
    flags = tables["packages"].set_index("name")["monorepo"].to_dict()
    assert flags == {"a": True, "b": True, "c": False}
    assert tables["packages"].set_index("name").loc["a", "is_seed"]


def test_requires_table_drops_self_dependencies_and_missing_versions():
    tables = build_tables([package("a")], [], seeds=["a"])
    assert tables["requires"].to_dict("records") == [
        {"name": "a", "version": "1.0", "dependency": "dep"}
    ]


def test_packages_not_on_pypi_are_left_out():
    tables = build_tables([package("a"), package("ghost", found=False)], [], seeds=["a", "ghost"])
    assert list(tables["packages"]["name"]) == ["a"]


def test_event_tables_stream_from_the_cache(tmp_path):
    url = "https://github.com/org/repo"
    cache = ResponseCache(tmp_path / "cache", offline=False)
    cache.put("git", cache_key(url), {"commits": [["2024-01-01T00:00:00+00:00", "abc", False]] * 2})
    cache.put(
        "ecosystems_issues",
        cache_key(url),
        {"issues": [["2024-01-01T00:00:00Z", None, False, False]], "truncated": False},
    )
    rows = write_event_tables([repository(url)], cache, tmp_path)
    assert rows == {"commits": 2, "issues": 1}
    commits = pd.read_parquet(tmp_path / "commits.parquet")
    assert str(commits["authored_at"].dt.tz) == "UTC"
