"""Turn raw crawl results into the five tables the rest of the pipeline reads.

| Table | One row per |
| --- | --- |
| ``packages`` | package, with repository link, archive status and coverage flags |
| ``releases`` | release upload |
| ``requires`` | (package, version, dependency) for the versions current at snapshots |
| ``commits`` | commit in a repository since ``history_start`` |
| ``issues`` | issue or pull request in a repository |
"""

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pypirisk.ingest.cache import ResponseCache, cache_key

UTC_TIME = pa.timestamp("ns", tz="UTC")

# Commits and issues: cache source, key in the cached document, and the table's schema.
EVENT_TABLES = {
    "commits": (
        "git",
        "commits",
        pa.schema([("authored_at", UTC_TIME), ("author_id", pa.string()), ("is_bot", pa.bool_())]),
    ),
    "issues": (
        "ecosystems_issues",
        "issues",
        pa.schema(
            [
                ("created_at", UTC_TIME),
                ("closed_at", UTC_TIME),
                ("is_pr", pa.bool_()),
                ("is_bot", pa.bool_()),
            ]
        ),
    ),
}


def _utc(values) -> pd.Series:
    return pd.to_datetime(pd.Series(values, dtype="object"), utc=True, format="ISO8601")


def archive_date(repo: dict[str, Any]) -> tuple[str | None, bool]:
    """Return when a repository was archived and whether that date is exact.

    Without GitHub GraphQL, the last push is used as the archive date; repositories are
    normally archived soon after their final push (see ``docs/HLD.md``).

    Args:
        repo: A repository crawl result.

    Returns:
        The archive timestamp (or ``None``) and ``True`` if it came from GitHub.
    """
    if repo.get("archived_at_exact"):
        return repo["archived_at_exact"], True
    metadata = repo.get("metadata") or {}
    return (metadata.get("pushed_at"), False) if metadata.get("archived") else (None, False)


def _repo_url(package: dict[str, Any]) -> str | None:
    repo = package.get("repository")
    return f"https://{repo['host']}/{repo['owner']}/{repo['name']}" if repo else None


def build_tables(
    packages: list[dict[str, Any]], repositories: list[dict[str, Any]], seeds: list[str]
) -> dict[str, pd.DataFrame]:
    """Build the package, release and dependency tables from crawl results.

    A repository shared by several packages in the universe is flagged ``monorepo``: its
    commits and issues describe all of them, not one package. Versions whose metadata
    PyPI could not provide get no dependency rows, rather than borrowing today's.

    Args:
        packages: Package crawl results.
        repositories: Repository crawl results.
        seeds: Names that seeded the universe.

    Returns:
        ``packages``, ``releases`` and ``requires`` tables. Commits and issues are
        streamed separately by :func:`write_event_tables`.
    """
    repos = {repo["url"]: repo for repo in repositories}
    found = [package for package in packages if package["found"]]
    urls = [_repo_url(package) for package in found]
    shared = Counter(url for url in urls if url)
    rows = []
    for package, url in zip(found, urls, strict=True):
        repo = repos.get(url, {})
        archived_at, exact = archive_date(repo)
        rows.append(
            {
                "name": package["name"],
                "latest_version": package["latest_version"],
                "is_seed": package["name"] in seeds,
                "repo_url": url,
                "repo_host": package["repository"]["host"] if url else None,
                "archived": bool((repo.get("metadata") or {}).get("archived")),
                "archived_at": archived_at,
                "archived_at_exact": exact,
                "has_commit_data": bool(repo.get("has_commits")),
                "has_issue_data": repo.get("issues") is not None,
                "issues_complete_since": (repo.get("issues") or {}).get("complete_since"),
                "monorepo": bool(url) and shared[url] > 1,
            }
        )
    table = pd.DataFrame(rows)
    for column in ("archived_at", "issues_complete_since"):
        table[column] = _utc(table[column])

    releases = pd.DataFrame(
        [(p["name"], r["version"], r["uploaded_at"]) for p in found for r in p["releases"]],
        columns=["name", "version", "uploaded_at"],
    )
    releases["uploaded_at"] = _utc(releases["uploaded_at"])
    requires = pd.DataFrame(
        [
            (p["name"], version, dependency)
            for p in found
            for version, dependencies in p["requires"].items()
            for dependency in dependencies or []
            if dependency != p["name"]
        ],
        columns=["name", "version", "dependency"],
    )
    return {"packages": table, "releases": releases, "requires": requires}


def write_event_tables(
    repositories: list[dict[str, Any]], cache: ResponseCache, output_dir: Path
) -> dict[str, int]:
    """Stream commits and issues from the response cache into Parquet, one repo at a time.

    Writing repository by repository keeps memory flat even for millions of rows.

    Args:
        repositories: Repository crawl results.
        cache: The response cache the crawl filled.
        output_dir: Directory for ``commits.parquet`` and ``issues.parquet``.

    Returns:
        Rows written per table.
    """
    counts = {}
    for table, (source, key, event_schema) in EVENT_TABLES.items():
        schema = pa.schema([("repo_url", pa.string()), *event_schema])
        counts[table] = 0
        with pq.ParquetWriter(output_dir / f"{table}.parquet", schema) as writer:
            for repo in repositories:
                document = cache.get(source, cache_key(repo["url"]))
                if not document or document.get("missing"):
                    continue
                frame = pd.DataFrame(document[key], columns=event_schema.names)
                for column in event_schema.names:
                    if event_schema.field(column).type == UTC_TIME:
                        frame[column] = _utc(frame[column])
                frame.insert(0, "repo_url", repo["url"])
                writer.write_table(pa.Table.from_pandas(frame, schema=schema, preserve_index=False))
                counts[table] += len(frame)
    return counts
