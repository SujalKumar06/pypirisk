"""Crawl every source in parallel with Ray, under shared per-host rate limits.

The crawl has two waves of Ray tasks:

1. one task per package (PyPI metadata, per-snapshot dependencies, repository link),
   which also discovers dependencies to add to the universe;
2. one task per unique repository (commit-only git clone, archive status, issues).

A single :class:`~pypirisk.ingest.ratelimit.RateLimiter` actor enforces the request
rate of each API host across all tasks.
"""

import logging
import os
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import ray
import requests

from pypirisk.ingest.cache import CacheMissError, ResponseCache
from pypirisk.ingest.http import HttpClient
from pypirisk.ingest.ratelimit import RateLimiter
from pypirisk.ingest.sources import ecosystems, git_history, github_graphql, pypi
from pypirisk.ingest.sources.repos import Repository, candidate_urls, first_repository
from pypirisk.ingest.universe import top_package_names, versions_current_at
from pypirisk.timeutils import parse_timestamp

logger = logging.getLogger(__name__)

EXPECTED_ERRORS = (requests.RequestException, CacheMissError, git_history.GitError, ValueError)
NO_LIMIT = {"pypi": 1e9, "ecosystems": 1e9, "github_graphql": 1e9}


@dataclass(frozen=True)
class CrawlSettings:
    """Everything a crawl task needs; plain data so Ray can ship it to workers.

    Attributes:
        cache_dir: Response cache directory.
        offline: Never call the network; replay the cache only.
        sources: The ``sources`` section of ``params.yaml`` plus ``rate_limits``.
        snapshots: Every snapshot date (labelled and scoring), ISO format.
        crawl: The ``crawl`` section of ``params.yaml``.
        bot_markers: Substrings that identify bot authors.
        github_token: Token for GitHub GraphQL; empty disables it.
    """

    cache_dir: str
    offline: bool
    sources: dict[str, Any]
    snapshots: tuple[str, ...]
    crawl: dict[str, Any]
    bot_markers: tuple[str, ...]
    github_token: str = ""


def _client(settings: CrawlSettings, limiter, headers: dict | None = None) -> HttpClient:
    return HttpClient(
        settings.sources["request_timeout_seconds"],
        reserve_slot=lambda host: ray.get(limiter.reserve.remote(host)),
        headers=headers,
    )


def _attempt(errors: list[str], source: str, subject: str, call: Callable[[], Any]) -> Any:
    """Run one source call; record a failure instead of failing the whole crawl."""
    try:
        return call()
    except EXPECTED_ERRORS as error:
        logger.warning(
            "Crawl error", extra={"source": source, "subject": subject, "error": str(error)}
        )
        errors.append(source)
        return None


@ray.remote(num_cpus=0.25)
def crawl_package(name: str, settings: CrawlSettings, limiter) -> dict[str, Any]:
    """Collect one package's releases, per-snapshot dependencies and repository link.

    Args:
        name: Package name.
        settings: Crawl settings.
        limiter: Handle of the rate-limiter actor.

    Returns:
        ``name``, ``found``, ``latest_version``, ``releases``, ``requires`` (version to
        dependency list), ``repository`` (or ``None``), ``errors`` and request counters.
    """
    client = _client(settings, limiter)
    cache = ResponseCache(settings.cache_dir, settings.offline)
    base, errors = settings.sources["pypi_base_url"], []
    meta = _attempt(errors, "pypi", name, lambda: pypi.fetch_package(client, cache, base, name))
    result = {"name": name, "found": meta is not None, "errors": errors}
    if meta is not None:
        snapshots = [parse_timestamp(snapshot) for snapshot in settings.snapshots]
        requires = {meta["version"]: pypi.required_packages(meta["requires_dist"])}
        for version in sorted(versions_current_at(meta["releases"], snapshots) - set(requires)):
            requires[version] = _attempt(
                errors,
                "pypi_version",
                f"{name}=={version}",
                lambda v=version: pypi.fetch_version_requires(client, cache, base, name, v),
            )
        repository = first_repository(candidate_urls(meta["project_urls"], meta["home_page"]))
        if repository is None:
            linked = _attempt(
                errors,
                "ecosystems_package",
                name,
                lambda: ecosystems.fetch_repository_url(
                    client, cache, settings.sources["ecosystems_packages_url"], name
                ),
            )
            repository = first_repository([linked])
        result.update(
            latest_version=meta["version"],
            releases=meta["releases"],
            requires=requires,
            repository=vars(repository) if repository else None,
        )
    return {**result, "requests": client.requests_sent, "cache_hits": cache.hits}


@ray.remote(num_cpus=1)
def crawl_repository(repo: dict[str, str], settings: CrawlSettings, limiter) -> dict[str, Any]:
    """Collect one repository's commits, archive status and issues.

    The commit and issue rows stay in the response cache, so the driver never holds
    millions of rows in memory; this returns only what the package table needs.

    Args:
        repo: ``host``, ``owner`` and ``name`` of the repository.
        settings: Crawl settings.
        limiter: Handle of the rate-limiter actor.

    Returns:
        ``url``, ``has_commits``, ``metadata``, ``issues`` (``complete_since``),
        ``archived_at_exact``, ``errors`` and request counters. Missing parts are ``None``.
    """
    repository = Repository(**repo)
    client = _client(settings, limiter)
    cache = ResponseCache(settings.cache_dir, settings.offline)
    crawl, sources, url, errors = settings.crawl, settings.sources, repository.url, []
    markers = list(settings.bot_markers)

    def commits():
        return git_history.fetch_commits(
            cache, repository, crawl["history_start"], crawl["clone_timeout_seconds"], markers
        )

    def metadata():
        return ecosystems.fetch_repository_metadata(
            client, cache, sources["ecosystems_repos_url"], repository
        )

    def issues():
        return ecosystems.fetch_issues(
            client,
            cache,
            sources["ecosystems_issues_url"],
            repository,
            crawl["history_start"],
            crawl["issues_page_size"],
            crawl["max_issue_pages"],
            markers,
        )

    has_commits = _attempt(errors, "git", url, commits) is not None
    meta = _attempt(errors, "ecosystems_repo", url, metadata)
    issue_history = _attempt(errors, "ecosystems_issues", url, issues)
    archived_at = None
    if (meta or {}).get("archived") and repository.host == "github.com" and settings.github_token:
        graphql = _client(settings, limiter, {"Authorization": f"bearer {settings.github_token}"})
        archived_at = _attempt(
            errors,
            "github_graphql",
            url,
            lambda: github_graphql.fetch_archived_at(
                graphql, cache, sources["github_graphql_url"], repository
            ),
        )
        client.requests_sent += graphql.requests_sent
    return {
        "url": url,
        "has_commits": has_commits,
        "metadata": meta,
        "issues": issue_history and {"complete_since": issue_history["complete_since"]},
        "archived_at_exact": archived_at,
        "errors": errors,
        "requests": client.requests_sent,
        "cache_hits": cache.hits,
    }


def _drain(pending: dict, on_result: Callable[[dict], None]) -> None:
    while pending:
        done, _ = ray.wait(list(pending), num_returns=1)
        on_result(ray.get(done[0]))
        del pending[done[0]]


def crawl(
    seeds: list[str], settings: CrawlSettings, profile: dict[str, Any]
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """Crawl the seed packages, their dependencies (if enabled) and their repositories.

    Args:
        seeds: Package names to start from.
        settings: Crawl settings.
        profile: Active data profile (``expand_dependencies``, ``max_packages``).

    Returns:
        Package results, repository results and crawl statistics.
    """
    rates = NO_LIMIT if settings.offline else settings.sources["rate_limits"]
    limiter = ray.remote(RateLimiter).remote(rates)
    started = time.monotonic()
    seen, packages = set(seeds), []
    pending = {crawl_package.remote(name, settings, limiter): name for name in seeds}

    def on_package(result: dict) -> None:
        packages.append(result)
        if len(packages) % 100 == 0:
            logger.info("Crawled packages", extra={"done": len(packages), "queued": len(pending)})
        if not profile["expand_dependencies"]:
            return
        for dependency in sorted(
            {d for deps in result.get("requires", {}).values() for d in deps or []} - seen
        ):
            if len(seen) >= profile["max_packages"]:
                return
            seen.add(dependency)
            pending[crawl_package.remote(dependency, settings, limiter)] = dependency

    _drain(pending, on_package)
    package_seconds = time.monotonic() - started

    repos = {
        tuple(p["repository"].values()): p["repository"] for p in packages if p.get("repository")
    }
    repositories: list[dict] = []
    _drain(
        {crawl_repository.remote(r, settings, limiter): r for r in repos.values()},
        repositories.append,
    )

    results = [*packages, *repositories]
    stats = {
        "packages": len(packages),
        "repositories": len(repositories),
        "http_requests": sum(r["requests"] for r in results),
        "cache_hits": sum(r["cache_hits"] for r in results),
        "errors_by_source": dict(Counter(source for r in results for source in r["errors"])),
        "duration_seconds": round(time.monotonic() - started, 2),
        "packages_per_minute": round(len(packages) / max(package_seconds, 1e-9) * 60, 1),
    }
    return packages, repositories, stats


def build_settings(params: dict[str, Any], profile: dict[str, Any]) -> CrawlSettings:
    """Assemble crawl settings from parameters and the environment.

    Args:
        params: The full parameter dictionary.
        profile: Active data profile.

    Returns:
        The crawl settings.
    """
    snapshots = params["snapshots"]
    return CrawlSettings(
        cache_dir=profile["cache_dir"],
        offline=profile["offline"],
        sources={**params["sources"], "rate_limits": params["rate_limits"]},
        snapshots=(
            *snapshots["train"],
            *snapshots["validation"],
            *snapshots["test"],
            snapshots["scoring"],
        ),
        crawl=params["crawl"],
        bot_markers=tuple(params["activity"]["bot_author_markers"]),
        github_token=os.environ.get("GITHUB_TOKEN", ""),
    )


def seed_packages(
    params: dict[str, Any], profile: dict[str, Any], settings: CrawlSettings
) -> list[str]:
    """Return the packages the crawl starts from: the fixture list or the top packages.

    Args:
        params: The full parameter dictionary.
        profile: Active data profile.
        settings: Crawl settings (for the cache and HTTP timeout).

    Returns:
        Canonical package names.
    """
    if profile["seed_source"] == "fixture_list":
        return list(params["fixture_packages"])[: profile["max_seed_packages"]]
    client = HttpClient(settings.sources["request_timeout_seconds"])
    cache = ResponseCache(settings.cache_dir, settings.offline)
    url = settings.sources["top_packages_url"]
    return top_package_names(client, cache, url, profile["max_seed_packages"])
