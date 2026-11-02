"""ecosyste.ms APIs: repository links, archive status and issue history.

These APIs need no token. The anonymous limit is 5,000 requests per hour, which the
shared rate limiter respects.
"""

from typing import Any
from urllib.parse import quote

from pypirisk.features.activity import is_bot_author
from pypirisk.ingest.cache import ResponseCache, cache_key
from pypirisk.ingest.http import HttpClient
from pypirisk.ingest.sources.repos import ECOSYSTEMS_HOSTS, Repository

HOST = "ecosystems"


def fetch_repository_url(
    client: HttpClient, cache: ResponseCache, packages_url: str, name: str
) -> str | None:
    """Return the repository URL ecosyste.ms has resolved for a PyPI package.

    Asked only when PyPI metadata links no repository, to save rate limit.

    Args:
        client: HTTP client.
        cache: Response cache.
        packages_url: ``.../registries/pypi.org/packages`` endpoint.
        name: Package name.

    Returns:
        The repository URL, or ``None`` if unknown.
    """

    def load() -> dict[str, Any]:
        raw = client.get_json(HOST, f"{packages_url}/{quote(name)}")
        return {"repository_url": (raw or {}).get("repository_url")}

    return cache.fetch("ecosystems_package", cache_key(name), load)["repository_url"]


def fetch_repository_metadata(
    client: HttpClient, cache: ResponseCache, repos_url: str, repository: Repository
) -> dict[str, Any] | None:
    """Return a repository's archive flag and last push time.

    Args:
        client: HTTP client.
        cache: Response cache.
        repos_url: ``https://repos.ecosyste.ms/api/v1/hosts`` endpoint.
        repository: The repository.

    Returns:
        ``archived`` and ``pushed_at``, or ``None`` if the host is not indexed or the
        repository is unknown.
    """
    service = ECOSYSTEMS_HOSTS.get(repository.host)
    if service is None:
        return None

    def load() -> dict[str, Any]:
        url = f"{repos_url}/{service}/repositories/{quote(repository.full_name, safe='')}"
        raw = client.get_json(HOST, url)
        if raw is None:
            return {"missing": True}
        return {"archived": bool(raw.get("archived")), "pushed_at": raw.get("pushed_at")}

    value = cache.fetch("ecosystems_repo", cache_key(repository.url), load)
    return None if value.get("missing") else value


def compact_issue(issue: dict[str, Any], bot_markers: list[str]) -> list:
    """Reduce an issue or pull request to the fields the features use.

    Args:
        issue: One item from the ecosyste.ms issues endpoint.
        bot_markers: Substrings that identify bot authors.

    Returns:
        ``[created_at, closed_at, is_pull_request, is_bot]``.
    """
    return [
        issue["created_at"],
        issue.get("closed_at"),
        bool(issue.get("pull_request")),
        is_bot_author(issue.get("user") or "", bot_markers),
    ]


def fetch_issues(
    client: HttpClient,
    cache: ResponseCache,
    issues_url: str,
    repository: Repository,
    history_start: str,
    page_size: int,
    max_pages: int,
    bot_markers: list[str],
) -> dict[str, Any] | None:
    """Download issue and pull-request history, newest first, back to ``history_start``.

    Busy repositories can have tens of thousands of issues, so at most ``max_pages`` are
    read. When the cap stops the download early, ``complete_since`` records the oldest
    date that is fully covered, and older snapshots get no issue features.

    Args:
        client: HTTP client.
        cache: Response cache.
        issues_url: ``https://issues.ecosyste.ms/api/v1/hosts`` endpoint.
        repository: The repository.
        history_start: ISO date; older issues are not needed.
        page_size: Issues per request (ecosyste.ms allows up to 1,000).
        max_pages: Request cap per repository.
        bot_markers: Substrings that identify bot authors.

    Returns:
        ``issues`` (compact rows), ``truncated`` and ``complete_since``, or ``None`` if
        the host is not indexed or the repository is unknown.
    """
    service = ECOSYSTEMS_HOSTS.get(repository.host)
    if service is None:
        return None

    def load() -> dict[str, Any]:
        url = f"{issues_url}/{service}/repositories/{quote(repository.full_name, safe='')}/issues"
        issues: list[list] = []
        for page in range(1, max_pages + 1):
            params = {"per_page": page_size, "page": page, "sort": "created_at", "order": "desc"}
            batch = client.get_json(HOST, url, params=params)
            if batch is None and page == 1:
                return {"missing": True}
            if batch is not None and not isinstance(batch, list):
                # ecosyste.ms answers some repositories with an error object, not a list.
                raise ValueError(
                    f"unexpected issues response for {repository.url}: {str(batch)[:200]}"
                )
            batch = batch or []
            issues.extend(compact_issue(issue, bot_markers) for issue in batch)
            reached_start = batch and batch[-1]["created_at"] < history_start
            if len(batch) < page_size or reached_start:
                return {"issues": issues, "truncated": False, "complete_since": history_start}
        oldest = issues[-1][0] if issues else history_start
        return {"issues": issues, "truncated": True, "complete_since": oldest}

    value = cache.fetch("ecosystems_issues", cache_key(repository.url), load)
    return None if value.get("missing") else value
