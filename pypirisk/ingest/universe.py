"""Choose which packages to crawl."""

from datetime import datetime
from typing import Any

from packaging.utils import canonicalize_name

from pypirisk.ingest.cache import ResponseCache
from pypirisk.ingest.http import HttpClient
from pypirisk.timeutils import parse_timestamp


def top_package_names(client: HttpClient, cache: ResponseCache, url: str, limit: int) -> list[str]:
    """Return the most downloaded PyPI packages.

    The list is cached on first use, so later runs crawl the same universe even though
    the published ranking changes daily.

    Args:
        client: HTTP client.
        cache: Response cache.
        url: The top-pypi-packages JSON (30-day download ranking).
        limit: How many names to return.

    Returns:
        Canonical names, most downloaded first.
    """

    def load() -> dict[str, Any]:
        rows = client.get_json("top_packages", url)["rows"]
        return {"names": [canonicalize_name(row["project"]) for row in rows]}

    return cache.fetch("top_packages", "top", load)["names"][:limit]


def versions_current_at(releases: list[dict[str, str]], snapshots: list[datetime]) -> set[str]:
    """Return the release that was the newest one at each snapshot.

    Args:
        releases: Releases with ``version`` and ``uploaded_at``, oldest first.
        snapshots: Snapshot times.

    Returns:
        The versions that were current at one or more snapshots.
    """
    times = [(parse_timestamp(release["uploaded_at"]), release["version"]) for release in releases]
    current = set()
    for snapshot in snapshots:
        before = [version for uploaded, version in times if uploaded < snapshot]
        if before:
            current.add(before[-1])
    return current
