"""PyPI JSON API: release history and per-version dependencies."""

from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from pypirisk.ingest.cache import ResponseCache, cache_key
from pypirisk.ingest.http import HttpClient

HOST = "pypi"


def trim_package(raw: dict[str, Any]) -> dict[str, Any]:
    """Keep only the fields of a package response the pipeline uses.

    Full responses list every uploaded file and reach megabytes for packages like numpy.

    Args:
        raw: A response from ``/pypi/<name>/json``.

    Returns:
        ``name``, ``version`` (latest), ``requires_dist``, ``project_urls``,
        ``home_page`` and ``releases``: each release's version and earliest upload
        time, oldest first. Releases without files are dropped.
    """
    info = raw["info"]
    releases = [
        {"version": version, "uploaded_at": min(f["upload_time_iso_8601"] for f in files)}
        for version, files in raw["releases"].items()
        if files
    ]
    releases.sort(key=lambda release: release["uploaded_at"])
    return {
        "name": info["name"],
        "version": info["version"],
        "requires_dist": info.get("requires_dist") or [],
        "project_urls": info.get("project_urls") or {},
        "home_page": info.get("home_page") or None,
        "releases": releases,
    }


def required_packages(requires_dist: list[str]) -> list[str]:
    """Return the packages a default install needs, from ``Requires-Dist`` entries.

    Optional dependencies (behind an ``extra == ...`` marker) and unparseable entries
    are ignored.

    Args:
        requires_dist: ``Requires-Dist`` strings, e.g. ``["idna<4,>=2.5"]``.

    Returns:
        Sorted, de-duplicated canonical package names.
    """
    names = set()
    for entry in requires_dist:
        try:
            requirement = Requirement(entry)
        except InvalidRequirement:
            continue
        if requirement.marker is None or "extra" not in str(requirement.marker):
            names.add(canonicalize_name(requirement.name))
    return sorted(names)


def fetch_package(
    client: HttpClient, cache: ResponseCache, base_url: str, name: str
) -> dict[str, Any] | None:
    """Fetch a package's trimmed metadata, from the cache when possible.

    Args:
        client: HTTP client.
        cache: Response cache.
        base_url: PyPI JSON API root.
        name: Package name.

    Returns:
        The trimmed metadata, or ``None`` if the package does not exist.
    """

    def load() -> dict[str, Any] | None:
        raw = client.get_json(HOST, f"{base_url}/{name}/json")
        return trim_package(raw) if raw else {"missing": True}

    value = cache.fetch("pypi", cache_key(name), load)
    return None if value.get("missing") else value


def fetch_version_requires(
    client: HttpClient, cache: ResponseCache, base_url: str, name: str, version: str
) -> list[str] | None:
    """Fetch the dependencies one release declared.

    The graph is rebuilt per snapshot from the release that was current then, because
    today's dependencies would leak the future into historical features.

    Args:
        client: HTTP client.
        cache: Response cache.
        base_url: PyPI JSON API root.
        name: Package name.
        version: Release version.

    Returns:
        Canonical names of the release's required packages, or ``None`` if PyPI has no
        metadata for that release.
    """

    def load() -> dict[str, Any]:
        raw = client.get_json(HOST, f"{base_url}/{name}/{version}/json")
        if raw is None:
            return {"missing": True}
        return {"requires": required_packages(raw["info"].get("requires_dist") or [])}

    value = cache.fetch("pypi_version", cache_key(name, version), load)
    return None if value.get("missing") else value["requires"]
