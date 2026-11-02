"""GitHub GraphQL: exact archive dates, used only when ``GITHUB_TOKEN`` is set.

ecosyste.ms reports whether a repository is archived but not when. With a token, this
crawler asks GitHub for ``archivedAt``; without one, the archive date is approximated
by the last push (see ``docs/HLD.md``).
"""

from typing import Any

from pypirisk.ingest.cache import ResponseCache, cache_key
from pypirisk.ingest.http import HttpClient
from pypirisk.ingest.sources.repos import Repository

HOST = "github_graphql"

ARCHIVED_AT_QUERY = """
query($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) { isArchived archivedAt }
}
"""


def parse_archived_at(response: dict[str, Any]) -> str | None:
    """Extract ``archivedAt`` from a GraphQL response.

    Args:
        response: The decoded GraphQL response.

    Returns:
        The archive timestamp, or ``None`` if not archived or not found.
    """
    repository = (response.get("data") or {}).get("repository") or {}
    return repository.get("archivedAt") if repository.get("isArchived") else None


def fetch_archived_at(
    client: HttpClient, cache: ResponseCache, graphql_url: str, repository: Repository
) -> str | None:
    """Return when a GitHub repository was archived.

    Args:
        client: HTTP client carrying the ``Authorization`` header.
        cache: Response cache.
        graphql_url: GitHub GraphQL endpoint.
        repository: A ``github.com`` repository.

    Returns:
        The archive timestamp, or ``None``.
    """

    def load() -> dict[str, Any]:
        body = {
            "query": ARCHIVED_AT_QUERY,
            "variables": {"owner": repository.owner, "name": repository.name},
        }
        return {"archived_at": parse_archived_at(client.post_json(HOST, graphql_url, body))}

    return cache.fetch("github_archived", cache_key(repository.url), load)["archived_at"]
