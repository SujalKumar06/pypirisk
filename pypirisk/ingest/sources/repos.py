"""Normalise repository URLs and identify their hosting service."""

import re
from dataclasses import dataclass

# host/owner/repo at the start of a URL; anything after (tree/main, issues, .git) is dropped.
# AI-generated (Claude Code). Prompt (condensed): "A regex that takes the host, owner and
# repository name from a GitHub, GitLab, Bitbucket or Codeberg URL and ignores the rest."
_REPO_URL = re.compile(
    r"^(?:git\+)?https?://(?:www\.)?(github\.com|gitlab\.com|bitbucket\.org|codeberg\.org)"
    r"/([\w.-]+)/([\w.-]+)",
    re.IGNORECASE,
)

# Paths on code hosts that look like owner/repo but are not repositories.
RESERVED_OWNERS = {"sponsors", "orgs", "apps", "marketplace", "features", "topics", "about"}

# project_urls labels that usually point at the source repository, checked first.
# AI-generated (Claude Code). Prompt (condensed): "A case-insensitive regex for PyPI
# project_urls labels that usually mean the source repository."
_SOURCE_LABEL = re.compile(r"source|repo|code|github|gitlab|git", re.IGNORECASE)

# ecosyste.ms names hosts by service, not by domain.
ECOSYSTEMS_HOSTS = {"github.com": "GitHub", "gitlab.com": "GitLab", "codeberg.org": "Codeberg"}


@dataclass(frozen=True)
class Repository:
    """A source repository.

    Attributes:
        host: Domain such as ``github.com``.
        owner: Account or organisation.
        name: Repository name.
    """

    host: str
    owner: str
    name: str

    @property
    def url(self) -> str:
        """Canonical ``https://host/owner/name`` URL."""
        return f"https://{self.host}/{self.owner}/{self.name}"

    @property
    def full_name(self) -> str:
        """``owner/name``."""
        return f"{self.owner}/{self.name}"


def parse_repository(url: str | None) -> Repository | None:
    """Parse a repository URL from package metadata.

    Args:
        url: Any URL; only GitHub, GitLab, Bitbucket and Codeberg repositories match.

    Returns:
        The repository in lowercase, or ``None`` if the URL is not a known code host.
    """
    match = _REPO_URL.match((url or "").strip())
    if not match:
        return None
    host, owner, name = (part.lower() for part in match.groups())
    if owner in RESERVED_OWNERS:
        return None
    return Repository(host, owner, name.removesuffix(".git"))


def first_repository(urls: list[str | None]) -> Repository | None:
    """Return the first URL in ``urls`` that is a repository.

    Args:
        urls: Candidate URLs in order of preference.

    Returns:
        The parsed repository, or ``None``.
    """
    for url in urls:
        repository = parse_repository(url)
        if repository:
            return repository
    return None


def candidate_urls(project_urls: dict[str, str], home_page: str | None) -> list[str | None]:
    """Order a package's URLs so the likely source repository comes first.

    Args:
        project_urls: The ``project_urls`` mapping from PyPI metadata.
        home_page: The legacy ``home_page`` field.

    Returns:
        URLs labelled like a source link, then the home page, then the rest.
    """
    labelled = [url for label, url in project_urls.items() if _SOURCE_LABEL.search(label)]
    others = [url for label, url in project_urls.items() if not _SOURCE_LABEL.search(label)]
    return [*labelled, home_page, *others]
