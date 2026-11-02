"""Commit history from commit-only git clones.

A treeless bare clone (``--filter=tree:0``) downloads commits but no file contents, so
even large repositories take seconds and a few megabytes. Using git instead of the
GitHub API costs no rate limit and works for any code host.
"""

import hashlib
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from pypirisk.features.activity import is_bot_author
from pypirisk.ingest.cache import ResponseCache, cache_key
from pypirisk.ingest.sources.repos import Repository

_LOG_FORMAT = "%aI%x09%ae%x09%an"


class GitError(RuntimeError):
    """Raised when a clone or ``git log`` fails or times out."""


def author_id(email: str) -> str:
    """Return a stable pseudonymous ID for a commit author.

    Bus-factor features only need to tell authors apart, so emails are hashed rather
    than stored; the project never ranks or names individual maintainers.

    Args:
        email: Commit author email.

    Returns:
        The first 12 hex digits of the SHA-1 of the lowercased email.
    """
    return hashlib.sha1(email.strip().lower().encode(), usedforsecurity=False).hexdigest()[:12]


def parse_log(output: str, bot_markers: list[str]) -> list[list]:
    r"""Parse ``git log`` output in the module's tab-separated format.

    Args:
        output: Lines of ``<ISO date>\\t<email>\\t<name>``.
        bot_markers: Substrings that identify bot authors.

    Returns:
        ``[authored_at, author_id, is_bot]`` per commit, in log order.
    """
    commits = []
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        date, email, name = parts
        is_bot = is_bot_author(name, bot_markers) or is_bot_author(email, bot_markers)
        commits.append([date, author_id(email), is_bot])
    return commits


def _run_git(args: list[str], timeout: float) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
            env={"GIT_TERMINAL_PROMPT": "0", "PATH": "/usr/bin:/bin:/usr/local/bin"},
        )
    except subprocess.TimeoutExpired as error:
        raise GitError(f"git {args[0]} timed out after {timeout}s") from error
    except subprocess.CalledProcessError as error:
        raise GitError(f"git {args[0]} failed: {error.stderr.strip()[:300]}") from error
    return result.stdout


def clone_history(repository: Repository, since: str, timeout: float, bot_markers: list[str]):
    """Clone a repository without file contents and read its commit log.

    Args:
        repository: The repository.
        since: ISO date; older commits are not read.
        timeout: Seconds allowed for the clone and for the log.
        bot_markers: Substrings that identify bot authors.

    Returns:
        Compact commit rows, see :func:`parse_log`.

    Raises:
        GitError: If cloning or reading the log fails.
    """
    with tempfile.TemporaryDirectory(prefix="pypirisk-clone-") as workdir:
        target = str(Path(workdir) / "repo.git")
        _run_git(
            ["clone", "--quiet", "--bare", "--filter=tree:0", "--no-tags", repository.url, target],
            timeout,
        )
        output = _run_git(
            ["-C", target, "log", f"--since={since}", f"--format={_LOG_FORMAT}"], timeout
        )
    return parse_log(output, bot_markers)


def fetch_commits(
    cache: ResponseCache,
    repository: Repository,
    since: str,
    timeout: float,
    bot_markers: list[str],
) -> list[list[Any]]:
    """Return a repository's commit history, cloning only on a cache miss.

    Args:
        cache: Response cache.
        repository: The repository.
        since: ISO date; older commits are not read.
        timeout: Seconds allowed per git command.
        bot_markers: Substrings that identify bot authors.

    Returns:
        Compact commit rows.

    Raises:
        GitError: If cloning fails.
        CacheMissError: In offline mode when the history is not cached.
    """
    value = cache.fetch(
        "git",
        cache_key(repository.url),
        lambda: {"commits": clone_history(repository, since, timeout, bot_markers)},
    )
    return value["commits"]
