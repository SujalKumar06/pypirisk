"""Lineage information logged with every MLflow run.

A run is reproducible from the Git commit plus the MLflow run ID, so every training run
records the commit, whether the working tree had uncommitted changes, and the MD5 of
each input file (the same hash DVC stores in ``dvc.lock``).
"""

import hashlib
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

UNKNOWN = "unknown"


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        logger.warning("git command failed", extra={"git_args": args, "error": str(error)})
        return None
    return result.stdout.strip()


def git_commit() -> str:
    """Return the current Git commit hash.

    Returns:
        The full hash of ``HEAD``, or ``"unknown"`` outside a Git checkout.
    """
    return _git("rev-parse", "HEAD") or UNKNOWN


def git_is_dirty() -> bool:
    """Tell whether tracked files have uncommitted changes.

    Returns:
        ``True`` if ``git status`` reports modified tracked files, or if Git is unavailable.
    """
    status = _git("status", "--porcelain", "--untracked-files=no")
    return status is None or bool(status)


def file_md5(path: Path) -> str:
    """Compute the MD5 of a file's contents.

    Args:
        path: File to hash.

    Returns:
        The hex digest.
    """
    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lineage_tags(data_paths: dict[str, Path]) -> dict[str, str]:
    """Build the MLflow tags that tie a run to its code and data.

    Args:
        data_paths: Input files keyed by a short role name such as ``"features"``.

    Returns:
        Tags ``git_commit``, ``git_dirty`` and ``data_md5.<role>`` for each input.
    """
    tags = {"git_commit": git_commit(), "git_dirty": str(git_is_dirty()).lower()}
    tags.update({f"data_md5.{role}": file_md5(path) for role, path in data_paths.items()})
    return tags
