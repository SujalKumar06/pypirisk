"""Row-level checks for each ingest table.

Each check takes a pandas batch and returns the reason each row is invalid ("" when
valid), so the same function runs on a Ray Data batch and in a unit test.
"""

import pandas as pd

EARLIEST_EVENT = pd.Timestamp("2000-01-01", tz="UTC")
REASON_COLUMN = "invalid_reason"


def _reasons(index: pd.Index, rules: list[tuple[str, pd.Series]]) -> pd.Series:
    reasons = pd.Series("", index=index, dtype="object")
    for reason, broken in rules:
        reasons = reasons.mask((reasons == "") & broken.fillna(True), reason)
    return reasons


def check_releases(batch: pd.DataFrame, packages: set[str], now: pd.Timestamp) -> pd.Series:
    """Check release rows.

    Args:
        batch: Rows of the ``releases`` table.
        packages: Names in the ``packages`` table.
        now: Upper bound for plausible timestamps.

    Returns:
        The invalid reason per row.
    """
    return _reasons(
        batch.index,
        [
            ("unknown package", ~batch["name"].isin(packages)),
            ("missing upload time", batch["uploaded_at"].isna()),
            ("upload time out of range", ~batch["uploaded_at"].between(EARLIEST_EVENT, now)),
        ],
    )


def check_requires(batch: pd.DataFrame, packages: set[str]) -> pd.Series:
    """Check dependency rows.

    Args:
        batch: Rows of the ``requires`` table.
        packages: Names in the ``packages`` table.

    Returns:
        The invalid reason per row.
    """
    return _reasons(
        batch.index,
        [
            ("unknown package", ~batch["name"].isin(packages)),
            ("missing dependency", batch["dependency"].isna() | (batch["dependency"] == "")),
        ],
    )


def check_commits(batch: pd.DataFrame, now: pd.Timestamp) -> pd.Series:
    """Check commit rows; misconfigured clocks produce commits dated 1970 or the future.

    Args:
        batch: Rows of the ``commits`` table.
        now: Upper bound for plausible timestamps.

    Returns:
        The invalid reason per row.
    """
    return _reasons(
        batch.index,
        [
            ("missing author", batch["author_id"].isna()),
            ("commit date out of range", ~batch["authored_at"].between(EARLIEST_EVENT, now)),
        ],
    )


def check_issues(batch: pd.DataFrame, now: pd.Timestamp) -> pd.Series:
    """Check issue rows.

    Args:
        batch: Rows of the ``issues`` table.
        now: Upper bound for plausible timestamps.

    Returns:
        The invalid reason per row.
    """
    closed = batch["closed_at"]
    return _reasons(
        batch.index,
        [
            ("created date out of range", ~batch["created_at"].between(EARLIEST_EVENT, now)),
            ("closed before created", closed.notna() & (closed < batch["created_at"])),
        ],
    )


def check_packages(packages: pd.DataFrame) -> pd.Series:
    """Check the package table.

    Args:
        packages: The ``packages`` table.

    Returns:
        The invalid reason per row.
    """
    return _reasons(
        packages.index,
        [
            ("missing name", packages["name"].isna() | (packages["name"] == "")),
            ("duplicate name", packages["name"].duplicated(keep=False)),
        ],
    )
