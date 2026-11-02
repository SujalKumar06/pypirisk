"""Snapshot splits and the checks that keep them free of temporal leakage."""

from datetime import datetime

from pypirisk.timeutils import add_months, parse_timestamp

SPLIT_ORDER = ("train", "validation", "test")


def all_snapshots(splits: dict[str, list[str]]) -> list[str]:
    """List the labelled snapshots of every split in chronological split order.

    Args:
        splits: The ``snapshots`` section of ``params.yaml``.

    Returns:
        Snapshot dates of the train, validation and test splits.
    """
    return [snapshot for split in SPLIT_ORDER for snapshot in splits[split]]


def every_snapshot(splits: dict[str, list[str]]) -> list[str]:
    """List the labelled snapshots followed by the scoring snapshot.

    Args:
        splits: The ``snapshots`` section of ``params.yaml``.

    Returns:
        Every snapshot date the graph and feature stages compute.
    """
    return [*all_snapshots(splits), splits["scoring"]]


def check_split_windows(splits: dict[str, list[str]], outcome_months: int) -> None:
    """Ensure no split's outcome windows reach into a later split.

    Yearly snapshots with 18-month outcomes overlap. If a training snapshot's outcome
    window runs past the first test snapshot, the model would learn from events that
    happen during the test period.

    Args:
        splits: The ``snapshots`` section of ``params.yaml``.
        outcome_months: Length of each outcome window.

    Raises:
        ValueError: If a split is empty or overlaps the next one.
    """
    for earlier, later in zip(SPLIT_ORDER, SPLIT_ORDER[1:], strict=False):
        if not splits[earlier] or not splits[later]:
            raise ValueError(f"Split {earlier!r} or {later!r} has no snapshots")
        last_outcome_end = add_months(max(map(parse_timestamp, splits[earlier])), outcome_months)
        next_start = min(map(parse_timestamp, splits[later]))
        if last_outcome_end > next_start:
            raise ValueError(
                f"Outcome window of split {earlier!r} ends {last_outcome_end.date()}, "
                f"after split {later!r} starts {next_start.date()}"
            )


def check_matured(splits: dict[str, list[str]], outcome_months: int, as_of: datetime) -> None:
    """Ensure every labelled snapshot's outcome window has finished.

    Args:
        splits: The ``snapshots`` section of ``params.yaml``.
        outcome_months: Length of each outcome window.
        as_of: The time the data was collected.

    Raises:
        ValueError: If any outcome window ends after ``as_of``.
    """
    for snapshot in all_snapshots(splits):
        outcome_end = add_months(parse_timestamp(snapshot), outcome_months)
        if outcome_end > as_of:
            raise ValueError(
                f"Snapshot {snapshot} has not matured: its outcome window ends "
                f"{outcome_end.date()}, after {as_of.date()}"
            )
