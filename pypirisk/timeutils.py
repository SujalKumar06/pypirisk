"""Date helpers for snapshot windows."""

import calendar
from datetime import UTC, datetime


def parse_timestamp(value: str) -> datetime:
    """Parse an ISO 8601 date or timestamp into a timezone-aware UTC datetime.

    Args:
        value: A string such as ``"2021-01-01"`` or ``"2021-01-01T10:00:00Z"``.

    Returns:
        The parsed datetime in UTC. Naive inputs are assumed to be UTC.

    Raises:
        ValueError: If ``value`` is not a valid ISO 8601 string.
    """
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def add_months(moment: datetime, months: int) -> datetime:
    """Shift a datetime by a whole number of calendar months.

    The day is clamped to the last day of the target month, so 31 January plus one
    month is 28 or 29 February.

    Args:
        moment: The starting datetime.
        months: Number of months to add; negative values go back in time.

    Returns:
        The shifted datetime, keeping the time of day and timezone.
    """
    month_index = moment.month - 1 + months
    year = moment.year + month_index // 12
    month = month_index % 12 + 1
    day = min(moment.day, calendar.monthrange(year, month)[1])
    return moment.replace(year=year, month=month, day=day)
