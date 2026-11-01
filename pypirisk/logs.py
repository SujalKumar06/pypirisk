"""Structured logging shared by pipeline stages and services."""

import json
import logging
import os
from datetime import UTC, datetime

_RESERVED_ATTRS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


def _extra_fields(record: logging.LogRecord) -> dict:
    return {key: value for key, value in vars(record).items() if key not in _RESERVED_ATTRS}


class JsonFormatter(logging.Formatter):
    """Format log records as one JSON object per line.

    Fields passed through ``extra=`` are added to the object, so callers can attach
    context such as a package name or a stage without building strings.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Render ``record`` as a JSON string.

        Args:
            record: The log record to format.

        Returns:
            A single-line JSON document.
        """
        payload = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(_extra_fields(record))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    """Format log records as readable text, with ``extra=`` fields appended as ``key=value``."""

    def __init__(self) -> None:
        """Use a timestamp, level and logger name prefix."""
        super().__init__("%(asctime)s %(levelname)s %(name)s: %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        """Render ``record`` as one line of text.

        Args:
            record: The log record to format.

        Returns:
            The formatted message followed by any extra fields.
        """
        line = super().format(record)
        extras = " ".join(f"{key}={value}" for key, value in _extra_fields(record).items())
        return f"{line} {extras}" if extras else line


def configure_logging(level: str | None = None, json_format: bool | None = None) -> None:
    """Configure the root logger once per process.

    Containers set ``LOG_FORMAT=json`` so logs are machine-readable; local runs default
    to plain text, which is easier to read in a terminal.

    Args:
        level: Log level name. Defaults to the ``LOG_LEVEL`` env var, then ``INFO``.
        json_format: Emit JSON lines. Defaults to ``LOG_FORMAT == "json"``.
    """
    level = level or os.environ.get("LOG_LEVEL", "INFO")
    if json_format is None:
        json_format = os.environ.get("LOG_FORMAT", "text").lower() == "json"

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter() if json_format else TextFormatter())
    logging.basicConfig(level=level.upper(), handlers=[handler], force=True)
