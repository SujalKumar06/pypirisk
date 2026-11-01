"""Run a pipeline stage and record the run for the Pipeline Console.

Every stage CLI calls :func:`run_stage`. Each run appends one JSON line to
``<PIPELINE_LOG_DIR>/pipeline_runs.jsonl`` (stage, status, duration, error and traceback),
which the Pipeline Console shows as the history of successful and failed runs.
"""

import argparse
import json
import logging
import os
import time
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pypirisk.config import DEFAULT_PARAMS_PATH, load_params
from pypirisk.logs import configure_logging

logger = logging.getLogger(__name__)

LOG_DIR_ENV = "PIPELINE_LOG_DIR"


def runs_path() -> Path:
    """Return the run-record file, ``logs/pipeline_runs.jsonl`` unless overridden."""
    return Path(os.environ.get(LOG_DIR_ENV, "logs")) / "pipeline_runs.jsonl"


def _append(record: dict[str, Any]) -> None:
    path = runs_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


@contextmanager
def stage_run(stage: str, profile: str) -> Iterator[None]:
    """Record one run of ``stage``, whether it succeeds or fails.

    Args:
        stage: Stage name as in ``dvc.yaml``.
        profile: Active data profile.

    Yields:
        Nothing; the stage body runs inside the ``with`` block.

    Raises:
        BaseException: Whatever the stage raised, after it has been recorded.
    """
    started_at = datetime.now(UTC)
    started = time.monotonic()
    record: dict[str, Any] = {
        "stage": stage,
        "profile": profile,
        "started_at": started_at.isoformat(),
        "status": "success",
    }
    try:
        yield
    except BaseException as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        record["traceback"] = traceback.format_exc()[-4000:]
        raise
    finally:
        duration = time.monotonic() - started
        record["duration_seconds"] = round(duration, 3)
        _append(record)


def run_stage(stage: str, body: Callable[[dict[str, Any]], None]) -> None:
    """Parse ``--params``, configure logging and run ``body`` inside :func:`stage_run`.

    Args:
        stage: Stage name as in ``dvc.yaml``.
        body: The stage logic; receives the full parameter dictionary.
    """
    parser = argparse.ArgumentParser(description=f"Run the {stage} stage.")
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    args = parser.parse_args()
    configure_logging()
    params = load_params(args.params)
    with stage_run(stage, params.get("profile", "unknown")):
        body(params)
