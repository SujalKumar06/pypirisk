"""Validate stage: ``python -m pypirisk.validate``.

The large tables (releases, requires, commits, issues) are checked with Ray Data in
streaming batches; the package table is small and checked with pandas. Invalid rows
are dropped and counted by reason. The stage fails if too many rows are invalid.
"""

import logging
import shutil
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import ray

from pypirisk.io import write_json
from pypirisk.rayutil import local_ray
from pypirisk.stagelog import run_stage
from pypirisk.validate import checks

logger = logging.getLogger("pypirisk.validate")


class TooManyInvalidRowsError(RuntimeError):
    """Raised when the invalid-row share exceeds ``max_invalid_fraction``."""


def _annotate(check: Callable[[pd.DataFrame], pd.Series]):
    def annotate(batch: pd.DataFrame) -> pd.DataFrame:
        return batch.assign(**{checks.REASON_COLUMN: check(batch)})

    return annotate


def _valid_rows(batch: pd.DataFrame) -> pd.DataFrame:
    return batch[batch[checks.REASON_COLUMN] == ""].drop(columns=checks.REASON_COLUMN)


def validate_table(
    source: Path, target: Path, check: Callable[[pd.DataFrame], pd.Series]
) -> dict[str, Any]:
    """Validate one Parquet table with Ray Data and write the valid rows.

    Args:
        source: Input Parquet file.
        target: Output directory for the valid rows.
        check: Returns the invalid reason per row of a batch.

    Returns:
        Row counts and rejection reasons.
    """
    annotated = (
        ray.data.read_parquet(str(source))
        .map_batches(_annotate(check), batch_format="pandas")
        .materialize()
    )
    reasons = Counter()
    for batch in annotated.select_columns([checks.REASON_COLUMN]).iter_batches(
        batch_format="pandas"
    ):
        reasons.update(batch[checks.REASON_COLUMN].tolist())
    total = sum(reasons.values())
    rejected = total - reasons.pop("", 0)
    shutil.rmtree(target, ignore_errors=True)
    if total:
        annotated.map_batches(_valid_rows, batch_format="pandas").write_parquet(str(target))
    else:
        target.mkdir(parents=True, exist_ok=True)
        pd.read_parquet(source).to_parquet(target / "empty.parquet", index=False)
    return {"rows_in": total, "rows_rejected": rejected, "reasons": dict(reasons)}


def validate(params: dict[str, Any]) -> None:
    """Validate every ingest table, write the valid rows, a report and metrics.

    Args:
        params: The full parameter dictionary.

    Raises:
        TooManyInvalidRowsError: If the invalid share exceeds the configured maximum.
    """
    stage = params["validate"]
    source, target = Path(stage["input_dir"]), Path(stage["output_dir"])
    target.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)
    started = time.monotonic()

    packages = pd.read_parquet(source / "packages.parquet")
    package_reasons = checks.check_packages(packages)
    valid_packages = packages[package_reasons == ""]
    valid_packages.to_parquet(target / "packages.parquet", index=False)
    names = set(valid_packages["name"])
    report: dict[str, Any] = {
        "packages": {
            "rows_in": len(packages),
            "rows_rejected": int((package_reasons != "").sum()),
            "reasons": dict(Counter(package_reasons[package_reasons != ""])),
        }
    }

    table_checks = {
        "releases": lambda batch: checks.check_releases(batch, names, now),
        "requires": lambda batch: checks.check_requires(batch, names),
        "commits": lambda batch: checks.check_commits(batch, now),
        "issues": lambda batch: checks.check_issues(batch, now),
    }
    with local_ray(params["crawl"]["ray_num_cpus"], params["crawl"]["ray_object_store_mb"]):
        for table, check in table_checks.items():
            report[table] = validate_table(source / f"{table}.parquet", target / table, check)

    seconds = time.monotonic() - started
    rows_in = sum(entry["rows_in"] for entry in report.values())
    rows_rejected = sum(entry["rows_rejected"] for entry in report.values())
    summary = {
        "rows_in": rows_in,
        "rows_rejected": rows_rejected,
        "invalid_fraction": rows_rejected / rows_in if rows_in else 0.0,
        "records_per_second": round(rows_in / max(seconds, 1e-9), 1),
    }
    write_json(Path(stage["report_path"]), {"summary": summary, "tables": report})
    if summary["invalid_fraction"] > stage["max_invalid_fraction"]:
        raise TooManyInvalidRowsError(
            f"{summary['invalid_fraction']:.1%} of rows are invalid "
            f"(maximum {stage['max_invalid_fraction']:.0%})"
        )
    logger.info("Validated tables", extra=summary)


if __name__ == "__main__":
    run_stage("validate", validate)
