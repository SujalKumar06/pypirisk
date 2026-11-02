"""Label stage: ``python -m pypirisk.labels``."""

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pypirisk.io import read_table
from pypirisk.labels.build import LabelSpec, build_label_table
from pypirisk.labels.windows import check_matured, check_split_windows
from pypirisk.stagelog import run_stage

logger = logging.getLogger("pypirisk.labels")


def labels(params: dict[str, Any]) -> None:
    """Check the split windows, then write labels for every eligible package.

    Args:
        params: The full parameter dictionary.

    Raises:
        ValueError: If the splits overlap or a snapshot has not matured yet.
    """
    stage, splits = params["labels"], params["snapshots"]
    check_split_windows(splits, stage["outcome_months"])
    check_matured(splits, stage["outcome_months"], datetime.now(UTC))
    spec = LabelSpec(
        lookback_months=stage["lookback_months"],
        outcome_months=stage["outcome_months"],
        max_commits_in_outcome=stage["max_commits_in_outcome"],
        min_releases_for_eligibility=stage["min_releases_for_eligibility"],
    )
    source = Path(stage["input_dir"])
    tables = {name: read_table(source, name) for name in ("packages", "releases", "commits")}
    table = build_label_table(tables, splits, spec)
    output = Path(stage["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output, index=False)
    prevalence = table.groupby("split")["label"].mean().round(3).to_dict()
    logger.info("Built labels", extra={"rows": len(table), "prevalence": prevalence})


if __name__ == "__main__":
    run_stage("label", labels)
