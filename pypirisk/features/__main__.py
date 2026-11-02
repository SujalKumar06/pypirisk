"""Features stage: ``python -m pypirisk.features``."""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from pypirisk.features.build import build_feature_table
from pypirisk.io import read_table
from pypirisk.labels.windows import every_snapshot
from pypirisk.stagelog import run_stage

logger = logging.getLogger("pypirisk.features")

TABLES = ("packages", "releases", "commits", "issues")


def features(params: dict[str, Any]) -> None:
    """Write features for every labelled snapshot and the scoring snapshot.

    Args:
        params: The full parameter dictionary.
    """
    stage = params["features"]
    source, graph_dir = Path(stage["input_dir"]), Path(stage["graph_dir"])
    tables = {name: read_table(source, name) for name in TABLES}
    table = build_feature_table(
        tables,
        pd.read_parquet(graph_dir / "nodes.parquet"),
        pd.read_parquet(graph_dir / "edges.parquet"),
        every_snapshot(params["snapshots"]),
        stage["lookback_months"],
    )
    output = Path(stage["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output, index=False)
    logger.info("Built features", extra={"rows": len(table), "columns": len(table.columns)})


if __name__ == "__main__":
    run_stage("features", features)
