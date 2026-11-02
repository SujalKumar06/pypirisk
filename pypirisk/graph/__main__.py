"""Graph stage: ``python -m pypirisk.graph``."""

import logging
from pathlib import Path
from typing import Any

from pypirisk.graph.build import build_graphs
from pypirisk.io import read_table
from pypirisk.labels.windows import every_snapshot
from pypirisk.stagelog import run_stage

logger = logging.getLogger("pypirisk.graph")


def graph(params: dict[str, Any]) -> None:
    """Write per-snapshot edges and node metrics to ``data/graph``.

    Args:
        params: The full parameter dictionary.
    """
    stage = params["graph"]
    source, target = Path(stage["input_dir"]), Path(stage["output_dir"])
    edges, nodes = build_graphs(
        read_table(source, "releases"),
        read_table(source, "requires"),
        every_snapshot(params["snapshots"]),
    )
    target.mkdir(parents=True, exist_ok=True)
    edges.to_parquet(target / "edges.parquet", index=False)
    nodes.to_parquet(target / "nodes.parquet", index=False)
    per_snapshot = edges.groupby("snapshot").size().to_dict()
    logger.info("Built graphs", extra={"edges_per_snapshot": per_snapshot})


if __name__ == "__main__":
    run_stage("graph", graph)
