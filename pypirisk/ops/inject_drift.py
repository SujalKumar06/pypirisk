"""Drift demo: distort the serving features so the drift alert fires.

``python -m pypirisk.ops.inject_drift`` rewrites ``data/serving/features.parquet`` as if
every package had gone quiet: no commits or releases in the lookback and four times the
months since the last release. The API reloads the file, its drift gauges jump, and
Prometheus fires ``FeatureDriftDetected`` (shown in Alertmanager and Grafana). Re-score
with ``dvc repro --single-item --force score`` to restore the real features.
"""

import argparse
import logging
import os
from pathlib import Path

import pandas as pd

from pypirisk.config import DEFAULT_PARAMS_PATH, load_params
from pypirisk.logs import configure_logging

logger = logging.getLogger("pypirisk.ops.inject_drift")

QUIET_COLUMNS = ("commits_in_lookback", "releases_in_lookback")


def drifted(features: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the features shifted far from the training distribution.

    Args:
        features: Serving features.

    Returns:
        The distorted copy.
    """
    shifted = features.copy()
    for column in QUIET_COLUMNS:
        if column in shifted:
            shifted[column] = 0.0
    if "months_since_last_release" in shifted:
        shifted["months_since_last_release"] = shifted["months_since_last_release"] * 4
    return shifted


def main() -> None:
    """Overwrite the serving features with a drifted copy and touch ``model.json``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    args = parser.parse_args()
    configure_logging()
    serving = Path(load_params(args.params)["score"]["output_dir"])
    path = serving / "features.parquet"
    drifted(pd.read_parquet(path)).to_parquet(path, index=False)
    os.utime(serving / "model.json")
    logger.warning("Injected drift into serving features", extra={"path": str(path)})


if __name__ == "__main__":
    main()
