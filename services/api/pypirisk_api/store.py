"""In-memory view of the batch-scoring output, reloaded when a new batch lands."""

import json
import logging
import threading
from pathlib import Path

import pandas as pd
from packaging.utils import canonicalize_name
from prometheus_client import Gauge

from pypirisk.monitoring.drift import feature_drift

logger = logging.getLogger(__name__)

FEATURE_DRIFT = Gauge(
    "pypirisk_feature_drift",
    "How far each serving feature's mean moved from the reference, in standard deviations.",
    ["feature"],
)
SCORED_PACKAGES = Gauge("api_scored_packages", "Packages with a batch risk score.")
MODEL_VERSION = Gauge("api_batch_model_version", "Model version that produced the batch scores.")

FILES = ("scores.parquet", "features.parquet", "edges.parquet", "model.json", "reference.json")


class ServingStore:
    """Batch scores, model inputs and the scoring graph, keyed by package.

    The pipeline's score stage rewrites ``data/serving``; the store notices the newer
    files on the next request and reloads, so the API never needs a restart. Loading
    also recomputes feature drift for Prometheus.
    """

    def __init__(self, directory: Path) -> None:
        """Create a store; data is loaded on first use.

        Args:
            directory: The batch-scoring output directory.
        """
        self.directory = Path(directory)
        self._lock = threading.Lock()
        self._stamp: tuple[float, ...] | None = None
        self.scores = pd.DataFrame()
        self.features = pd.DataFrame()
        self.edges = pd.DataFrame(columns=["source", "target"])
        self.model: dict = {}

    def _current_stamp(self) -> tuple[float, ...] | None:
        paths = [self.directory / name for name in FILES]
        if not all(path.exists() for path in paths):
            return None
        return tuple(path.stat().st_mtime for path in paths)

    def refresh(self) -> None:
        """Reload the files if they changed since the last load."""
        stamp = self._current_stamp()
        if stamp is None or stamp == self._stamp:
            return
        with self._lock:
            if stamp == self._stamp:
                return
            self.scores = pd.read_parquet(self.directory / "scores.parquet").set_index("name")
            self.features = pd.read_parquet(self.directory / "features.parquet").set_index("name")
            self.edges = pd.read_parquet(self.directory / "edges.parquet")
            self.model = json.loads((self.directory / "model.json").read_text())
            reference = json.loads((self.directory / "reference.json").read_text())
            self._export_drift(reference)
            SCORED_PACKAGES.set(len(self.scores))
            MODEL_VERSION.set(float(self.model.get("version", 0)))
            self._stamp = stamp
            logger.info("Loaded batch scores", extra={"packages": len(self.scores), **self.model})

    def _export_drift(self, reference: dict) -> None:
        inputs = self.features.drop(columns=["snapshot"], errors="ignore")
        for feature, value in feature_drift(reference, inputs).items():
            FEATURE_DRIFT.labels(feature).set(value)

    @property
    def loaded(self) -> bool:
        """Whether batch scores are available."""
        return not self.scores.empty

    def score(self, package: str) -> dict | None:
        """Return the batch score of a package.

        Args:
            package: Name in any PyPI spelling.

        Returns:
            The row as a dict (``risk``, ``grade``, ``top_factors``, context), or ``None``.
        """
        name = canonicalize_name(package)
        if name not in self.scores.index:
            return None
        row = self.scores.loc[name].to_dict()
        row["top_factors"] = json.loads(row["top_factors"])
        return {"name": name, **row}

    def model_inputs(self, package: str) -> dict[str, float | None] | None:
        """Return the model inputs of a package, for live scoring.

        Args:
            package: Name in any PyPI spelling.

        Returns:
            Feature values (NaN as ``None``), or ``None`` if unknown.
        """
        name = canonicalize_name(package)
        if name not in self.features.index:
            return None
        row = self.features.loc[name].drop(labels=["snapshot"], errors="ignore")
        return {key: (None if pd.isna(value) else float(value)) for key, value in row.items()}

    def risks(self) -> dict[str, float]:
        """Return every package's batch risk."""
        return self.scores["risk"].to_dict() if self.loaded else {}

    def neighbours(self, package: str) -> tuple[list[str], list[str]]:
        """Return a package's direct dependencies and direct dependents.

        Args:
            package: Canonical name.

        Returns:
            Sorted dependencies and dependents in the scoring snapshot's graph.
        """
        dependencies = sorted(self.edges.loc[self.edges["source"] == package, "target"])
        dependents = sorted(self.edges.loc[self.edges["target"] == package, "source"])
        return dependencies, dependents
