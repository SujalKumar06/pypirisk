"""API configuration read from environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path


class MissingSettingError(RuntimeError):
    """Raised when a required environment variable is not set."""


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise MissingSettingError(f"Environment variable {name} must be set")
    return value


def _optional_url(name: str) -> str | None:
    value = os.environ.get(name)
    return value.rstrip("/") if value else None


@dataclass(frozen=True)
class Settings:
    """Runtime settings of the API service.

    Attributes:
        model_server_url: MLflow model server (primary serving path).
        secondary_model_server_url: Ray Serve deployment (second serving path), if any.
        serving_dir: Batch-scoring output (``scores.parquet``, ``edges.parquet``, ...).
        dvc_yaml_path: Pipeline definition, for the Pipeline Console's DAG.
        reports_dir: Stage reports (ingest, validation, metrics, registry).
        pipeline_runs_path: Run records written by every stage.
        mlflow_url: MLflow tracking server, for registry information.
        pypi_url: PyPI JSON API, for the heuristic fallback on unscored packages.
        timeout_seconds: Timeout for calls to other services.
        model_name: Registered model name.
        model_alias: Serving alias.
        depth_decay: Project-scan weight per dependency level.
        max_depth: Deepest dependency level a project scan follows.
        grades: Upper risk bound of each letter grade.
    """

    model_server_url: str
    serving_dir: Path
    secondary_model_server_url: str | None = None
    dvc_yaml_path: Path = Path("dvc.yaml")
    reports_dir: Path = Path("reports")
    pipeline_runs_path: Path = Path("logs/pipeline_runs.jsonl")
    mlflow_url: str | None = None
    pypi_url: str = "https://pypi.org/pypi"
    timeout_seconds: float = 5.0
    model_name: str = "pypirisk-abandonment"
    model_alias: str = "champion"
    depth_decay: float = 0.5
    max_depth: int = 6
    grades: tuple[tuple[str, float], ...] = (("A", 0.1), ("B", 0.2), ("C", 0.35), ("D", 0.5))

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables.

        ``MODEL_SERVER_URL`` and ``SERVING_DIR`` are required, so a misconfigured
        container fails at start-up instead of on the first request.

        Returns:
            The settings.

        Raises:
            MissingSettingError: If a required variable is missing.
        """
        env = os.environ.get
        return cls(
            model_server_url=_required("MODEL_SERVER_URL").rstrip("/"),
            serving_dir=Path(_required("SERVING_DIR")),
            secondary_model_server_url=_optional_url("SECONDARY_MODEL_SERVER_URL"),
            dvc_yaml_path=Path(env("DVC_YAML_PATH", "dvc.yaml")),
            reports_dir=Path(env("REPORTS_DIR", "reports")),
            pipeline_runs_path=Path(env("PIPELINE_RUNS_PATH", "logs/pipeline_runs.jsonl")),
            mlflow_url=_optional_url("MLFLOW_URL"),
            pypi_url=env("PYPI_URL", "https://pypi.org/pypi").rstrip("/"),
            timeout_seconds=float(env("TIMEOUT_SECONDS", "5")),
            model_name=env("MODEL_NAME", "pypirisk-abandonment"),
            model_alias=env("MODEL_ALIAS", "champion"),
        )

    @property
    def grade_bands(self) -> dict[str, float]:
        """Grade bands as an ordered mapping."""
        return dict(self.grades)
