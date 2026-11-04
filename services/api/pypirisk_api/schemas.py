"""Request and response bodies of the API (documented in docs/LLD.md)."""

from typing import Literal

from pydantic import BaseModel, Field


class Factor(BaseModel):
    """One reason behind a risk score."""

    feature: str
    value: float | None
    contribution: float = Field(description="SHAP value in log-odds; positive raises risk")


class HealthResponse(BaseModel):
    """Body of ``GET /health``."""

    status: Literal["ok", "degraded"]
    model_server: Literal["up", "down"]
    secondary_model_server: Literal["up", "down", "not configured"]
    packages_scored: int
    model_version: str | None


class RiskResponse(BaseModel):
    """Body of ``GET /packages/{name}/risk``."""

    package: str
    risk: float
    grade: str
    snapshot: str
    top_factors: list[Factor]
    source: Literal["batch", "live"]
    backend: str
    model_version: str | None


class PackageDetail(RiskResponse):
    """Body of ``GET /packages/{name}``."""

    repo_url: str | None
    months_since_last_release: float | None
    releases_in_lookback: float | None
    commits_in_lookback: float | None
    bus_factor: float | None
    dependents: float | None
    pagerank: float | None
    dependencies: list[str]
    dependents_in_universe: list[str]


class ScanRequest(BaseModel):
    """Body of ``POST /projects/scan``."""

    requirements: str = Field(max_length=100_000, description="requirements.txt contents")


class ScanContribution(BaseModel):
    """One scored dependency in a project scan."""

    package: str
    depth: int
    path: list[str]
    risk: float
    grade: str
    weighted: float


class UnscoredPackage(BaseModel):
    """A dependency outside the scored universe, with the heuristic signal."""

    package: str
    months_since_last_release: float | None


class ScanResponse(BaseModel):
    """Body of ``POST /projects/scan``."""

    grade: str
    score: float
    expected_abandoned: float
    declared: list[str]
    dependencies_total: int
    contributions: list[ScanContribution]
    unscored: list[UnscoredPackage]
    parse_errors: list[str]
    edges: list[tuple[str, str]]


class WatchEntry(BaseModel):
    """One row of ``GET /ecosystem/watch``."""

    package: str
    risk: float
    grade: str
    dependents: float
    pagerank: float
    priority: float
    top_factors: list[Factor]
