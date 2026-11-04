"""FastAPI backend for the four frontend pages.

Start it with ``uvicorn --factory pypirisk_api.main:create_app_from_env``. Every endpoint
and its request and response schema is listed in ``docs/LLD.md``.
"""

import logging
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Query
from prometheus_client import Counter, Histogram, Summary

from pypirisk.grades import grade
from pypirisk.logs import configure_logging
from pypirisk.monitoring.http import instrument_app
from pypirisk.propagate.project import parse_requirements, project_risk, resolve
from pypirisk_api import pipeline, schemas
from pypirisk_api.external import (
    RegistryClient,
    UpstreamError,
    heuristic_fallback,
)
from pypirisk_api.model_client import ModelServerClient, ModelServerError
from pypirisk_api.settings import Settings
from pypirisk_api.store import ServingStore

logger = logging.getLogger("pypirisk_api")

SERVICE_NAME = "api"

RISK_SCORES = Histogram(
    "api_risk_score",
    "Risk scores returned to clients.",
    ["source"],
    buckets=(0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0),
)
SCAN_DEPENDENCIES = Summary(
    "api_scan_dependencies", "Dependencies reached per project scan (declared + transitive)."
)
UNSCORED_LOOKUPS = Counter("api_unscored_lookups_total", "Lookups of packages without a score.")


@dataclass
class Services:
    """The API's collaborators, injected so tests can replace them.

    Attributes:
        store: Batch-scoring output.
        primary: MLflow model server client.
        secondary: Ray Serve client, if configured.
        registry: MLflow registry client.
    """

    store: ServingStore
    primary: ModelServerClient
    secondary: ModelServerClient | None
    registry: RegistryClient


def create_app(settings: Settings, services: Services) -> FastAPI:  # noqa: C901 - route table
    """Build the API with its collaborators injected.

    Args:
        settings: Service configuration.
        services: Store and clients.

    Returns:
        The configured FastAPI app.
    """
    app = FastAPI(title="pypirisk API", version="1.0.0")
    instrument_app(app, SERVICE_NAME)
    store = services.store

    def loaded_store() -> ServingStore:
        store.refresh()
        if not store.loaded:
            raise HTTPException(status_code=503, detail="No batch scores yet; run the pipeline")
        return store

    def model_version() -> str | None:
        return str(store.model.get("version")) if store.model else None

    @app.get("/health", response_model=schemas.HealthResponse)
    def health() -> schemas.HealthResponse:
        """Report liveness and whether model servers and batch scores are available."""
        store.refresh()
        primary_up = services.primary.is_healthy()
        secondary = "not configured"
        if services.secondary is not None:
            secondary = "up" if services.secondary.is_healthy() else "down"
        return schemas.HealthResponse(
            status="ok" if primary_up and store.loaded else "degraded",
            model_server="up" if primary_up else "down",
            secondary_model_server=secondary,
            packages_scored=len(store.scores),
            model_version=model_version(),
        )

    def risk_response(name: str, live: bool, backend: str) -> schemas.RiskResponse:
        data = loaded_store()
        row = data.score(name)
        if row is None:
            UNSCORED_LOOKUPS.inc()
            raise HTTPException(
                status_code=404, detail=f"{name!r} is not in the scored package set"
            )
        risk, factors, source = row["risk"], row["top_factors"], "batch"
        if live:
            client = services.secondary if backend == "ray" else services.primary
            if client is None:
                raise HTTPException(status_code=400, detail="Ray Serve backend is not configured")
            try:
                risk, factors = client.predict(data.model_inputs(row["name"]))
            except ModelServerError as error:
                logger.error("Live prediction failed", extra={"package": name, "error": str(error)})
                raise HTTPException(status_code=502, detail="Model server unavailable") from error
            source = "live"
        RISK_SCORES.labels(source).observe(risk)
        return schemas.RiskResponse(
            package=row["name"],
            risk=risk,
            grade=grade(risk, settings.grade_bands),
            snapshot=row["snapshot"],
            top_factors=factors,
            source=source,
            backend=backend if live else "batch",
            model_version=model_version(),
        )

    @app.get("/packages/{name}/risk", response_model=schemas.RiskResponse)
    def package_risk(
        name: str,
        live: bool = Query(False, description="Score now through a model server"),
        backend: str = Query("mlflow", pattern="^(mlflow|ray)$"),
    ) -> schemas.RiskResponse:
        """Return one package's abandonment risk and its top reasons.

        Raises:
            HTTPException: 404 if not scored, 502 if the live model server fails.
        """
        return risk_response(name, live, backend)

    @app.get("/packages/{name}", response_model=schemas.PackageDetail)
    def package_detail(name: str) -> schemas.PackageDetail:
        """Return a package's risk with its context and graph neighbours.

        Raises:
            HTTPException: 404 if the package is not scored.
        """
        base = risk_response(name, live=False, backend="batch")
        row = store.score(name)
        dependencies, dependents = store.neighbours(row["name"])

        def number(key: str) -> float | None:
            value = row.get(key)
            return None if value is None or value != value else float(value)

        return schemas.PackageDetail(
            **base.model_dump(),
            repo_url=row.get("repo_url") if isinstance(row.get("repo_url"), str) else None,
            months_since_last_release=number("months_since_last_release"),
            releases_in_lookback=number("releases_in_lookback"),
            commits_in_lookback=number("commits_in_lookback"),
            bus_factor=number("bus_factor"),
            dependents=number("dependents"),
            pagerank=number("pagerank"),
            dependencies=dependencies,
            dependents_in_universe=dependents,
        )

    @app.post("/projects/scan", response_model=schemas.ScanResponse)
    def scan_project(request: schemas.ScanRequest) -> schemas.ScanResponse:
        """Score a project from its requirements through the dependency graph.

        Raises:
            HTTPException: 422 if the file is too long or names no package.
        """
        data = loaded_store()
        try:
            parsed = parse_requirements(request.requirements)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        if not parsed.packages:
            raise HTTPException(status_code=422, detail="No package names found")
        dependencies = resolve(parsed.packages, data.edges, settings.max_depth)
        result = project_risk(dependencies, data.risks(), settings.depth_decay)
        SCAN_DEPENDENCIES.observe(len(dependencies))
        names = {dependency.name for dependency in dependencies}
        edges = data.edges[data.edges["source"].isin(names) & data.edges["target"].isin(names)]
        return schemas.ScanResponse(
            grade=grade(result.score, settings.grade_bands),
            score=result.score,
            expected_abandoned=result.expected_abandoned,
            declared=parsed.packages,
            dependencies_total=len(dependencies),
            contributions=[
                {**item, "grade": grade(item["risk"], settings.grade_bands)}
                for item in result.contributions
            ],
            unscored=[
                {
                    "package": name,
                    "months_since_last_release": heuristic_fallback(
                        settings.pypi_url, name, settings.timeout_seconds
                    ),
                }
                for name in result.unscored[:20]
            ],
            parse_errors=parsed.errors,
            edges=list(edges[["source", "target"]].itertuples(index=False, name=None)),
        )

    @app.get("/ecosystem/watch", response_model=list[schemas.WatchEntry])
    def ecosystem_watch(
        limit: int = Query(50, ge=1, le=500), min_dependents: int = Query(0, ge=0)
    ) -> list[schemas.WatchEntry]:
        """List widely used packages that most need support.

        Priority is risk times the package's PageRank percentile, so a risky package
        that much of the ecosystem depends on ranks above an equally risky leaf.
        """
        scores = loaded_store().scores
        scores = scores[scores["dependents"] >= min_dependents]
        priority = scores["risk"] * scores["pagerank"].rank(pct=True)
        top = scores.assign(priority=priority).sort_values("priority", ascending=False).head(limit)
        return [
            schemas.WatchEntry(
                package=name,
                risk=row["risk"],
                grade=row["grade"],
                dependents=row["dependents"],
                pagerank=row["pagerank"],
                priority=row["priority"],
                top_factors=store.score(name)["top_factors"],
            )
            for name, row in top.iterrows()
        ]

    @app.get("/pipeline/dag")
    def pipeline_dag() -> dict:
        """Return the DVC stages, their dependencies and each stage's last run."""
        dag = pipeline.read_dag(settings.dvc_yaml_path)
        latest = pipeline.last_run_per_stage(settings.pipeline_runs_path)
        for stage in dag["stages"]:
            stage["last_run"] = latest.get(stage["name"])
        return dag

    @app.get("/pipeline/runs")
    def pipeline_runs(
        limit: int = Query(50, ge=1, le=500),
        status: str | None = Query(None, pattern="^(success|failed)$"),
    ) -> list[dict]:
        """Return recent stage runs, newest first, with errors for failed runs."""
        return pipeline.read_runs(settings.pipeline_runs_path, limit, status)

    @app.get("/pipeline/reports")
    def pipeline_reports() -> dict:
        """Return the ingest, validation, evaluation and registry reports."""
        return pipeline.read_reports(settings.reports_dir)

    @app.get("/models")
    def models() -> dict:
        """Return the registered model versions and the batch-scoring model.

        Raises:
            HTTPException: 502 if MLflow cannot be reached.
        """
        try:
            versions = services.registry.versions(settings.model_name)
        except UpstreamError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        store.refresh()
        return {
            "name": settings.model_name,
            "alias": settings.model_alias,
            "versions": versions,
            "batch": store.model,
        }

    return app


def create_app_from_env() -> FastAPI:
    """Build the API from environment variables; used as the uvicorn factory.

    Returns:
        The configured FastAPI app.
    """
    configure_logging()
    settings = Settings.from_env()
    timeout = settings.timeout_seconds
    secondary = None
    if settings.secondary_model_server_url:
        secondary = ModelServerClient(settings.secondary_model_server_url, timeout, backend="ray")
    services = Services(
        store=ServingStore(settings.serving_dir),
        primary=ModelServerClient(settings.model_server_url, timeout, backend="mlflow"),
        secondary=secondary,
        registry=RegistryClient(settings.mlflow_url, timeout),
    )
    return create_app(settings, services)
