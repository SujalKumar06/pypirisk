"""Request metrics and a ``/metrics`` endpoint for FastAPI services."""

import time

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

REQUESTS = Counter(
    "http_requests_total",
    "HTTP requests handled, by route template and status code.",
    ["service", "method", "route", "status"],
)
LATENCY = Histogram(
    "http_request_duration_seconds",
    "Time spent handling an HTTP request.",
    ["service", "method", "route"],
)
IN_PROGRESS = Gauge(
    "http_requests_in_progress",
    "HTTP requests currently being handled.",
    ["service"],
)

UNMATCHED_ROUTE = "unmatched"


def route_template(request: Request) -> str:
    """Return the matched route's path template, such as ``/packages/{name}/risk``.

    Labelling metrics with the template instead of the raw path keeps one time series
    per endpoint rather than one per package name.

    Args:
        request: The handled request.

    Returns:
        The route template, or ``"unmatched"`` for paths no route handled.
    """
    route = request.scope.get("route")
    return getattr(route, "path", UNMATCHED_ROUTE)


async def metrics_endpoint(_: Request) -> Response:
    """Serve every registered metric in the Prometheus text format.

    Returns:
        The exposition-format response.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def instrument_app(app: FastAPI, service: str) -> None:
    """Add request metrics middleware and a ``GET /metrics`` route to ``app``.

    Args:
        app: The FastAPI application to instrument.
        service: Value of the ``service`` label on every metric.
    """

    @app.middleware("http")
    async def record_request_metrics(request: Request, call_next):
        in_progress = IN_PROGRESS.labels(service)
        in_progress.inc()
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            in_progress.dec()
            route = route_template(request)
            LATENCY.labels(service, request.method, route).observe(time.perf_counter() - started)
            REQUESTS.labels(service, request.method, route, str(status)).inc()

    app.add_route("/metrics", metrics_endpoint, methods=["GET"], include_in_schema=False)
