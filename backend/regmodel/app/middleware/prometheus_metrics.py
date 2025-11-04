"""Prometheus metrics helpers and middleware for FastAPI app."""

from __future__ import annotations

import time
from typing import Optional

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# HTTP-level instrumentation
REQUEST_COUNT = Counter(
    "fastapi_requests_total",
    "Total HTTP requests",
    labelnames=("method", "endpoint", "status_code"),
)
REQUEST_LATENCY = Histogram(
    "fastapi_request_duration_seconds",
    "Request latency in seconds",
    labelnames=("method", "endpoint"),
)
REQUEST_ERRORS = Counter(
    "fastapi_errors_total",
    "HTTP 5xx responses",
    labelnames=("method", "endpoint"),
)

# Domain-specific metrics
# NOTE: Business metrics (drift, model performance, training) are exposed by
# airflow-exporter (source of truth from BigQuery audit logs).
# FastAPI only exposes HTTP-level metrics for API health monitoring.


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Collect baseline HTTP metrics for every FastAPI request."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        start_time = time.perf_counter()
        method = request.method
        endpoint = request.url.path
        response: Optional[Response] = None

        try:
            response = await call_next(request)
            return response
        finally:
            elapsed = time.perf_counter() - start_time
            status_code = response.status_code if response else 500

            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code,
            ).inc()
            REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(elapsed)
            if status_code >= 500:
                REQUEST_ERRORS.labels(method=method, endpoint=endpoint).inc()


def prometheus_response() -> Response:
    """Return a Response suitable for `/metrics` endpoint.

    Exposes only HTTP-level metrics (requests, latency, errors).
    Business metrics are exposed by airflow-exporter.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
