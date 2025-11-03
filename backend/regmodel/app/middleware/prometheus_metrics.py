"""Prometheus metrics helpers and middleware for FastAPI app."""

from __future__ import annotations

import time
from typing import Optional

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
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
PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Predictions generated",
    labelnames=("model_type",),
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    labelnames=("model_type",),
)
TRAINING_RUNS = Counter(
    "training_runs_total",
    "Training runs executed",
    labelnames=("status", "model_type"),
)
TRAINING_DURATION = Histogram(
    "training_duration_seconds",
    "Training duration in seconds",
    labelnames=("model_type",),
)
DRIFT_CHECKS = Counter(
    "evidently_drift_checks_total",
    "Total Evidently drift checks executed",
)
DRIFT_DETECTED = Gauge(
    "drift_detected",
    "Latest drift detection flag (0/1)",
)
DRIFT_SHARE = Gauge(
    "drift_share",
    "Latest drift share (0.0-1.0)",
)
DRIFTED_FEATURES_COUNT = Gauge(
    "drifted_features_count",
    "Number of drifted features in last run",
)
MODEL_R2 = Gauge(
    "model_r2_score",
    "Latest model R² score",
    labelnames=("model_type",),
)
MODEL_RMSE = Gauge(
    "model_rmse",
    "Latest model RMSE",
    labelnames=("model_type",),
)


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
    """Return a Response suitable for `/metrics` endpoint."""

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def observe_prediction(
    model_type: str, latency_seconds: float, predictions_count: int
) -> None:
    """Record prediction metrics for a model type."""

    PREDICTIONS_TOTAL.labels(model_type=model_type).inc(predictions_count)
    PREDICTION_LATENCY.labels(model_type=model_type).observe(max(latency_seconds, 0.0))


def record_training(duration_seconds: float, model_type: str, status: str) -> None:
    """Record training metrics for successes or failures."""

    TRAINING_DURATION.labels(model_type=model_type).observe(max(duration_seconds, 0.0))
    TRAINING_RUNS.labels(status=status, model_type=model_type).inc()


def record_drift_check() -> None:
    """Increment total drift checks."""

    DRIFT_CHECKS.inc()


def update_drift_metrics(
    drift_detected: bool,
    drift_share: Optional[float],
    drifted_features: Optional[int],
) -> None:
    """Update gauges related to data drift."""

    DRIFT_DETECTED.set(1 if drift_detected else 0)
    if drift_share is not None:
        DRIFT_SHARE.set(max(min(drift_share, 1.0), 0.0))
    if drifted_features is not None:
        DRIFTED_FEATURES_COUNT.set(max(drifted_features, 0))


def update_model_metrics(
    model_type: str, r2: Optional[float], rmse: Optional[float]
) -> None:
    """Update gauges for model evaluation metrics."""

    if r2 is not None:
        MODEL_R2.labels(model_type=model_type).set(r2)
    if rmse is not None:
        MODEL_RMSE.labels(model_type=model_type).set(rmse)
