# Metrics Alignment Documentation

**Date:** 2025-11-04
**Status:** ✅ Fully Aligned

## Architecture Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                    METRICS SOURCES                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1️⃣ Airflow Exporter (PRIMARY SOURCE)                       │
│     - Scrapes Airflow API + BigQuery audit logs            │
│     - Exposes business metrics (drift, model perf, etc)    │
│     - Port: 9101                                            │
│     - File: monitoring/custom_exporters/airflow_exporter.py │
│                                                             │
│  2️⃣ FastAPI App (SECONDARY SOURCE)                          │
│     - Exposes HTTP-level metrics only                      │
│     - Request count, latency, errors                       │
│     - Port: 8000/metrics                                    │
│     - File: backend/regmodel/app/middleware/               │
│             prometheus_metrics.py                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   PROMETHEUS SCRAPER                        │
│     - Scrapes both endpoints every 15s/30s                 │
│     - Stores time-series data                              │
│     - Config: monitoring/prometheus.yml                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    CONSUMERS                                │
├─────────────────────────────────────────────────────────────┤
│  📊 Grafana Dashboards (4 files)                            │
│  🚨 Grafana Alert Rules (9 rules)                           │
└─────────────────────────────────────────────────────────────┘
```

## Metrics Inventory

### 🔵 Airflow Exporter Metrics (Source of Truth)

| Metric Name | Type | Labels | Description | Used In |
|-------------|------|--------|-------------|---------|
| `bike_model_r2_champion_current` | Gauge | - | Champion R² on test_current | Dashboards ✅, Alerts ✅ |
| `bike_model_r2_champion_baseline` | Gauge | - | Champion R² on test_baseline | Dashboards ✅ |
| `bike_model_r2_challenger_current` | Gauge | - | Challenger R² on test_current | Dashboards ✅ |
| `bike_model_r2_challenger_baseline` | Gauge | - | Challenger R² on test_baseline | Dashboards ✅ |
| `bike_model_rmse_production` | Gauge | - | Champion RMSE (weekly validation) | Dashboards ✅, Alerts ✅ |
| `bike_drift_detected` | Gauge | - | Drift flag (0/1) | Dashboards ✅ |
| `bike_drift_share` | Gauge | - | Drift share (0.0-1.0) | Dashboards ✅, Alerts ✅ |
| `bike_drifted_features_count` | Gauge | - | Number of drifted features | Dashboards ✅ |
| `bike_model_improvement_delta` | Gauge | - | R² improvement (new - old) | Dashboards ✅ |
| `bike_model_deployments_total` | Counter | `decision` | Deployment decisions (deploy/skip/reject) | Dashboards ✅ |
| `bike_training_runs_total` | Counter | `status` | Training runs (success/failed) | Dashboards ✅ |
| `bike_records_ingested_total` | Gauge | - | Records ingested (DAG 1) | Dashboards ✅ |
| `bike_predictions_generated_total` | Gauge | - | Predictions generated (DAG 2) | Dashboards ✅ |
| `bike_prediction_rmse` | Gauge | - | Latest prediction RMSE | Dashboards ✅ |
| `bike_prediction_mae` | Gauge | - | Latest prediction MAE | Dashboards ✅ |
| `bike_prediction_r2` | Gauge | - | Latest prediction R² | Dashboards ✅ |
| `airflow_dag_run_duration_seconds` | Histogram | `dag_id`, `state` | DAG run duration | Dashboards ✅ |
| `airflow_task_duration_seconds` | Histogram | `dag_id`, `task_id`, `state` | Task duration | Dashboards ✅ |
| `airflow_dag_runs_total` | Counter | `dag_id`, `state` | Total DAG runs | Dashboards ✅ |

### 🟢 FastAPI Metrics (HTTP Only)

| Metric Name | Type | Labels | Description | Used In |
|-------------|------|--------|-------------|---------|
| `fastapi_requests_total` | Counter | `method`, `endpoint`, `status_code` | Total HTTP requests | Dashboards ✅, Alerts ✅ |
| `fastapi_request_duration_seconds` | Histogram | `method`, `endpoint` | Request latency | Dashboards ✅ |
| `fastapi_errors_total` | Counter | `method`, `endpoint` | HTTP 5xx errors | Dashboards ✅, Alerts ✅ |

## Removed Metrics (2025-11-04)

The following metrics were **defined but never exposed/instrumented** in FastAPI.
They have been removed to eliminate confusion and redundancy:

| Removed Metric | Reason | Replacement |
|----------------|--------|-------------|
| `predictions_total` | Never instrumented | Use `bike_predictions_generated_total` |
| `prediction_latency_seconds` | Never instrumented | Use `fastapi_request_duration_seconds` |
| `training_runs_total` | Never instrumented | Use `bike_training_runs_total` |
| `training_duration_seconds` | Never instrumented | Use `airflow_task_duration_seconds` |
| `model_r2_score` | Never instrumented | Use `bike_model_r2_champion_current` |
| `bike_model_r2_production` | Removed (double-eval redesign) | Use `bike_model_r2_champion_current` |
| `model_rmse` | Never instrumented | Use `bike_model_rmse_production` |
| `drift_detected` | Never instrumented | Use `bike_drift_detected` |
| `drift_share` | Never instrumented | Use `bike_drift_share` |
| `drifted_features_count` | Never instrumented | Use `bike_drifted_features_count` |

**Impact:** 4 alert rules removed (see Alert Rules section below) - metrics either non-existent or redundant.

## Grafana Alert Rules

**Total Rules:** 9 (down from 13)
**File:** `monitoring/grafana/provisioning/alerting/rules.yml`

### Active Alert Rules

| Rule ID | Metric(s) Used | Status |
|---------|----------------|--------|
| `model_performance_critical` | `bike_model_r2_champion_current` | ✅ Active |
| `model_performance_warning` | `bike_model_r2_champion_current` | ✅ Active |
| `model_rmse_high` | `bike_model_rmse_production` | ✅ Active (threshold 70) |
| `high_drift_detected` | `bike_drift_share` | ✅ Active |
| `critical_drift_with_performance` | `bike_drift_share`, `bike_model_r2_champion_current` | ✅ Active |
| `service_down` | `up` | ✅ Active |
| `api_error_rate_high` | `fastapi_errors_total`, `fastapi_requests_total` | ✅ Active |
| `airflow_dag_too_slow` | `airflow_dag_run_duration_seconds` | ✅ Active |
| `no_data_ingested` | `bike_records_ingested_total` | ✅ Active |

### Removed Alert Rules (2025-11-04)

| Rule ID | Reason |
|---------|--------|
| `prediction_latency_high` | Used non-existent `prediction_latency_seconds` metric |
| `training_failure` | Used non-existent `training_runs_total` metric |
| `airflow_dag_failed` | Metric `airflow_task_failures_total` not exposed by exporter |
| `low_data_ingestion` | Redundant with `no_data_ingested` (same metric) |

## Grafana Dashboards

**Total Dashboards:** 4
**Location:** `monitoring/grafana/provisioning/dashboards/`

| Dashboard | Metrics Used | Status |
|-----------|--------------|--------|
| `overview.json` | All `bike_*`, `fastapi_*`, `airflow_*` | ✅ Aligned |
| `model_performance.json` | `bike_model_r2_champion_*`, `bike_model_r2_challenger_*`, `bike_model_rmse_production` | ✅ Aligned |
| `drift_monitoring.json` | `bike_drift_*` | ✅ Aligned |
| `training_deployment.json` | `bike_training_runs_total`, `bike_model_deployments_total` | ✅ Aligned |

**Verification:** No dashboard uses removed FastAPI metrics ✅

## Data Flow

### Business Metrics (Model Performance, Drift)

```text
DAG: monitor_and_fine_tune
  ↓ (writes audit record)
BigQuery: monitoring_audit.logs
  ↓ (queried every 60s)
Airflow Exporter: /metrics
  ↓ (scraped every 30s)
Prometheus
  ↓ (queried)
Grafana Dashboards + Alerts
```

### HTTP Metrics (API Health)

```text
FastAPI Request
  ↓ (PrometheusMiddleware)
Prometheus Metrics (in-memory)
  ↓ (exposed at /metrics)
Prometheus
  ↓ (queried)
Grafana Dashboards + Alerts
```

## Best Practices

1. **Single Source of Truth:** Airflow Exporter is the canonical source for business metrics
2. **No Duplication:** Never define the same metric in multiple places
3. **BigQuery as Audit Log:** All critical metrics are persisted in BigQuery
4. **FastAPI = HTTP Only:** FastAPI only exposes HTTP-level metrics (request count, latency, errors)
5. **Naming Convention:** Business metrics use `bike_*` prefix, HTTP metrics use `fastapi_*` prefix

## Testing Checklist

- [x] Prometheus scraping both endpoints successfully
- [x] Airflow exporter showing correct R² (0.8673) from BigQuery
- [x] Grafana alert rules provisioned (11 rules)
- [x] Discord webhook receiving test alerts
- [ ] All dashboards display data correctly (Phase 3)
- [ ] False-positive alerts resolved (Phase 1)

## Maintenance

**When adding new metrics:**

1. Decide: Business metric (Airflow) or HTTP metric (FastAPI)?
2. Add metric definition to appropriate file
3. Instrument the code to populate the metric
4. Update this documentation
5. Add dashboard panels if needed
6. Add alert rules if needed
7. Verify metric appears in Prometheus

**When removing metrics:**

1. Check usage in dashboards (`grep -r "metric_name" dashboards/`)
2. Check usage in alerts (`grep -r "metric_name" alerting/`)
3. Remove metric definition
4. Update this documentation
5. Restart affected services

## Troubleshooting

**Metric not appearing in Prometheus:**

- Check exporter logs: `docker logs airflow-exporter`
- Verify Prometheus scrape config: `monitoring/prometheus.yml`
- Test endpoint directly: `curl http://localhost:9101/metrics | grep metric_name`

**Dashboard showing "No data":**

- Verify metric exists: Prometheus → Status → Targets
- Check query syntax in dashboard panel
- Verify time range (some metrics are counters, need rate())

**Alert not firing:**

- Check alert rule status in Grafana → Alerting → Alert Rules
- Verify metric threshold is correct
- Check evaluation interval (some alerts have `for: 5m` delay)

---

**Last Updated:** 2025-11-04
**Next Review:** After Phase 3 (dashboard audit)
