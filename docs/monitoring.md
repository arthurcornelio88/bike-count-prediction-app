# MLOps Monitoring Stack

**Status**: ✅ Phase 3 Complete
**Branch**: `feat/mlops-monitoring`
**Date**: 2025-11-03

---

## Quick Start

```bash
# Start monitoring stack
docker compose --profile monitoring up -d

# Check all services are UP
docker compose --profile monitoring ps

# Verify Prometheus targets
curl http://localhost:9090/api/v1/targets
```

**Access**:

- Prometheus: <http://localhost:9090>
- Grafana: <http://localhost:3000> (admin/admin)
- Airflow: <http://localhost:8081> (admin/admin)
- Airflow Exporter: <http://localhost:9101/metrics>

---

## Architecture

```text
┌─────────────────┐
│   Prometheus    │  ← Scrapes metrics every 10-30s
│   (port 9090)   │  ← 15 days retention
└────────┬────────┘
         │
    ┌────┴────────────────┐
    │                     │
┌───▼────────┐   ┌────────▼──────────┐
│  FastAPI   │   │ Airflow Exporter  │
│ :8000      │   │ :9101             │
│ /metrics   │   │ → Airflow API     │
└────────────┘   └───────────────────┘
```

---

## Metrics Collected

### Airflow DAGs (via custom exporter)

#### DAG 1: daily_fetch_bike_data

- `bike_records_ingested_total`: Records ingested (after dedup)

#### DAG 2: daily_prediction

- `bike_predictions_generated_total`: Predictions count
- `bike_prediction_r2`: R² score
- `bike_prediction_rmse`: RMSE
- `bike_prediction_mae`: MAE

#### DAG 3: monitor_and_fine_tune

- `bike_drift_detected`: Drift flag (0/1)
- `bike_drift_share`: Drift percentage (0.0-1.0)
- `bike_drifted_features_count`: Drifted features count
- `bike_model_r2_production`: Production R²
- `bike_model_rmse_production`: Production RMSE
- `bike_model_improvement_delta`: R² improvement
- `bike_training_runs_total{status}`: Training runs
- `bike_model_deployments_total{decision}`: Deployment decisions

### FastAPI (if instrumented)

- `fastapi_requests_total`: HTTP requests
- `fastapi_request_duration_seconds`: Request latency
- `fastapi_errors_total`: 5xx errors
- `predictions_total`: Predictions count
- `training_runs_total`: Training runs
- `model_r2_score`, `model_rmse`: Model metrics

---

## Configuration

### Prometheus

**File**: `monitoring/prometheus.yml`

```yaml
scrape_configs:
  - job_name: 'regmodel-api'
    static_configs:
      - targets: ['regmodel-backend:8000']
    scrape_interval: 10s

  - job_name: 'airflow-metrics'
    static_configs:
      - targets: ['airflow-exporter:9101']
    scrape_interval: 30s
```

### Airflow Exporter

**File**: `monitoring/custom_exporters/airflow_exporter.py`

- Collection interval: 60s
- Lookback period: 7 days
- Monitored DAGs: `daily_fetch_bike_data`, `daily_prediction`, `monitor_and_fine_tune`

**Environment** (docker-compose.yaml):

```yaml
environment:
  - AIRFLOW_BASE_URL=http://airflow-webserver:8080
  - AIRFLOW_USERNAME=admin
  - AIRFLOW_PASSWORD=admin
  - PYTHONUNBUFFERED=1
```

---

## Troubleshooting

### Airflow exporter DOWN

```bash
# Check exporter logs
docker logs airflow-exporter

# Test endpoint
curl http://localhost:9101/health

# Restart
docker compose --profile monitoring restart airflow-exporter
```

### No DAGs visible in Airflow UI

**Cause**: WSL2 bind mount bug

**Fix**:

```bash
docker compose restart airflow-webserver airflow-scheduler airflow-worker
```

See [troubleshooting_wsl2_docker.md](./troubleshooting_wsl2_docker.md) for details.

### Prometheus not scraping

```bash
# Reload config
docker compose --profile monitoring restart prometheus

# Check targets
curl http://localhost:9090/api/v1/targets | jq
```

### No metrics data

**Cause**: No recent DAG runs

**Fix**:

```bash
# Trigger a DAG manually
docker exec airflow-webserver airflow dags trigger monitor_and_fine_tune
```

---

## Files Created

```text
monitoring/
├── custom_exporters/
│   ├── airflow_exporter.py       # Exporter (Flask app)
│   ├── requirements.txt           # Dependencies
│   └── Dockerfile                 # Container
├── prometheus.yml                 # Scrape config
├── alerts.yml                     # Alert rules
└── grafana/provisioning/          # Grafana auto-config

docs/
├── monitoring.md                  # This file
├── prometheus_queries.md          # PromQL queries
├── troubleshooting_wsl2_docker.md # WSL2 fixes
└── MONITORING_SUMMARY.md          # Executive summary
```

---

## Useful Commands

```bash
# Start/stop
docker compose --profile monitoring up -d
docker compose --profile monitoring down

# Logs
docker logs -f prometheus
docker logs -f airflow-exporter
docker logs -f grafana

# Restart specific service
docker compose --profile monitoring restart prometheus
docker compose --profile monitoring restart airflow-exporter

# Check metrics
curl http://localhost:9101/metrics | grep bike_
curl http://localhost:8000/metrics | grep fastapi_
```

---

## Next Steps

- [ ] **Phase 4**: Grafana Dashboards
- [ ] **Phase 5**: Alerting (Discord webhook)

---

**Last update**: 2025-11-03
