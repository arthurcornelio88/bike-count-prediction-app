# MLOps Monitoring Stack - Architecture

**Status**: ✅ Production Ready
**Last Updated**: 2025-11-04
**Version**: V1.0 (9 alert rules, aligned metrics)

---

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Metrics Sources](#metrics-sources)
- [Configuration](#configuration)
- [Screenshots](#screenshots)
- [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Start Monitoring Stack

```bash
# Start all services including monitoring
docker compose --profile monitoring up -d

# Verify all services are UP
docker compose --profile monitoring ps

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, health: .health}'
```

### Access Points

| Service | URL | Credentials | Purpose |
|---------|-----|-------------|---------|
| **Prometheus** | <http://localhost:9090> | None | Metrics storage & queries |
| **Grafana** | <http://localhost:3000> | admin / `$GF_SECURITY_ADMIN_PASSWORD` | Dashboards & alerts |
| **Airflow** | <http://localhost:8081> | admin / admin | DAG orchestration |
| **Airflow Exporter** | <http://localhost:9101/metrics> | None | Custom metrics endpoint |
| **RegModel API** | <http://localhost:8000/metrics> | None | FastAPI metrics |

---

## 🏗️ Architecture Overview

### Component Diagram

```text
┌─────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────┐        ┌──────────────────────────┐  │
│  │  BigQuery Audit      │        │  Airflow API             │  │
│  │  (Source of Truth)   │        │  (DAG Runs, Tasks)       │  │
│  └──────────┬───────────┘        └──────────┬───────────────┘  │
│             │                               │                   │
│             └───────────────┬───────────────┘                   │
│                             │                                   │
│                    ┌────────▼─────────┐                         │
│                    │ Airflow Exporter │                         │
│                    │ :9101/metrics    │                         │
│                    │ (Flask + BQ)     │                         │
│                    └────────┬─────────┘                         │
└─────────────────────────────┼─────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
┌────────▼─────────┐  ┌───────▼────────┐  ┌───────▼────────┐
│  Prometheus      │  │  FastAPI       │  │  Prometheus    │
│  Scraper         │  │  :8000/metrics │  │  Self-Monitor  │
│  (15s interval)  │  │  (HTTP only)   │  │  :9090         │
└────────┬─────────┘  └────────────────┘  └────────────────┘
         │
         │ Stores time-series data (15 days retention)
         │
┌────────▼──────────────────────────────────────────────────┐
│                    PROMETHEUS TSDB                        │
│  • bike_model_r2_production                               │
│  • bike_model_rmse_production                             │
│  • bike_drift_share                                       │
│  • fastapi_requests_total                                 │
│  • ... (19 total metrics)                                 │
└────────┬──────────────────────────────────────────────────┘
         │
         │ Queries metrics every 10-30s
         │
┌────────▼──────────────────────────────────────────────────┐
│                      GRAFANA                              │
│  ┌────────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Dashboards    │  │  Alerting    │  │  Discord     │  │
│  │  (4 boards)    │  │  (9 rules)   │  │  Webhook     │  │
│  └────────────────┘  └──────────────┘  └──────────────┘  │
└───────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Single Source of Truth**: BigQuery `monitoring_audit.logs` table stores all critical metrics
2. **Dual Collection**: Airflow Exporter (batch) + FastAPI (real-time HTTP)
3. **Grafana Unified Alerting**: No separate Alertmanager needed
4. **Non-redundant Metrics**: Each metric exposed once, no duplicates
5. **Notification-Only**: Alerts inform humans, no auto-remediation

---

## 📊 Metrics Sources

### 1. Airflow Exporter (Primary Source)

**Purpose**: Expose business metrics from MLOps pipeline
**Source**: BigQuery audit logs (primary) + Airflow XCom (fallback)
**Port**: 9101
**Scrape Interval**: 30s
**Collection Interval**: 60s (internal cache)

**Metrics Exposed** (16 total):

| Category | Metric | Type | Description |
|----------|--------|------|-------------|
| **Model Performance** | `bike_model_r2_production` | Gauge | Production R² score (0-1) |
| | `bike_model_rmse_production` | Gauge | Production RMSE |
| **Data Drift** | `bike_drift_detected` | Gauge | Drift flag (0/1) |
| | `bike_drift_share` | Gauge | Drift share (0.0-1.0) |
| | `bike_drifted_features_count` | Gauge | Count of drifted features |
| **Training** | `bike_training_runs_total` | Counter | Training runs by status |
| | `bike_model_improvement_delta` | Gauge | R² improvement (new - old) |
| | `bike_model_deployments_total` | Counter | Deployments by decision |
| **Ingestion** | `bike_records_ingested_total` | Gauge | Records ingested (DAG 1) |
| **Predictions** | `bike_predictions_generated_total` | Gauge | Predictions count (DAG 2) |
| | `bike_prediction_r2` | Gauge | Prediction R² |
| | `bike_prediction_rmse` | Gauge | Prediction RMSE |
| | `bike_prediction_mae` | Gauge | Prediction MAE |
| **Airflow** | `airflow_dag_run_duration_seconds` | Histogram | DAG execution time |
| | `airflow_task_duration_seconds` | Histogram | Task execution time |
| | `airflow_dag_runs_total` | Counter | Total DAG runs by state |

**Implementation**: `monitoring/custom_exporters/airflow_exporter.py`

---

### 2. FastAPI Metrics (HTTP Monitoring Only)

**Purpose**: Monitor API health (requests, latency, errors)
**Port**: 8000
**Scrape Interval**: 10s

**Metrics Exposed** (3 total):

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `fastapi_requests_total` | Counter | method, endpoint, status_code | Total HTTP requests |
| `fastapi_request_duration_seconds` | Histogram | method, endpoint | Request latency |
| `fastapi_errors_total` | Counter | method, endpoint | HTTP 5xx errors |

**Implementation**: `backend/regmodel/app/middleware/prometheus_metrics.py`

**Note**: Business metrics (predictions, training, drift) were **removed** from FastAPI.
Airflow Exporter is the single source for these.

---

## ⚙️ Configuration

### Prometheus

**File**: `monitoring/prometheus.yml`

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  # RegModel FastAPI - HTTP metrics only
  - job_name: 'regmodel-api'
    static_configs:
      - targets: ['regmodel-backend:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  # Airflow Exporter - Business metrics
  - job_name: 'airflow-metrics'
    static_configs:
      - targets: ['airflow-exporter:9101']
    metrics_path: '/metrics'
    scrape_interval: 30s  # Exporter caches for 60s

  # Prometheus self-monitoring
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
```

**Storage**:

- Retention: 15 days
- Volume: `prometheus_data`
- Path: `/prometheus`

---

### Airflow Exporter

**File**: `monitoring/custom_exporters/airflow_exporter.py`

**Environment Variables** (docker-compose.yaml):

```yaml
environment:
  - AIRFLOW_BASE_URL=http://airflow-webserver:8080
  - AIRFLOW_USERNAME=${_AIRFLOW_WWW_USER_USERNAME:-admin}
  - AIRFLOW_PASSWORD=${_AIRFLOW_WWW_USER_PASSWORD:-admin}
  - GOOGLE_APPLICATION_CREDENTIALS=/app/gcp.json
  - GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT}
  - PYTHONUNBUFFERED=1
```

**Dependencies** (requirements.txt):

```txt
flask==3.0.0
prometheus-client==0.20.0
requests==2.31.0
python-dateutil==2.9.0
google-cloud-bigquery==3.25.0  # For audit log queries
```

**Data Flow**:

1. Every 60s: Query BigQuery `monitoring_audit.logs` for latest metrics
2. Fallback: If BigQuery unavailable, query Airflow XCom
3. Expose metrics at `/metrics` endpoint
4. Prometheus scrapes every 30s

---

### Grafana

**Auto-Provisioning** (on container start):

```text
monitoring/grafana/provisioning/
├── datasources/
│   └── prometheus.yml         # Add Prometheus datasource
├── dashboards/
│   ├── dashboards.yml         # Dashboard provider
│   ├── overview.json          # Main dashboard
│   ├── model_performance.json # Model metrics
│   ├── drift_monitoring.json  # Drift detection
│   └── training_deployment.json # Training pipeline
└── alerting/
    ├── contactpoints.yml      # Discord webhook
    ├── policies.yml           # Routing policies
    └── rules.yml              # 9 alert rules
```

**Volumes**:

- Config: `./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro`
- Data: `grafana_data:/var/lib/grafana`

---

## 📸 Screenshots

> **Note**: Add your Grafana/Discord screenshots to `docs/monitoring/screenshots/`

### Recommended Screenshots

1. **`grafana_overview_dashboard.png`**
   - Main dashboard with all key metrics
   - Show R² = 0.867, RMSE = 66.66

2. **`grafana_alert_rules.png`**
   - Alerting → Alert Rules
   - Show 9 rules, state (Normal/Pending/Firing)

3. **`discord_notifications.png`**
   - Discord channel with alert examples
   - Different severity levels (critical/warning/info)

4. **`prometheus_targets.png`**
   - Status → Targets
   - All 3 jobs UP (regmodel-api, airflow-metrics, prometheus)

---

## 🔧 Troubleshooting

### No metrics in Grafana

**Symptoms**: Dashboards show "No data"

**Checks**:

```bash
# 1. Verify Prometheus targets UP
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, health: .health}'

# Expected:
# {"job":"regmodel-api","health":"up"}
# {"job":"airflow-metrics","health":"up"}
# {"job":"prometheus","health":"up"}

# 2. Check metrics exist in Prometheus
curl "http://localhost:9090/api/v1/query?query=bike_model_r2_production" | jq '.data.result[0].value'

# 3. Check exporter is exposing metrics
curl http://localhost:9101/metrics | grep bike_model_r2_production
```

**Fixes**:

```bash
# Restart Prometheus
docker compose --profile monitoring restart prometheus

# Restart exporter (if BigQuery issue)
docker compose --profile monitoring restart airflow-exporter

# Check exporter logs
docker logs airflow-exporter | tail -50
```

---

### Airflow Exporter DOWN

**Symptoms**: Target `airflow-metrics` shows DOWN in Prometheus

**Checks**:

```bash
# 1. Check exporter health
curl http://localhost:9101/health

# 2. Check container status
docker compose --profile monitoring ps airflow-exporter

# 3. Check logs for errors
docker logs airflow-exporter --tail 100
```

**Common Issues**:

| Error | Cause | Fix |
|-------|-------|-----|
| `BigQuery client initialization failed` | Missing/invalid `gcp.json` | Verify `./gcp.json` exists and has BigQuery read permissions |
| `Airflow API connection refused` | Airflow webserver not ready | Wait 30s, restart exporter |
| `401 Unauthorized` | Wrong Airflow credentials | Check `AIRFLOW_USERNAME`/`PASSWORD` in `.env` |

---

### Grafana alerts not firing

**Symptoms**: Metrics show alert condition met, but no notification

**Checks**:

```bash
# 1. Verify alert rules loaded
curl -s "http://localhost:3000/api/v1/provisioning/alert-rules" \
  -u "admin:$GF_SECURITY_ADMIN_PASSWORD" | jq 'length'

# Expected: 9

# 2. Check alert evaluation
curl -s "http://localhost:3000/api/alertmanager/grafana/api/v2/alerts" \
  -u "admin:$GF_SECURITY_ADMIN_PASSWORD" | jq 'length'

# 3. Test contact point
# Grafana UI → Alerting → Contact points → discord-mlops → Test
```

**Fixes**:

```bash
# Restart Grafana to reload provisioning
docker compose --profile monitoring restart grafana

# Check Discord webhook valid
curl -X POST "$DISCORD_WEBHOOK_URL" \
  -H "Content-Type: application/json" \
  -d '{"content": "Test from Grafana"}'
```

---

### WSL2-specific issues

**Issue**: DAGs not visible in Airflow UI after host restart

**Cause**: WSL2 bind mount bug (files show empty)

**Fix**:

```bash
# Restart Airflow services
docker compose restart airflow-webserver airflow-scheduler airflow-worker

# Wait 30s, refresh UI
```

See [troubleshooting_wsl2_docker.md](../troubleshooting_wsl2_docker.md) for details.

---

## 📚 Related Documentation

- [02_alerting.md](./02_alerting.md) - Alert rules & Discord setup
- [03_metrics_reference.md](./03_metrics_reference.md) - Complete metrics inventory
- [../phase4_monitoring_implementation.md](../phase4_monitoring_implementation.md) - Implementation log

---

## 🔄 Maintenance

### Regular Tasks

```bash
# Weekly: Check Prometheus disk usage
docker exec prometheus du -sh /prometheus

# Monthly: Review alert firing frequency
# Grafana → Alerting → Alert Rules → Sort by "Last Evaluation"

# Quarterly: Update retention policy if needed
# Edit monitoring/prometheus.yml → --storage.tsdb.retention.time
```

---

**Last updated**: 2025-11-04
**Status**: ✅ Production Ready (9 alert rules, 19 metrics aligned)
