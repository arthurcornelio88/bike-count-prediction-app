# Grafana Dashboards - MLOps Monitoring

**Access**: <http://localhost:3000> (admin/{GF_SECURITY_ADMIN_PASSWORD} in `.env` or `Google Secrets`})

---

## Available Dashboards

### 1. System Overview

**UID**: `mlops-overview`
**Refresh**: 10s
**Time range**: Last 6 hours

**Panels**:

- Drift Detected (gauge) - Shows if drift is currently detected (0/1)
- Drift Share (gauge) - Percentage of drifted features (0-100%)
- Production R² (gauge) - Current model R² score
- Services Status (timeseries) - UP/DOWN status of all services
- API Request Rate (timeseries) - Requests per second by endpoint
- API Error Rate (timeseries) - Error percentage over time
- Records Ingested (stat) - Total records processed
- Predictions Generated (stat) - Total predictions made
- Drifted Features (stat) - Number of features with drift
- Production RMSE (stat) - Current RMSE value

**Use case**: Quick health check of the entire system

![Grafana System Overview](/docs/img/grafana_system.png)

---

### 2. Training & Deployment

**UID**: `mlops-training-deployment`
**Refresh**: 1m
**Time range**: Last 7 days

**Panels**:

- Training Runs (6h windows) - Success/failure counts
- Deployment Decisions (6h windows) - Deploy/skip/reject counts
- Model Improvement Over Time - R² delta trend
- Latest Model Improvement (stat) - Most recent improvement value
- Total Training Runs (stat) - Breakdown by status
- Total Deployments (stat) - Breakdown by decision
- Training Success Rate (gauge) - Percentage of successful runs
- Champion vs Challenger R² - Compare models
- Champion vs Challenger RMSE - Compare models

**Use case**: Monitor training pipeline and deployment decisions

![Grafana Training](/docs/img/grafana_training.png)

---

### 3. Drift Monitoring

**UID**: `mlops-drift-monitoring`
**Refresh**: 30s
**Time range**: Last 7 days

**Panels**:

- Data Drift Over Time - Drift percentage evolution
- Drift Status (gauge) - Current drift state (YES/NO)
- Drifted Features Count - Number of features with drift
- Current Drift Share (gauge) - Current drift percentage
- R² vs Drift Correlation - See how drift affects model performance
- Summary Statistics - Key metrics at a glance

**Use case**: Track data drift and its impact on model quality

![Grafana Drift](/docs/img/grafana_drift.png)

---

### 4. Model Performance

**UID**: `mlops-model-performance`
**Refresh**: 30s
**Time range**: Last 24 hours

**Panels**:

- R² Score Trend - Compare production vs prediction R²
- RMSE Trend - Compare production vs prediction RMSE
- MAE Trend - Prediction MAE over time
- Model Improvement Delta - R² improvement from training
- API Latency Percentiles - P50/P95/P99 response times
- Data Processing Rate - Records ingested and predictions per hour

**Use case**: Monitor model accuracy and API performance

![Grafana Model](/docs/img/.png)

---

## Quick Start

### Access Grafana

```bash
# Open in browser
http://localhost:3000

# Login
Username: admin
Password: admin
```

### Navigate to Dashboards

1. Click on "Dashboards" (left menu)
2. Select "MLOps Monitoring" folder
3. Choose a dashboard

### Alternative: Direct URLs

- Overview: <http://localhost:3000/d/mlops-overview>
- Performance: <http://localhost:3000/d/mlops-model-performance>
- Drift: <http://localhost:3000/d/mlops-drift-monitoring>
- Training: <http://localhost:3000/d/mlops-training-deployment>

---

## Customization

### Time Range

- Click the time picker (top right)
- Select preset (Last 6h, 24h, 7d, 30d)
- Or set custom range

### Refresh Rate

- Click refresh dropdown (top right)
- Choose: 5s, 10s, 30s, 1m, 5m, 15m, 30m, 1h

---

## Troubleshooting

### No Data in Dashboards

**Cause**: Metrics not being collected

**Fix**:

```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Ensure all targets are UP
# - airflow-metrics
# - regmodel-api
# - prometheus

# Trigger DAG manually to generate metrics
docker exec airflow-webserver airflow dags trigger monitor_and_fine_tune
```

### Dashboard Not Loading

**Cause**: Grafana provisioning issue

**Fix**:

```bash
# Restart Grafana
docker compose --profile monitoring restart grafana

# Wait 10 seconds, then refresh browser
```

### "Panel plugin not found"

**Cause**: Grafana version mismatch

**Fix**:

- These dashboards are designed for Grafana 10.x
- Update Grafana if using older version
- Or edit panels to use basic visualizations

---

## Dashboard Files

Location: `monitoring/grafana/provisioning/dashboards/`

```text
├── dashboards.yml                 # Auto-import configuration
├── overview.json                  # System Overview
├── model_performance.json         # Model Performance
├── drift_monitoring.json          # Drift Monitoring
└── training_deployment.json       # Training & Deployment
```

**Note**: Dashboards are auto-imported on Grafana startup. Changes require Grafana restart.

---

## Useful PromQL Queries

See [prometheus_queries.md](./prometheus_queries.md) for all available queries.

**Quick examples**:

```promql
# Drift share percentage
bike_drift_share * 100

# Production R²
bike_model_r2_production

# API request rate
rate(fastapi_requests_total[5m])

# Error rate
100 * sum(rate(fastapi_errors_total[5m])) / sum(rate(fastapi_requests_total[5m]))
```

---

**Last update**: 2025-11-04
