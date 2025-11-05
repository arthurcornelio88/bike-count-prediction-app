# Grafana Dashboards & Prometheus Queries

**Access**: <http://localhost:3000> (admin / see `.env` for password)

## Data Sources

All dashboards use template variable `$instance` to switch between:

| Source | Value | Purpose |
|--------|-------|---------|
| **Production** | `airflow-exporter:9101` | Real metrics from DAGs via airflow-exporter |
| **Test (Pushgateway)** | `pushgateway-test:9091` | Mock metrics injected via test_grafana_alerts.py |

Switch data source via dropdown at top of each dashboard.

## Testing Alerts & Dashboards

![Testing alerts](/docs/img/test_alerting_terminal.png)

Use `scripts/test_grafana_alerts_and_dashboard.py` to inject mock metrics via Pushgateway:

```bash
# Inject all test scenarios
python scripts/test_grafana_alerts_and_dashboard.py

# Inject specific test (e.g., high drift)
python scripts/test_grafana_alerts_and_dashboard.py --test drift

# Restore normal (delete test metrics)
python scripts/test_grafana_alerts_and_dashboard.py --test restore
```

**Important**: Test script uses `job="test-metrics"` and `instance="pushgateway-test:9091"` to avoid
conflicting with production metrics from airflow-exporter.

See [TESTING_ALERTS.md](./TESTING_ALERTS.md) for full details.

---

## Dashboard 1: MLOps - Overview

**UID**: `mlops-overview` | **Refresh**: 10s | **Time**: Last 6h

**Panels**:

| Panel | Metric | Threshold | Purpose |
|-------|--------|-----------|---------|
| Drift Detected | `max_over_time(bike_drift_detected[15m])` | 0=ok, 1=drift | Binary drift indicator |
| Drift Share (%) | `bike_drift_share * 100` | <30% ok, >50% critical | % of drifted features |
| Champion R² (test_current) | `bike_model_r2_champion_current` | >0.70 ideal, <0.65 alert | Current model health |
| Services Status | `up{job="..."}` | 1=UP | Prometheus/API/exporter availability |
| API Request Rate | `rate(fastapi_requests_total[5m])` | ~0.1 req/s | Load per endpoint |
| API Error Rate | `rate(fastapi_errors_total[5m])` | >5% critical | 5xx error % |
| Records Ingested | `increase(bike_records_ingested_total[1h])` | 0=blocked | Hourly ingestion volume |
| Predictions Generated | `increase(bike_predictions_generated_total[1h])` | Should match ingestion | Hourly prediction volume |
| Production RMSE | `bike_model_rmse_production` | Sudden rise=drift | Absolute error of champion |

**Quick health check**: All services UP, R²≥0.70, drift<30%, API error rate=0.

---

## Dashboard 2: MLOps - Model Performance

**UID**: `mlops-model-performance` | **Refresh**: 30s | **Time**: Last 24h

**Panels**:

| Panel | Metrics | Purpose |
|-------|---------|---------|
| R² Trend - test_current | `bike_model_r2_champion_current`, `bike_model_r2_challenger_current` | Compare models on recent data; challenger must beat champion for deploy |
| R² Trend - test_baseline | `bike_model_r2_champion_baseline`, `bike_model_r2_challenger_baseline` | Detect regression against reference set; challenger<0.60 → reject |
| RMSE Trend | `bike_model_rmse_production`, `bike_prediction_rmse` | Champion (weekly validation) vs daily predictions RMSE |
| MAE Trend | `bike_prediction_mae` | Daily predictions MAE for historical comparison |
| Model Improvement Delta | `bike_model_improvement_delta` | R² gain (green=improvement, red=regression) |
| API Latency Percentiles | `histogram_quantile(..., fastapi_request_duration_seconds)` | P50/P95/P99 (P50<50ms, P99<500ms ideal) |
| Data Processing Rate | `increase(bike_records_ingested_total[1h])`, `increase(bike_predictions_generated_total[1h])` | Correlate volumes with performance |

**Double-evaluation summary**:

| Test set | Champion | Challenger | Purpose |
|----------|----------|------------|---------|
| test_current | `bike_model_r2_champion_current` | `bike_model_r2_challenger_current` | Compare on recent data |
| test_baseline | `bike_model_r2_champion_baseline` | `bike_model_r2_challenger_baseline` | Detect regression vs reference |

**Quick read**:

- Champion_current <0.70 → anticipate retraining
- Challenger_current > Champion_current +0.02 → likely deployment
- Challenger_baseline < Champion_baseline → regression; check features
- Exploding RMSE/MAE → suspect corrupted data

---

## Dashboard 3: MLOps - Drift Monitoring

**UID**: `mlops-drift-monitoring` | **Refresh**: 30s | **Time**: Last 7d

**Panels**:

| Panel | Metric | Threshold | Purpose |
|-------|--------|-----------|---------|
| Data Drift Over Time | `bike_drift_share * 100` | <30% ok, >50% critical | Drift % evolution |
| Drift Status (gauge) | `bike_drift_detected` | 0=NO, 1=YES | Current drift state |
| Drifted Features Count | `bike_drifted_features_count` | Monitor trend | Number of features with drift |
| Current Drift Share (gauge) | `bike_drift_share * 100` | 0-100% | Current drift percentage |
| R² vs Drift Correlation | `bike_model_r2_champion_current` vs `bike_drift_share` | See correlation | How drift affects model |

**Usage**: Track data drift and its impact on model quality. Drift >50% + R² declining → retrain urgently.

---

## Dashboard 4: MLOps - Training & Deployment

**UID**: `mlops-training-deployment` | **Refresh**: 1m | **Time**: Last 7d

**Panels**:

| Panel | Metric | Purpose |
|-------|--------|---------|
| Training Runs (6h windows) | `increase(bike_training_runs_total{status="success/failed"}[6h])` | Success/failure counts |
| Deployment Decisions (6h windows) | `increase(bike_model_deployments_total{decision="deploy/skip/reject"}[6h])` | Deploy/skip/reject counts |
| Model Improvement Over Time | `bike_model_improvement_delta` | R² delta trend |
| Latest Model Improvement (stat) | `bike_model_improvement_delta` | Most recent improvement value |
| Total Training Runs (stat) | `bike_training_runs_total` | Breakdown by status |
| Total Deployments (stat) | `bike_model_deployments_total` | Breakdown by decision |
| Training Success Rate (gauge) | `100 * sum(...{status="success"}) / sum(...)` | % of successful runs (>80% ideal) |
| Champion vs Challenger R² (test_current) | `bike_model_r2_champion_current`, `bike_model_r2_challenger_current` | Compare models |
| Champion vs Challenger R² (test_baseline) | `bike_model_r2_champion_baseline`, `bike_model_r2_challenger_baseline` | Check no regression |
| Champion vs Challenger RMSE | `bike_model_rmse_production`, `bike_prediction_rmse` | Champion (weekly validation) vs daily predictions |

**Quick read**:

- Training success <90% → check Airflow logs (`task retrain_model`)
- Too many `reject` → challenger no longer meets thresholds; re-evaluate features
- No `deploy` for long time + drift rising → escalate

---

## Key Prometheus Queries

### Drift Detection

```promql
# Drift detected? (1=yes, 0=no)
bike_drift_detected

# Drift percentage (0-100%)
bike_drift_share * 100

# Number of drifted features
bike_drifted_features_count
```

### Model Performance

```promql
# Champion R² (test_current)
bike_model_r2_champion_current

# Champion R² (test_baseline)
bike_model_r2_champion_baseline

# Challenger R² (test_current)
bike_model_r2_challenger_current

# Challenger R² (test_baseline)
bike_model_r2_challenger_baseline

# Champion RMSE (weekly validation from dag_monitor_and_train)
bike_model_rmse_production

# Model improvement (delta R²)
bike_model_improvement_delta
```

### Predictions

```promql
# Daily predictions RMSE (from dag_daily_prediction)
bike_prediction_rmse

# Daily predictions MAE
bike_prediction_mae

# Predictions count
bike_predictions_generated_total
```

### API Metrics

```promql
# P95 latency
histogram_quantile(0.95, rate(fastapi_request_duration_seconds_bucket[5m]))

# Request rate (requests/sec)
rate(fastapi_requests_total[5m])

# Error rate (%)
100 * sum(rate(fastapi_errors_total[5m])) / sum(rate(fastapi_requests_total[5m]))
```

### Service Health

```promql
# All targets (1=UP, 0=DOWN)
up

# Airflow exporter status
up{job="airflow-metrics"}

# FastAPI status
up{job="regmodel-api"}
```

### Training & Deployment

```promql
# Total training runs
sum(bike_training_runs_total)

# Training success rate (%)
100 * sum(bike_training_runs_total{status="success"}) / sum(bike_training_runs_total)

# Total deployments by decision
bike_model_deployments_total{decision="deploy"}
bike_model_deployments_total{decision="skip"}
bike_model_deployments_total{decision="reject"}
```

---

## Metric Sources

| Metric | Source DAG | Description |
|--------|------------|-------------|
| `bike_model_r2_champion_current` | `dag_monitor_and_train` | Champion evaluated on test_current (recent data) |
| `bike_model_r2_champion_baseline` | `dag_monitor_and_train` | Champion evaluated on test_baseline (reference set) |
| `bike_model_r2_challenger_current` | `dag_monitor_and_train` | Challenger evaluated on test_current |
| `bike_model_r2_challenger_baseline` | `dag_monitor_and_train` | Challenger evaluated on test_baseline |
| `bike_model_rmse_production` | `dag_monitor_and_train` | Champion RMSE on weekly validation |
| `bike_prediction_rmse` | `dag_daily_prediction` | Daily predictions RMSE |
| `bike_prediction_mae` | `dag_daily_prediction` | Daily predictions MAE |
| `bike_drift_detected` | `dag_monitor_and_train` | Binary drift indicator (0/1) |
| `bike_drift_share` | `dag_monitor_and_train` | Ratio of drifted features (0-1) |
| `bike_drifted_features_count` | `dag_monitor_and_train` | Count of drifted features |
| `bike_model_improvement_delta` | `dag_monitor_and_train` | R² gain from training |
| `bike_training_runs_total` | `dag_monitor_and_train` | Training runs counter |
| `bike_model_deployments_total` | `dag_monitor_and_train` | Deployment decisions counter |
| `bike_records_ingested_total` | `dag_daily_fetch_data` | Records ingested counter |
| `bike_predictions_generated_total` | `dag_daily_prediction` | Predictions generated counter |
| `fastapi_requests_total` | FastAPI middleware | HTTP requests counter |
| `fastapi_errors_total` | FastAPI middleware | HTTP 5xx errors counter |
| `fastapi_request_duration_seconds` | FastAPI middleware | Request latency histogram |

---

## On-Call Checklist

1. All services `up` = 1
2. `bike_model_r2_champion_current` ≥0.70 and stable
3. `bike_drift_share` <30%
4. Latest trainings successful and at least one recent `deploy`/`skip` decision
5. API latency P50 <50ms and zero error rate

---

**Maintainer**: Arthur Cornélio
**Last update**: 2025-11-05
