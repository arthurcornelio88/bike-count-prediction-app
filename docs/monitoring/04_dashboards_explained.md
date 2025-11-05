docs/monitoring/04_dashboards_explained.md

# Grafana Dashboards — Updated Guide

**Date**: 2024-11-04
**Subject**: Dashboard documentation after double-evaluation redesign

---

## Quick overview

- `MLOps - Overview`: service health, API traffic and drift indicators.
- `MLOps - Model Performance`: champion/challenger comparison on *test_current*
  and *test_baseline*, associated errors and runtime capacities.
- `MLOps - Training & Deployment`: verdicts from the `dag_monitor_and_train`
  DAG, tracking trainings and deployment decisions.

### Double-evaluation in two lines

| Test set | Champion | Challenger | Purpose |
|----------|----------|------------|---------|
| `test_current` | `bike_model_r2_champion_current` | `bike_model_r2_challenger_current` | compare models on recent data |
| `test_baseline` | `bike_model_r2_champion_baseline` | `bike_model_r2_challenger_baseline` | detect regression against the reference set |

> BigQuery keeps the four columns for audit; Prometheus exposes both pairs via the Airflow exporter.

---

## Dashboard `MLOps - Overview`

| Panel | Purpose | Query / metric | Comfort zone |
|-------|---------|----------------|--------------|
| Drift Detected | Binary indicator | `max_over_time(bike_drift_detected[15m])` | 0 = no active drift |
| Drift Share (%) | Ratio of drifting features | `bike_drift_share * 100` | <30 % ok, >50 % critical |
| Champion R² (test_current) | Health of deployed model | `bike_model_r2_champion_current` | >0.70 ideal, <0.65 critical alert |
| Services Status | Prometheus/FastAPI/exporter availability | `up{job="..."}` | 1 for each service |
| API Request Rate | Load per endpoint | `rate(fastapi_requests_total[5m])` | ~0.1 req/s; zero → check API |
| API Error Rate | 5xx error % | `rate(fastapi_errors_total[5m])` | No data = no errors; >5 % critical |
| Records Ingested / Predictions | Hourly volumes | `increase(bike_records_ingested_total[1h])` | curves close; 0 → blocked |
| Production RMSE | Absolute error of champion | `bike_model_rmse_production` | sudden rise = drift or data incident |

---

## Dashboard `MLOps - Model Performance`

### Key panels

- **R² Trend - test_current**: Compares `bike_model_r2_champion_current` vs
  `bike_model_r2_challenger_current`. Challenger must beat champion for `deploy`.

- **R² Trend - test_baseline**: Tracks `bike_model_r2_champion_baseline` and
  `bike_model_r2_challenger_baseline`. If challenger <0.60, DAG forces `reject`.

- **RMSE Trend**: `bike_model_rmse_production` vs `bike_prediction_rmse` on
  *test_current*. Checks absolute error follows R² trend.

- **MAE Trend**
  `bike_prediction_mae`. Kept for historical comparison with business indicators.

- **Model Improvement Delta**
  `bike_model_improvement_delta` (R² gain). Green = improvement, red = regression. Value shown in legend.

- **API Latency Percentiles**: `histogram_quantile(...)` on
  `fastapi_request_duration_seconds`. P50 <50 ms; P99 >500 ms → investigate.

- **Data Processing Rate**: `increase(bike_predictions_generated_total[1h])`
  and `increase(bike_records_ingested_total[1h])`. Correlates volumes/performance.

### Quick read

1. Champion_current <0.70 → anticipate retraining.
2. Challenger_current > Champion_current +0.02 → likely deployment.
3. Challenger_baseline < Champion_baseline → regression; check dataset or features.
4. Exploding RMSE/MAE → suspect corrupted data or pipeline bug.

---

## Dashboard `MLOps - Training & Deployment`

| Panel | What it tells | Metrics |
|-------|---------------|---------|
| Latest Model Improvement | Last `bike_model_improvement_delta` captured by DAG | `bike_model_improvement_delta` |
| Total Training Runs (failed/success) | Global cumulative | `bike_training_runs_total{status="success"/"failed"}` |
| Total Deployments | `deploy` decisions over the time-picker window | `increase(bike_model_deployments_total{decision="deploy"}[window])` |
| Training Success Rate | Success / total ratio computed in Grafana | same `bike_training_runs_total` counters |
| Champion vs Challenger R² (test_current) | Reprise of key panel for decision | `bike_model_r2_champion_current`, `bike_model_r2_challenger_current` |
| Champion vs Challenger R² (test_baseline) | Checks no regression | `bike_model_r2_champion_baseline`, `bike_model_r2_challenger_baseline` |
| Champion vs Challenger RMSE | Absolute error comparison | `bike_model_rmse_production`, `bike_prediction_rmse` |

### Reading

- Training success <90 % → check Airflow logs (`task retrain_model`).
- Too many `reject` → challenger no longer meets thresholds; re-evaluate features or hyper-parameters.
- No `deploy` for a long time while drift rises → escalate.

---

## On-call checklist

1. All services `up` = 1.
2. `bike_model_r2_champion_current` ≥0.70 and stable.
3. `bike_drift_share` <30 % (otherwise monitor alert).
4. Latest trainings successful and at least one recent `deploy`/`skip` decision.
5. API latency P50 <50 ms and zero error rate.

---

## Screenshots to produce

- Export each panel via `Share > Download PNG` for documentation or post-mortem.
- Save under `docs/monitoring/screenshots/` with explicit names
  (`model_performance_r2_current.png`, `overview_latency.png`, etc.).

---

**Maintainers**: Arthur Cornélio and MLOps team.
**Revision**: completed after dashboard redesign (branch `feat/mlops-monitoring`).
