# Prometheus Queries - Bike Traffic MLOps

## Quick Access

- **Prometheus UI**: <http://localhost:9090>
- **Graph**: <http://localhost:9090/graph>
- **Targets**: <http://localhost:9090/targets>

All targets should be **UP** ✅

---

## 🚀 Top Queries (Copy & Paste)

### Drift Detection

```promql
# Drift detected? (1=yes, 0=no)
bike_drift_detected

# Drift percentage (0-100%)
bike_drift_share * 100

# Number of drifted features
bike_drifted_features_count
```

![Drift detected query](/docs/img/prometheus_drift_detected.png)

### Model Performance

```promql
# Production R²
bike_model_r2_production

# Production RMSE
bike_model_rmse_production

# Model improvement (delta R²)
bike_model_improvement_delta
```

![R² production](/docs/img/prometheus_r2_production.png)
*R² = 0.1872 (improvement between new model and legacy performance)*

### Predictions

```promql
# Predictions count
bike_predictions_generated_total

# Predictions R²
bike_prediction_r2

# Predictions RMSE
bike_prediction_rmse

# Predictions MAE
bike_prediction_mae
```

![Predictions generated](/docs/img/prometheus_predictions_total.png)
*192 predictions generated*

### Ingestion

```promql
# Records ingested
bike_records_ingested_total
```

---

## 📊 FastAPI Metrics (if instrumented)

### Latency

```promql
# P95 latency
histogram_quantile(0.95, rate(fastapi_request_duration_seconds_bucket[5m]))

# P50 (median)
histogram_quantile(0.50, rate(fastapi_request_duration_seconds_bucket[5m]))
```

![FastAPI latency P95](/docs/img/prometheus_fastapi_p95.png)
*Shows P95 latency for /metrics endpoint (~4.9ms)*

### Requests

```promql
# Total requests
fastapi_requests_total

# Request rate (requests/sec)
rate(fastapi_requests_total[5m])

# Requests by endpoint
sum by (endpoint) (fastapi_requests_total)
```

### Errors

```promql
# Error rate (%)
100 * sum(rate(fastapi_errors_total[5m])) / sum(rate(fastapi_requests_total[5m]))
```

---

## 🔍 Diagnostic Queries

### Check targets status

```promql
# All targets (1=UP, 0=DOWN)
up

# Airflow exporter status
up{job="airflow-metrics"}

# FastAPI status
up{job="regmodel-api"}
```

### List metrics

```promql
# All bike_* metrics
{__name__=~"bike_.*"}

# All fastapi_* metrics
{__name__=~"fastapi_.*"}
```

---

## 📈 Advanced Queries

### Comparisons

```promql
# R² delta (predictions vs production)
bike_prediction_r2 - bike_model_r2_production

# Improvement percentage
(bike_model_improvement_delta / bike_model_r2_production) * 100
```

### Time series

```promql
# Drift over 24h
bike_drift_share[24h]

# R² over 7 days
bike_model_r2_production[7d]
```

### Aggregations

```promql
# Total training runs
sum(bike_training_runs_total)

# Training success rate (%)
100 * sum(bike_training_runs_total{status="success"}) / sum(bike_training_runs_total)

# Average R² over 1h
avg_over_time(bike_model_r2_production[1h])
```

---

## ⚠️ Alert Examples

```promql
# R² too low (< 0.65)
bike_model_r2_production < 0.65

# High drift (> 50%)
bike_drift_share > 0.5

# RMSE too high (> 60)
bike_model_rmse_production > 60

# API error rate > 5%
100 * sum(rate(fastapi_errors_total[5m])) / sum(rate(fastapi_requests_total[5m])) > 5
```

---

## 💡 Tips

### Visualization

- **Table view**: Exact values
- **Graph view**: Time series evolution
- **Time range**: 1h, 6h, 24h, 7d (top right)
- **Auto-refresh**: 5s, 15s, 30s, 1m

### Functions

- `rate(metric[5m])`: Rate per second over 5 minutes
- `increase(metric[1h])`: Total increase over 1 hour
- `avg_over_time(metric[1h])`: Average over 1 hour
- `histogram_quantile(0.95, ...)`: 95th percentile

---

**Last update**: 2025-11-03
