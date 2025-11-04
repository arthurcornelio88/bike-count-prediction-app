# Discord Alerting Setup - MLOps Monitoring

**Status**: ✅ COMPLETE
**Last updated**: 2025-11-04

---

## Overview

The MLOps pipeline uses **two alerting channels** for different types of notifications:

1. **Prometheus → Grafana → Discord**: Infrastructure & system alerts (services DOWN, API errors, high latency)
2. **Airflow DAG → Discord**: Pipeline events (drift detected, training completed, deployment decisions)

**Alert Philosophy**: **Notification-only** - alerts inform humans for investigation, no automated remediation.

---

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    INFRASTRUCTURE ALERTS                    │
│  Prometheus → Grafana Contact Points → Discord Webhook     │
└─────────────────────────────────────────────────────────────┘
         ↓
    [Discord Channel]
         ↑
┌─────────────────────────────────────────────────────────────┐
│                      PIPELINE ALERTS                        │
│  Airflow DAG (monitor_and_fine_tune) → Discord Webhook     │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. Configuration

### Environment Variable

Add to [.env](.env) or `.env.airflow`:

```bash
# Discord webhook URL for MLOps alerts
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_WEBHOOK_ID/YOUR_WEBHOOK_TOKEN
```

**How to get webhook URL**:

1. Open Discord → Server Settings → Integrations → Webhooks
2. Create New Webhook
3. Name: "MLOps Bike Traffic Alerts"
4. Select channel (e.g., `#mlops-alerts`)
5. Copy Webhook URL
6. Paste into `.env`

### Services Using Discord Webhook

- **Grafana** (`docker-compose.yaml` line 264): Receives Prometheus alerts
- **Airflow** (`.env.airflow`): DAG tasks send pipeline notifications

---

## 2. Alert Types

### 2.1 Infrastructure Alerts (Prometheus → Grafana)

**Source**: [monitoring/alerts.yml](../monitoring/alerts.yml)
**Routing**: [monitoring/grafana/provisioning/alerting/policies.yml](../monitoring/grafana/provisioning/alerting/policies.yml)

| Alert | Severity | Trigger | Action |
|-------|----------|---------|--------|
| `ServiceDown` | CRITICAL | Service unreachable for 1 min | Check `docker compose ps`, restart service |
| `APIErrorRateHigh` | CRITICAL | Error rate > 5% for 5 min | Check FastAPI logs, investigate errors |
| `ModelPerformanceCritical` | CRITICAL | R² < 0.65 for 5 min | Immediate retraining required |
| `ModelRMSEHigh` | CRITICAL | RMSE > 60 for 5 min | Immediate retraining required |
| `CriticalDriftWithDecliningPerformance` | CRITICAL | Drift > 50% AND R² < 0.70 | Immediate retraining required |
| `ModelPerformanceWarning` | WARNING | R² < 0.70 for 10 min | Plan proactive retraining |
| `HighDriftDetected` | WARNING | Drift > 50% for 15 min | Monitor closely, prepare for retraining |
| `PredictionLatencyHigh` | WARNING | P95 latency > 5s for 10 min | Investigate API performance |
| `AirflowDAGFailed` | WARNING | Task failures in 30 min | Check Airflow UI for logs |
| `AirflowDAGTooSlow` | WARNING | DAG runtime > 1 hour | Check for stuck tasks |
| `NoDataIngested` | WARNING | No data in 2 hours | Check Paris Open Data API availability |
| `LowDataIngestion` | INFO | < 100 records in 1 hour | Verify API availability |

**Notification Frequency**:

- CRITICAL (infrastructure): Repeat every **1 hour**
- CRITICAL (model): Repeat every **2 hours**
- WARNING: Repeat every **4 hours**
- INFO: Repeat every **12 hours**

---

### 2.2 Pipeline Alerts (Airflow DAG)

**Source**: [dags/utils/discord_alerts.py](../dags/utils/discord_alerts.py)
**Integration**: [dags/dag_monitor_and_train.py](../dags/dag_monitor_and_train.py)

| Function | When Called | Severity | Color |
|----------|-------------|----------|-------|
| `send_drift_alert()` | Drift > 50% detected | WARNING | Orange |
| `send_performance_alert()` | R² < 0.70 (WARNING) or R² < 0.65 (CRITICAL) | WARNING/CRITICAL | Orange/Red |
| `send_training_success()` | Training completes successfully | INFO | Green/Gray/Orange |
| `send_training_failure()` | Training fails (timeout or error) | CRITICAL | Red |

**Training Success Outcomes**:

- 🚀 **Deploy** (Green): New model deployed to production
- ⏭️ **Skip** (Gray): Training skipped, no improvement
- ❌ **Reject** (Orange): New model rejected, worse than champion

---

## 3. Alert Logic in DAG

### In `end_monitoring()` Task

**Decision tree** (lines 272-322):

```python
# PRIORITY 1: Critical performance → retrain immediately
if r2 < 0.65 or rmse > 60:
    send_performance_alert(r2, rmse, threshold=0.70)
    return "fine_tune_model"

# PRIORITY 2: Critical drift + declining metrics → retrain immediately
if drift and drift_share >= 0.5 and r2 < 0.70:
    send_drift_alert(drift_share, r2, drifted_features)
    send_performance_alert(r2, rmse, threshold=0.70)
    return "fine_tune_model"

# PRIORITY 3: High drift but metrics OK → monitor
if drift and drift_share >= 0.5:
    send_drift_alert(drift_share, r2, drifted_features)
    return "end_monitoring"

# PRIORITY 4: Low drift or good metrics → no action
return "end_monitoring"
```

### In `fine_tune_model()` Task

**Success notification** (after deployment decision, line 598):

```python
send_training_success(
    improvement_delta=r2_improvement,
    new_r2=r2_current,
    old_r2=current_champion_r2,
    deployment_decision=decision  # "deploy", "skip", or "reject"
)
```

**Failure notifications** (exception handlers, lines 620-634):

```python
# Timeout (> 10 minutes)
send_training_failure("Training API timeout (>10 minutes)", dag_run_id)

# Generic exception
send_training_failure(f"Training failed: {str(e)}", dag_run_id)
```

---

## 4. Files Created/Modified

### New Files

1. ✅ [dags/utils/discord_alerts.py](../dags/utils/discord_alerts.py)
    - Discord notification functions
2. ✅ [monitoring/grafana/provisioning/alerting/contactpoints.yml](../monitoring/grafana/provisioning/alerting/contactpoints.yml)
    - Grafana Discord webhook config
3. ✅ [monitoring/grafana/provisioning/alerting/policies.yml](../monitoring/grafana/provisioning/alerting/policies.yml)
    - Alert routing policies
4. ✅ [docs/alerting_setup.md](../docs/alerting_setup.md)
    - This documentation

### Modified Files

1. ✅ [dags/dag_monitor_and_train.py](../dags/dag_monitor_and_train.py)
    - Added Discord alert calls
2. ✅ [docker-compose.yaml](../docker-compose.yaml)
    - Added `DISCORD_WEBHOOK_URL` to Grafana environment
3. ✅ [monitoring/alerts.yml](../monitoring/alerts.yml)
    - Added `ServiceDown` alert rule
4. ✅ [docs/phase4_monitoring_implementation.md](../docs/phase4_monitoring_implementation.md)
    - Updated Phase 5 checklist

---

## 5. Testing Alerts

### 5.1 Test Discord Webhook

```bash
# Send test message to verify webhook works
curl -X POST ${DISCORD_WEBHOOK_URL} \
  -H "Content-Type: application/json" \
  -d '{
    "embeds": [{
      "title": "Test Alert",
      "description": "Discord webhook is working!",
      "color": 65280,
      "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%S.000Z)'"
    }]
  }'
```

Expected: Message appears in Discord channel within seconds.

---

### 5.2 Test Grafana Alerts

**Prerequisites**:

- Grafana running: `docker compose --profile monitoring ps grafana`
- Prometheus targets UP: <http://localhost:9090/targets>

**Test Steps**:

1. Check Grafana contact points:
   - Go to <http://localhost:3000/alerting/notifications>
   - Should see "discord-mlops" contact point
   - Click "Test" → should receive Discord message

2. View active alerts:
   - Go to <http://localhost:3000/alerting/list>
   - Should see alert rules from `monitoring/alerts.yml`

3. Simulate service DOWN:

   ```bash
   # Stop regmodel-backend
   docker compose stop regmodel-backend

   # Wait 2 minutes for Prometheus to detect
   # Check Prometheus alerts: http://localhost:9090/alerts
   # Should see "ServiceDown" alert FIRING

   # Restart service
   docker compose up -d regmodel-backend
   ```

Expected: Discord notification with "Service regmodel-backend is DOWN" within 2-3 minutes.

---

### 5.3 Test DAG Alerts

#### Method 1: Trigger DAG Manually

```bash
# Trigger monitor_and_fine_tune DAG
docker exec airflow-webserver airflow dags trigger monitor_and_fine_tune

# Monitor execution
docker exec airflow-webserver airflow dags state monitor_and_fine_tune

# Check logs for Discord notifications
docker logs airflow-scheduler | grep "Discord notification"
```

**Expected alerts** (depends on current metrics):

- If drift > 50%: Drift alert (orange)
- If R² < 0.70: Performance alert (orange/red)
- If training triggered: Training success/failure alert

#### Method 2: Force Specific Scenario

Modify DAG temporarily to force alerts:

```python
# In end_monitoring() - force drift alert
drift_share = 0.60  # Force high drift
send_drift_alert(drift_share, r2, drifted_features=5)
```

---

### 5.4 Test Training Alerts

**Prerequisites**: Training must be triggered (drift > 50% or R² < 0.65)

**Verify** in Discord:

- **Success**: Green embed with R² improvement, deployment decision
- **Failure**: Red embed with error message and DAG run ID

---

## 6. Troubleshooting

### No Discord messages received

#### Check 1: Webhook URL Configured

```bash
# In container
docker exec grafana printenv DISCORD_WEBHOOK_URL
docker exec airflow-scheduler printenv DISCORD_WEBHOOK_URL
```

Should return webhook URL, not empty.

#### Check 2: Webhook Valid

```bash
# Test webhook directly
curl -X POST $DISCORD_WEBHOOK_URL \
  -H "Content-Type: application/json" \
  -d '{"content": "Test"}'
```

If 404: Webhook deleted or invalid
If 401: Webhook token incorrect
If 200: Webhook works

#### Check 3: Grafana Contact Points

```bash
# Restart Grafana to reload provisioning
docker compose --profile monitoring restart grafana

# Check logs
docker logs grafana | grep -i discord
```

---

### Duplicate alerts

**Cause**: Both Prometheus and DAG sending similar alerts.

**Solution**: This is intentional:

- **Prometheus**: Infrastructure monitoring (always on)
- **DAG**: Pipeline context (only when DAG runs)

If too noisy, adjust `repeat_interval` in [policies.yml](../monitoring/grafana/provisioning/alerting/policies.yml).

---

### Alert not firing in Prometheus

**Check alert rules loaded**:

```bash
# View loaded rules
curl http://localhost:9090/api/v1/rules | jq '.data.groups[].rules[].name'
```

Should include: `ServiceDown`, `ModelPerformanceCritical`, etc.

**Check alert state**:

- Go to <http://localhost:9090/alerts>
- Find your alert
- States: `Inactive` → `Pending` → `Firing`

**Common issues**:

- Metric doesn't exist: Check <http://localhost:9090/graph>, query `bike_model_r2_production`
- `for` duration not met: Wait longer (e.g., 5 minutes for critical alerts)
- Expression wrong: Test PromQL in Prometheus UI

---

### DAG alerts not sent

#### Check Logs

```bash
# Airflow scheduler logs
docker logs airflow-scheduler | grep -A 5 "Discord notification"

# Should see:
# ✅ Discord notification sent: Data Drift Detected
# OR
# ⚠️  Discord webhook URL not configured. Skipping notification.
```

**If webhook not configured**:

```bash
# Add to .env.airflow
echo "DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/..." >> .env.airflow

# Restart Airflow
docker compose restart airflow-scheduler airflow-webserver
```

---

## 7. Alert Examples

### Example 1: Service Down (Prometheus)

**Discord message**:

```text
🚨 MLOps Alert - ServiceDown

Severity: critical
Component: infrastructure

Alert: ServiceDown
Summary: Service regmodel-backend is DOWN
Description: regmodel-backend:8000 has been unreachable for more than 1 minute. MLOps pipeline may be broken.
Status: firing
```

---

### Example 2: High Drift Detected (DAG)

**Discord message** (orange embed):

```text
⚠️  Data Drift Detected

WARNING: High data drift detected in production model.

Impact: Model predictions may degrade over time. Monitor closely and plan retraining.

Drift Share: 54.2%
Production R²: 0.7234
Drifted Features: 8

MLOps Bike Traffic Monitoring
```

---

### Example 3: Training Success - Deployed (DAG)

**Discord message** (green embed):

```text
🚀 Training Completed

Model training completed.

Decision: New model DEPLOYED to production

New R²: 0.7856
Old R²: 0.7234
Improvement: +0.0622

MLOps Bike Traffic Monitoring
```

---

### Example 4: Training Failure (DAG)

**Discord message** (red embed):

```text
🚨 Training Pipeline Failed

CRITICAL: Model training pipeline failed.

Error: Training API timeout (>10 minutes)

Impact: No new model available for deployment.

DAG Run ID: manual__2025-11-04T10:30:00+00:00
Action: Check Airflow logs for details

MLOps Bike Traffic Monitoring
```

---

## 8. Next Steps

After confirming alerts work:

1. ✅ Configure `.env` with real Discord webhook
2. ✅ Test all alert types (service down, drift, training)
3. ⏸️ Adjust `repeat_interval` if too noisy
4. ⏸️ Add more alert rules if needed (e.g., disk space, memory)
5. ⏸️ Set up on-call rotation for critical alerts

---

**See also**:

- [Grafana Dashboards](./grafana_dashboards.md) - Visualize metrics
- [Prometheus Queries](./prometheus_queries.md) - All available metrics
- [Phase 4 Implementation](./phase4_monitoring_implementation.md) - Full monitoring setup

---

**Last updated**: 2025-11-04
