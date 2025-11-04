# MLOps Alerting - Grafana & Discord

**Status**: ✅ Production Ready
**Last Updated**: 2025-11-04
**Alert Rules**: 9 (optimized, non-redundant)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Alert Rules](#alert-rules)
- [Discord Setup](#discord-setup)
- [Testing Alerts](#testing-alerts)
- [Screenshots](#screenshots)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

### Alerting Strategy

**Philosophy**: **Notification-Only Alerts**

- Alerts inform humans for investigation
- No automated remediation actions
- Clear severity levels (CRITICAL, WARNING, INFO)

### Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                     METRICS SOURCES                          │
│  • Airflow Exporter (business metrics)                       │
│  • FastAPI (HTTP metrics)                                    │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│                      PROMETHEUS                              │
│  Scrapes metrics every 10-30s                                │
│  Stores 15 days retention                                    │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│                  GRAFANA UNIFIED ALERTING                    │
│  ┌──────────────────┐  ┌───────────────┐  ┌──────────────┐  │
│  │  Alert Rules     │  │ Notification  │  │  Contact     │  │
│  │  (9 rules)       │→│  Policies     │→│  Points      │  │
│  │  rules.yml       │  │ policies.yml  │  │ Discord      │  │
│  └──────────────────┘  └───────────────┘  └──────┬───────┘  │
└───────────────────────────────────────────────────┼──────────┘
                                                    │
                                                    ▼
                                        ┌──────────────────────┐
                                        │   DISCORD WEBHOOK   │
                                        │   #mlops-alerts     │
                                        └──────────────────────┘
```

### Key Features

- ✅ **9 non-redundant alert rules** (down from 13 obsolete rules)
- ✅ **Aligned with exposed metrics** (no dead metrics)
- ✅ **Proper noDataState handling** (OK vs Alerting)
- ✅ **BigQuery source of truth** for business metrics
- ✅ **Discord notifications** with severity colors
- ✅ **Auto-provisioning** via YAML files

---

## 📊 Alert Rules

### Configuration File

**Location**: `monitoring/grafana/provisioning/alerting/rules.yml`

**Total**: 9 alert rules organized in 6 groups

---

### Group 1: Model Performance (3 rules)

#### 1. Model R² Critically Low

- **UID**: `model_performance_critical`
- **Severity**: CRITICAL
- **Condition**: `bike_model_r2_production < 0.65` for 5 minutes
- **Action**: Immediate retraining required
- **noDataState**: NoData (alert if metric missing)

#### 2. Model R² Declining

- **UID**: `model_performance_warning`
- **Severity**: WARNING
- **Condition**: `bike_model_r2_production < 0.70` (and >= 0.65) for 10 minutes
- **Action**: Plan proactive retraining
- **noDataState**: NoData

#### 3. Model RMSE Above Threshold

- **UID**: `model_rmse_high`
- **Severity**: CRITICAL
- **Condition**: `bike_model_rmse_production > 70` for 5 minutes
- **Action**: Investigate model performance
- **noDataState**: NoData
- **Note**: Threshold adjusted from 60 → 70 (RMSE=66.66 is acceptable with R²=0.867)

---

### Group 2: Data Drift (2 rules)

#### 4. High Data Drift Detected

- **UID**: `high_drift_detected`
- **Severity**: WARNING
- **Condition**: `bike_drift_share > 0.5` for 15 minutes
- **Action**: Monitor closely, prepare for retraining
- **noDataState**: NoData

#### 5. Critical Drift + Declining R²

- **UID**: `critical_drift_with_performance`
- **Severity**: CRITICAL
- **Condition**: `(bike_drift_share > 0.5) AND (bike_model_r2_production < 0.70)` for 5 minutes
- **Action**: Immediate proactive retraining needed
- **noDataState**: **OK** (expression returns empty when condition false)

---

### Group 3: Infrastructure (1 rule)

#### 6. Service Down

- **UID**: `service_down`
- **Severity**: CRITICAL
- **Condition**: `up < 1` for 1 minute
- **Action**: Check `docker compose ps`, restart service
- **noDataState**: Alerting (missing `up` metric = service down)

---

### Group 4: API Health (1 rule)

#### 7. API Error Rate High

- **UID**: `api_error_rate_high`
- **Severity**: CRITICAL
- **Condition**: `rate(fastapi_errors_total[5m]) / rate(fastapi_requests_total[5m]) > 0.05` for 5 minutes
- **Action**: Check FastAPI logs, investigate errors
- **noDataState**: **OK** (no errors = no metric = OK state)

---

### Group 5: Airflow Health (1 rule)

#### 8. DAG Running Too Long

- **UID**: `airflow_dag_too_slow`
- **Severity**: WARNING
- **Condition**: `airflow_dag_run_duration_seconds{dag_id="monitor_and_fine_tune"} > 3600` for 10 minutes
- **Action**: Check for stuck tasks in Airflow UI
- **noDataState**: **OK** (histogram empty before first run = OK)

---

### Group 6: Data Ingestion (1 rule)

#### 9. No Data Ingested

- **UID**: `no_data_ingested`
- **Severity**: WARNING
- **Condition**: `increase(bike_records_ingested_total[1h]) == 0` for 2 hours
- **Action**: Verify Paris Open Data API availability, check DAG logs
- **noDataState**: NoData

---

## 🗑️ Removed Alerts (Cleanup 2025-11-04)

| Alert UID | Reason Removed |
|-----------|----------------|
| `prediction_latency_high` | Used `prediction_latency_seconds` metric (never instrumented in FastAPI) |
| `training_failure` | Used `training_runs_total` metric (FastAPI, never exposed) |
| `airflow_dag_failed` | Used `airflow_task_failures_total` (not exposed by exporter) |
| `low_data_ingestion` | Redundant with `no_data_ingested` (same metric, lower threshold) |

**Result**: 13 rules → **9 rules** (cleaner, aligned, non-redundant)

---

## 🔔 Discord Setup

### 1. Create Webhook

**Steps**:

1. Open Discord → Server Settings → Integrations → Webhooks
2. Click "Create Webhook"
3. Name: "MLOps Bike Traffic Alerts"
4. Select channel: `#mlops-alerts`
5. Copy Webhook URL

### 2. Configure Environment

**File**: `.env` (root directory)

```bash
# Discord webhook for Grafana alerts
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_ID/YOUR_TOKEN

# Grafana admin password (for API access)
GF_SECURITY_ADMIN_PASSWORD=your_secure_password_here
```

**File**: `.env.airflow` (for DAG alerts - optional)

```bash
# Same webhook for Airflow DAG notifications
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_ID/YOUR_TOKEN
```

### 3. Verify Configuration

```bash
# Check Grafana has webhook
docker exec grafana printenv DISCORD_WEBHOOK_URL

# Test webhook directly
curl -X POST "$DISCORD_WEBHOOK_URL" \
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

### 4. Contact Point Configuration

**File**: `monitoring/grafana/provisioning/alerting/contactpoints.yml`

```yaml
apiVersion: 1

contactPoints:
  - orgId: 1
    name: discord-mlops
    receivers:
      - uid: discord-mlops-webhook
        type: discord
        settings:
          url: ${DISCORD_WEBHOOK_URL}
          title: "MLOps Alert - {{ .GroupLabels.alertname }}"
          message: |
            **Severity**: {{ .CommonLabels.severity }}
            **Component**: {{ .CommonLabels.component }}

            **Alert**: {{ .GroupLabels.alertname }}
            **Summary**: {{ index .Annotations "summary" }}
            **Description**: {{ index .Annotations "description" }}
            **Status**: {{ .Status }}
```

### 5. Notification Policy

**File**: `monitoring/grafana/provisioning/alerting/policies.yml`

```yaml
apiVersion: 1

policies:
  - orgId: 1
    receiver: discord-mlops
    group_by: ['alertname', 'severity']
    group_wait: 30s
    group_interval: 5m
    repeat_interval: 4h
    routes:
      - receiver: discord-mlops
        matchers:
          - severity = critical
        repeat_interval: 2h
      - receiver: discord-mlops
        matchers:
          - severity = warning
        repeat_interval: 4h
```

**Repeat Intervals**:

- CRITICAL: Every 2 hours
- WARNING: Every 4 hours
- Others: Every 4 hours (default)

---

## 🧪 Testing Alerts

### Test 1: Test Contact Point (Grafana UI)

```bash
# 1. Open Grafana
open http://localhost:3000

# 2. Navigate to: Alerting → Contact points
# 3. Find "discord-mlops"
# 4. Click "Test"
# 5. Click "Send test notification"
```

**Expected**: Discord message within seconds:

```text
🚨 MLOps Alert - [FIRING:1]

Severity: info
Component: testing

Alert: TestAlert
Summary: This is a test notification
Status: firing
```

---

### Test 2: Force Service Down Alert

```bash
# Stop regmodel-backend
docker compose stop regmodel-backend

# Wait 90 seconds (1 min for condition + 30s for Prometheus scrape)

# Check Prometheus alerts
curl -s http://localhost:9090/api/v1/alerts | jq '.data.alerts[] | select(.labels.alertname=="ServiceDown")'

# Expected state: "firing"

# Restart service
docker compose up -d regmodel-backend
```

**Expected Discord Message**:

```text
🚨 MLOps Alert - Service Down

Severity: critical
Component: infrastructure

Alert: ServiceDown
Summary: Service is DOWN
Description: Service has been unreachable for more than 1 minute. MLOps pipeline may be broken.
Status: firing
```

---

### Test 3: Check Alert States

```bash
# List all alert rules and their state
curl -s "http://localhost:3000/api/v1/provisioning/alert-rules" \
  -u "admin:$GF_SECURITY_ADMIN_PASSWORD" | \
  jq -r '.[] | "\(.title): \(.uid)"'

# Check firing alerts
curl -s "http://localhost:3000/api/alertmanager/grafana/api/v2/alerts" \
  -u "admin:$GF_SECURITY_ADMIN_PASSWORD" | \
  jq '.[] | {name: .labels.alertname, severity: .labels.severity, state: .status.state}'
```

**Expected** (after cleanup):

- Total rules: 9
- Firing alerts: 0-1 (only legitimate alerts)

---

### Test 4: Simulate High RMSE

**Method**: Temporarily modify threshold in rules.yml

```yaml
# Edit monitoring/grafana/provisioning/alerting/rules.yml
# Change RMSE threshold from 70 to 60

                - evaluator:
                    params:
                      - 60  # Lower threshold to trigger alert
                    type: gt
```

```bash
# Restart Grafana
docker compose --profile monitoring restart grafana

# Wait 5 minutes for evaluation
# Check Prometheus alert state
curl -s http://localhost:9090/api/v1/alerts | grep model_rmse_high

# Revert threshold back to 70
```

---

## 📸 Screenshots

> **👉 Add your screenshots here**

Create the following screenshots and save them to `docs/monitoring/screenshots/`:

### Required Screenshots

#### 1. `grafana_alert_rules_list.png`

- **Path**: Grafana → Alerting → Alert Rules
- **Show**: All 9 alert rules with state (Normal/Firing)
- **Highlight**: No obsolete rules (prediction_latency, training_failure removed)

#### 2. `grafana_alert_rules_detail.png`

- **Path**: Click on one alert rule (e.g., "Model R² Critically Low")
- **Show**: Query, thresholds, noDataState configuration

#### 3. `discord_alert_critical.png`

- **Show**: Example of CRITICAL alert in Discord
- **Content**: Service Down or Model R² Critically Low
- **Highlight**: Red color, severity level, actionable description

#### 4. `discord_alert_warning.png`

- **Show**: Example of WARNING alert
- **Content**: High Drift Detected or Model R² Declining
- **Highlight**: Orange color, less urgent

#### 5. `grafana_contact_points.png`

- **Path**: Grafana → Alerting → Contact points
- **Show**: discord-mlops contact point configured
- **Highlight**: Test button, last delivery status

#### 6. `grafana_notification_policies.png`

- **Path**: Grafana → Alerting → Notification policies
- **Show**: Routing tree with repeat intervals

---

### Screenshot Embedding

Once you've added screenshots, reference them like this:

```markdown
### Alert Rules Overview

![Grafana Alert Rules](./screenshots/grafana_alert_rules_list.png)

*Figure 1: All 9 alert rules in Grafana (post-cleanup)*

### Discord Notifications

![Critical Alert in Discord](./screenshots/discord_alert_critical.png)

*Figure 2: Example of CRITICAL alert sent to Discord #mlops-alerts channel*
```

---

## 🔧 Troubleshooting

### No Discord notifications received

#### Check 1: Webhook Valid

```bash
# Test webhook directly
curl -X POST "$DISCORD_WEBHOOK_URL" \
  -H "Content-Type: application/json" \
  -d '{"content": "Test from curl"}'
```

**Possible responses**:

- `200 OK`: Webhook works ✅
- `404 Not Found`: Webhook deleted or invalid ❌
- `401 Unauthorized`: Wrong token ❌

#### Check 2: Grafana Logs

```bash
# Check for Discord delivery errors
docker logs grafana 2>&1 | grep -i discord

# Look for:
# ✅ "Notification sent successfully"
# ❌ "Failed to send notification"
```

#### Check 3: Contact Point Status

```bash
# Via Grafana UI
# → Alerting → Contact points → discord-mlops
# Check "Last Delivery" column
```

---

### Alert not firing when condition met

#### Check 1: Metric Exists

```bash
# Query Prometheus for the metric
curl -s "http://localhost:9090/api/v1/query?query=bike_model_r2_production" | jq

# If empty result: metric not being scraped
# Check: curl http://localhost:9101/metrics | grep bike_model_r2_production
```

#### Check 2: Alert Evaluation

```bash
# Check alert state in Prometheus
curl -s http://localhost:9090/api/v1/rules | \
  jq '.data.groups[].rules[] | select(.name=="ModelPerformanceCritical")'

# States:
# - inactive: condition not met
# - pending: condition met, waiting for "for" duration
# - firing: alert active
```

#### Check 3: noDataState Configuration

For alerts with `noDataState: OK`:

- If query returns no data (empty result), alert state = OK (no notification)
- Example: `api_error_rate_high` won't fire if `fastapi_errors_total` doesn't exist (no errors = OK)

For alerts with `noDataState: NoData`:

- If metric missing, alert fires with "No data" state

---

### Too many alerts (alert fatigue)

#### Solution 1: Adjust Repeat Interval

Edit `monitoring/grafana/provisioning/alerting/policies.yml`:

```yaml
# Increase repeat interval for non-critical alerts
- receiver: discord-mlops
  matchers:
    - severity = warning
  repeat_interval: 12h  # Instead of 4h
```

#### Solution 2: Increase Thresholds

Edit thresholds in `rules.yml`:

- RMSE threshold: 70 → 80
- Drift threshold: 0.5 → 0.6
- R² warning: 0.70 → 0.68

#### Solution 3: Use Silences

Temporarily mute alerts via Grafana UI:

- Alerting → Silences → New Silence
- Matcher: `alertname = no_data_ingested`
- Duration: 24 hours

---

## 📚 Related Documentation

- [01_architecture.md](./01_architecture.md) - Monitoring stack overview
- [03_metrics_reference.md](./03_metrics_reference.md) - Complete metrics catalog
- [contactpoints.yml](../../monitoring/grafana/provisioning/alerting/contactpoints.yml) - Discord config
- [policies.yml](../../monitoring/grafana/provisioning/alerting/policies.yml) - Routing policies
- [rules.yml](../../monitoring/grafana/provisioning/alerting/rules.yml) - Alert definitions

---

## 📝 Alert Changelog

### 2025-11-04: Major Cleanup

**Removed** (4 alerts):

- `prediction_latency_high` - Dead metric (FastAPI never instrumented)
- `training_failure` - Dead metric (FastAPI never instrumented)
- `airflow_dag_failed` - Dead metric (exporter doesn't expose)
- `low_data_ingestion` - Redundant (same as `no_data_ingested`)

**Modified** (4 alerts):

- `model_rmse_high`: Threshold 60 → 70
- `api_error_rate_high`: noDataState NoData → OK
- `critical_drift_with_performance`: noDataState NoData → OK
- `airflow_dag_too_slow`: noDataState NoData → OK

**Result**:

- 13 rules → 9 rules (-31%)
- 9 firing alerts → 0-1 firing alerts (-89% false positives)
- 100% metrics alignment ✅

---

**Last updated**: 2025-11-04
**Status**: ✅ Production Ready (9 alert rules, aligned with metrics)
