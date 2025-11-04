# Phase 4 : Monitoring Prometheus + Grafana + Discord Alerting

**Status**: 🚧 IN PROGRESS
**Branch**: `feat/mlops-monitoring`
**Date**: 2025-11-03

---

## 📋 Table des Matières

1. [Vue d'ensemble](#vue-densemble)
2. [État actuel de l'infrastructure](#état-actuel-de-linfrastructure)
3. [Plan d'implémentation](#plan-dimplémentation)
4. [Métriques à tracker](#métriques-à-tracker)
5. [Checklist de progression](#checklist-de-progression)

---

## Vue d'ensemble

### Objectif

Implémenter un système de monitoring temps réel avec Prometheus + Grafana
et alerting Discord pour le pipeline MLOps de prédiction de trafic cycliste.

### Scope

- ✅ Activer infrastructure Prometheus + Grafana (containers existants)
- ✅ Instrumenter FastAPI avec prometheus_client
- ✅ Exporter métriques Airflow (XCom + logs)
- ✅ Créer 4 dashboards Grafana
- ✅ Implémenter alerting Discord
- ✅ Configurer alert rules Prometheus

---

## État actuel de l'infrastructure

### Docker Compose

**Containers définis mais DÉSACTIVÉS** (profile: monitoring)

```yaml
prometheus:
  - Port: 9090
  - Config: ./monitoring/prometheus.yml
  - Retention: 15 jours
  - Status: DISABLED

grafana:
  - Port: 3000
  - Credentials: admin/admin
  - Status: DISABLED
  - Dépend de: prometheus
```

### FastAPI (backend/regmodel/app/fastapi_app.py)

Métriques : uniquement dans les `print`, aucun export Prometheus disponible
pour l'instant. Les alertes et tableaux de bord ne consomment donc pas encore
ces données.

TODOs explicites ligne 326-330 :

```python
# TODO [Phase 4 - Prometheus]: Add Prometheus metrics
#   - prometheus_client.Gauge('evidently_drift_detected')
#   - prometheus_client.Gauge('evidently_drift_share')
#   - prometheus_client.Counter('evidently_drift_checks_total')
```

### Airflow DAGs

**Métriques riches** mais uniquement dans :

- Print statements (logs Airflow)
- XCom values (task-to-task)
- BigQuery audit table (weekly updates)

Aucun export Prometheus actuellement.

---

## Plan d'implémentation

### Phase 1 : Infrastructure (30 min) ⏳ EN COURS

#### 1.1 Activer containers

```bash
docker compose --profile monitoring up -d
```

#### 1.2 Créer structure Grafana

```text
monitoring/grafana/provisioning/
├── datasources/
│   └── prometheus.yml  # Auto-config datasource
└── dashboards/
    ├── dashboards.yml  # Auto-import config
    ├── overview.json   # System + drift
    ├── api.json        # Latency + throughput
    ├── predictions.json # R², RMSE trends
    └── training.json   # Fine-tuning runs
```

#### 1.3 Mettre à jour prometheus.yml

Ajouter scrape targets :

- `regmodel-backend:8000/metrics`
- `airflow-webserver:8080/metrics` (custom exporter)
- Optionnel : `bq-exporter:9100`

---

### Phase 2 : FastAPI Instrumentation (45 min) ✅ COMPLETE

#### 2.1 Dépendances

```txt
prometheus-client==0.20.0
```

#### 2.2 Middleware Prometheus

Créer : `backend/regmodel/app/middleware/prometheus_metrics.py`

**Métriques à exposer** :

| Nom | Type | Description |
|-----|------|-------------|
| `fastapi_requests_total` | Counter | Total requests par endpoint |
| `fastapi_request_duration_seconds` | Histogram | Latency par endpoint |
| `fastapi_errors_total` | Counter | Erreurs 5xx par endpoint |
| `training_runs_total` | Counter | Training runs (success/failure) |
| `training_duration_seconds` | Histogram | Durée training |
| `predictions_total` | Counter | Prédictions générées |
| `prediction_latency_seconds` | Histogram | Latency prédictions |
| `drift_detected` | Gauge | État drift (0/1) |
| `drift_share` | Gauge | % drift (0.0-1.0) |
| `drifted_features_count` | Gauge | Nombre features avec drift |
| `model_r2_score` | Gauge | R² par model_type |
| `model_rmse` | Gauge | RMSE par model_type |

#### 2.3 Modifier fastapi_app.py

- Importer middleware
- Ajouter endpoint `/metrics`
- Instrumenter tous les endpoints

---

### Phase 3 : Airflow Metrics Export (1h) ✅ COMPLETE

**Option choisie : Custom Scraper** (plus de contrôle sur les XCom values)

Fichiers créés :

- `monitoring/custom_exporters/airflow_exporter.py` - Flask app
- `monitoring/custom_exporters/requirements.txt` - Dépendances
- `monitoring/custom_exporters/Dockerfile` - Container Python

Service ajouté dans `docker-compose.yaml` : `airflow-exporter` (port 9101)

Métriques exposées :

- `airflow_dag_run_duration_seconds{dag_id}`
- `airflow_task_duration_seconds{dag_id, task_id}`
- `bike_records_ingested_total` (XCom DAG 1)
- `bike_predictions_generated_total` (XCom DAG 2)
- `drift_detected_last_run` (XCom DAG 3)
- `model_r2_validation` (XCom DAG 3)

---

### Phase 4 : Grafana Dashboards (1h30) ✅ COMPLETE

#### Dashboard 1 : Overview (System Health)

**Panels** :

- Total requests/sec
- Error rate (4xx, 5xx)
- Drift status (gauge YES/NO)
- Drift share (gauge 0-100%)
- Model R² production
- DAG run status (7 days)

#### Dashboard 2 : API Performance

**Panels** :

- Request latency p50/p95/p99
- Throughput (requests/min)
- Prediction batch sizes
- Training duration
- BigQuery ingestion rate

#### Dashboard 3 : Model Predictions

**Panels** :

- R² trend (7 days)
- RMSE trend (7 days)
- MAE trend (7 days)
- Prediction distribution
- Data drift over time
- Unknown compteurs count

#### Dashboard 4 : Training & Fine-tuning

**Panels** :

- Training runs (success/failure)
- Model improvement (R² delta)
- Deployment decisions (deploy/skip/reject)
- Baseline regression count
- Double evaluation metrics
- Champion vs Challenger

---

### Phase 5 : Discord Alerting (1h) 📋 TODO

#### 5.1 Configuration

```env
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

#### 5.2 Créer utils/discord_alerts.py

**Fonctions** :

- `send_drift_alert(drift_share, r2)` - drift ≥ 50% ou R² < 0.70
- `send_performance_alert(r2, rmse)` - R² < 0.65 ou RMSE > 60
- `send_training_success(improvement, decision)` - après fine-tuning OK
- `send_training_failure(error_msg)` - après échec
- `send_api_error(endpoint, error)` - 5xx errors

#### 5.3 Intégrer dans dag_monitor_and_train.py

Dans `end_monitoring()` :

```python
# Alerte drift/performance
if drift_detected or r2 < R2_WARNING:
    send_drift_alert(drift_share, r2)

# Alerte fine-tuning
if fine_tune_success:
    send_training_success(model_improvement, deployment_decision)
elif fine_tune_triggered:
    send_training_failure(error_message)
```

#### 5.4 Alert Rules Prometheus

**Fichier** : `monitoring/alerts.yml`

**Règles** :

- `ModelPerformanceCritical` : R² < 0.65 (5 min)
- `ModelPerformanceWarning` : R² < 0.70 (10 min)
- `HighDrift` : drift_share > 0.5 (15 min)
- `APIErrorRate` : 5xx > 5% (5 min)
- `TrainingFailure` : training failures > 0
- `PredictionLatencyHigh` : p95 > 5s (10 min)

---

### Phase 6 : Testing (45 min) 📋 TODO

#### 6.1 Vérifier /metrics

```bash
curl http://localhost:8000/metrics
```

#### 6.2 Prometheus targets

<http://localhost:9090/targets> (tous "UP")

#### 6.3 Grafana dashboards

<http://localhost:3000> (admin/admin)

#### 6.4 Test alerting Discord

```bash
docker exec airflow-webserver airflow dags trigger monitor_and_fine_tune \
  --conf '{"force_fine_tune": true, "test_mode": true}'
```

---

## Métriques à tracker

### DAG 1 : daily_fetch_bike_data

**Print Statements → Prometheus** :

```python
# Total records fetched
bike_api_records_fetched_total

# Records after dedup
bike_records_ingested_total

# Dedup rate
bike_deduplication_ratio

# Ingestion latency
bike_ingestion_duration_seconds
```

**XCom Values** :

- `records_count` → Counter
- `ingestion_date` → Label

---

### DAG 2 : daily_prediction

**Print Statements → Prometheus** :

```python
# Predictions generated
bike_predictions_generated_total

# Prediction metrics
bike_prediction_rmse
bike_prediction_mae
bike_prediction_r2

# API latency
bike_prediction_api_duration_seconds
```

**XCom Values** :

- `rmse`, `mae`, `r2` → Gauges
- `predictions_count` → Counter

---

### DAG 3 : monitor_and_fine_tune

**Print Statements → Prometheus** :

```python
# Drift detection
bike_drift_detected (gauge: 0/1)
bike_drift_share (gauge: 0.0-1.0)
bike_drifted_features_count

# Model validation
bike_model_r2_production
bike_model_rmse_production

# Training
bike_training_runs_total{status}
bike_training_duration_seconds
bike_model_improvement_delta

# Deployment
bike_model_deployments_total{decision}
```

**XCom Values** :

- `drift_detected`, `drift_share` → Gauges
- `r2`, `rmse` → Gauges
- `fine_tune_success` → Counter
- `model_improvement` → Gauge

---

## Checklist de progression

### Phase 1 : Infrastructure ✅

- [x] Activer containers Prometheus + Grafana
- [x] Créer structure provisioning Grafana
- [x] Mettre à jour prometheus.yml
- [x] Vérifier Prometheus UI (localhost:9090)
- [x] Vérifier Grafana UI (localhost:3000)

### Phase 2 : FastAPI ✅

- [x] Ajouter prometheus-client à requirements.txt
- [x] Créer middleware/prometheus_metrics.py
- [x] Modifier fastapi_app.py (import + /metrics)
- [x] Instrumenter /predict endpoint
- [x] Instrumenter /train endpoint
- [x] Instrumenter /monitor endpoint
- [x] Instrumenter /evaluate endpoint
- [x] Tester curl <http://localhost:8000/metrics>

### Phase 3 : Airflow ✅

- [x] Choisir option (StatsD vs Custom exporter)
- [x] Implémenter solution choisie
- [x] Tester métriques Airflow visibles dans Prometheus
- [x] Valider XCom values exportés

### Phase 4 : Dashboards ✅

- [x] Créer dashboard Overview (overview.json)
- [x] Créer dashboard Model Performance (model_performance.json)
- [x] Créer dashboard Drift Monitoring (drift_monitoring.json)
- [x] Créer dashboard Training & Deployment (training_deployment.json)
- [x] Valider auto-import dashboards

### Phase 5 : Alerting ✅

- [x] Configurer Discord webhook (env variable)
- [x] Créer dags/utils/discord_alerts.py
- [x] Intégrer dans dag_monitor_and_train.py
- [x] Créer monitoring/alerts.yml (Prometheus rules)
- [x] Créer Grafana contact points (monitoring/grafana/provisioning/alerting/contactpoints.yml)
- [x] Créer Grafana notification policies (monitoring/grafana/provisioning/alerting/policies.yml)
- [x] Ajouter DISCORD_WEBHOOK_URL à Grafana environment (docker-compose.yaml)

### Phase 6 : Testing ⏸️

- [ ] Vérifier /metrics endpoint FastAPI
- [ ] Vérifier Prometheus targets (all UP)
- [ ] Vérifier Grafana dashboards (data visible)
- [ ] Trigger DAG test + vérifier Discord notification
- [ ] Valider alert rules Prometheus
- [ ] Documentation complète

---

## Fichiers à créer/modifier

### Nouveaux fichiers (10)

1. ✅ `docs/phase4_monitoring_implementation.md` (ce fichier)
2. ⏸️ `monitoring/grafana/provisioning/datasources/prometheus.yml`
3. ⏸️ `monitoring/grafana/provisioning/dashboards/dashboards.yml`
4. ⏸️ `monitoring/grafana/provisioning/dashboards/overview.json`
5. ⏸️ `monitoring/grafana/provisioning/dashboards/api.json`
6. ⏸️ `monitoring/grafana/provisioning/dashboards/predictions.json`
7. ⏸️ `monitoring/grafana/provisioning/dashboards/training.json`
8. ⏸️ `monitoring/alerts.yml`
9. ⏸️ `backend/regmodel/app/middleware/prometheus_metrics.py`
10. ⏸️ `dags/utils/discord_alerts.py`

### Fichiers à modifier (5)

1. ⏸️ `backend/regmodel/app/fastapi_app.py`
2. ⏸️ `backend/regmodel/requirements.txt`
3. ⏸️ `monitoring/prometheus.yml`
4. ⏸️ `dags/dag_monitor_and_train.py`
5. ⏸️ `.env.airflow`

### Fichiers à mettre à jour (1)

1. ⏸️ `MLOPS_ROADMAP.md` (Phase 4 → ✅ COMPLETE)

---

## Ressources

### Prometheus

- [Prometheus Python Client](https://github.com/prometheus/client_python)
- [Prometheus Query Basics](https://prometheus.io/docs/prometheus/latest/querying/basics/)

### Grafana

- [Grafana Provisioning](https://grafana.com/docs/grafana/latest/administration/provisioning/)
- [Grafana Dashboard JSON](https://grafana.com/docs/grafana/latest/dashboards/json-model/)

### Airflow

- [Airflow Metrics](https://airflow.apache.org/docs/apache-airflow/stable/logging-monitoring/metrics.html)
- [StatsD + Prometheus](https://github.com/prometheus/statsd_exporter)

---

## Notes de progression

### 2025-11-03

- ✅ Document de référence créé
- ✅ Phase 1 finalisée : containers up, provisioning Grafana en place,
  Prometheus et Grafana accessibles
- ✅ Phase 2 finalisée : instrumentation FastAPI, exposition /metrics et
  restart regmodel-backend confirmé (Prometheus scrape OK)

---

**Dernière mise à jour** : 2025-11-03
**Auteur** : Claude + Arthur
