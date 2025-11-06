# Infrastructure Documentation

**Purpose**: Complete documentation of the MLOps infrastructure including Docker containers, GCP services, and external integrations.

**Last Updated**: 2025-11-06

---

## Table of Contents

1. [Overview](#overview)
2. [Docker Services](#docker-services)
3. [GCP Services](#gcp-services)
4. [External Services](#external-services)
5. [Network & Ports](#network--ports)
6. [Environment Configuration](#environment-configuration)
7. [Resource Management](#resource-management)
8. [Service Dependencies](#service-dependencies)
9. [Profiles & Optional Services](#profiles--optional-services)

---

## Overview

The infrastructure is organized in a layered architecture:

```
┌─────────────────────────────────────────────────┐
│  External Services (APIs, Alerts)               │
│  - Paris Open Data API                          │
│  - Discord Webhook                              │
└─────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────┐
│  GCP Cloud Services                             │
│  - BigQuery (3 datasets)                        │
│  - Cloud SQL PostgreSQL (MLflow)                │
│  - Cloud Storage (Artifacts, Models, Data)      │
└─────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────┐
│  Docker Services (12 containers)                │
│  - MLflow Stack (2)                             │
│  - FastAPI Backend (1)                          │
│  - Airflow Stack (6)                            │
│  - Monitoring Stack (3) [optional]              │
└─────────────────────────────────────────────────┘
```

---

## Docker Services

### 1. MLflow Stack

#### 1.1 Cloud SQL Proxy

- **Container**: `cloud-sql-proxy`
- **Image**: `gcr.io/cloud-sql-connectors/cloud-sql-proxy:latest`
- **Purpose**: Secure connection to Cloud SQL PostgreSQL for MLflow metadata

**Configuration**:
- **Port**: 5432 (internal)
- **Command**:
  - `--address=0.0.0.0`
  - `--port=5432`
  - `--health-check`
  - Instance connection from `${MLFLOW_INSTANCE_CONNECTION}`
- **Volumes**:
  - `./gcp.json:/config:ro` (GCP service account credentials)
- **Environment**:
  - `GOOGLE_APPLICATION_CREDENTIALS=/config`
- **Restart Policy**: `unless-stopped`

**Purpose**: Provides encrypted connection to Cloud SQL without exposing database credentials or requiring direct network access.

#### 1.2 MLflow Tracking Server

- **Container**: `mlflow-server`
- **Image**: `ghcr.io/mlflow/mlflow:v2.22.0`
- **Purpose**: Centralized experiment tracking and model registry

**Configuration**:
- **Port**: 5000:5000
- **Volumes**:
  - `./mlflow-ui-access.json:/mlflow/gcp.json:ro` (GCS read/write access)
- **Environment**:
  - `GOOGLE_APPLICATION_CREDENTIALS=/mlflow/gcp.json`
  - `MLFLOW_DB_USER=${MLFLOW_DB_USER}`
  - `MLFLOW_DB_PASSWORD=${MLFLOW_DB_PASSWORD}`
  - `MLFLOW_DB_NAME=${MLFLOW_DB_NAME}`
- **Dependencies**: `cloud-sql-proxy`
- **Backend Store**: PostgreSQL via Cloud SQL Proxy
- **Artifact Store**: `gs://df_traffic_cyclist1/mlflow-artifacts`
- **Restart Policy**: `unless-stopped`

**Entrypoint**: Installs `google-cloud-storage` and `psycopg2-binary`, waits for proxy readiness, then starts MLflow server.

---

### 2. FastAPI Backend

#### 2.1 RegModel API

- **Container**: `regmodel-api`
- **Image**: Built from `backend/regmodel/Dockerfile`
- **Purpose**: Main ML prediction and training API

**Configuration**:
- **Port**: 8000:8000
- **Memory Limits**:
  - Limit: 6GB
  - Reservation: 2GB
- **Volumes**:
  - `./backend/regmodel/app:/app/app` (hot reload)
  - `./mlflow-trainer.json:/app/gcp.json:ro` (GCS access)
  - **Data mounts** (read-only):
    - `./data/train_baseline.csv` (724K rows)
    - `./data/test_baseline.csv`
    - `./data/test_sample.csv`
    - `./data/reference_data.csv`
    - `./data/reference_data_sample.csv`
    - `./data/current_data.csv`
- **Environment**:
  - `MLFLOW_TRACKING_URI=http://mlflow:5000`
  - `GOOGLE_APPLICATION_CREDENTIALS=/app/gcp.json`
  - `MODEL_ENV=dev` (or `prod`)
  - `MODEL_TEST_MODE=false` (or `true`)
  - `MODEL_SUMMARY_PATH=gs://df_traffic_cyclist1/models/summary.json`
- **Dependencies**: `mlflow`
- **Command**: `uvicorn app.fastapi_app:app --host 0.0.0.0 --port 8000`

**Key Endpoints**:
- `POST /train` - Train new models
- `POST /predict` - Generate predictions
- `POST /promote_champion` - Deploy champion model
- `GET /metrics` - Prometheus metrics

---

### 3. Airflow Stack

#### 3.1 PostgreSQL Database

- **Container**: `postgres-airflow`
- **Image**: `postgres:15-alpine`
- **Purpose**: Airflow metadata database

**Configuration**:
- **Internal Port**: 5432
- **Environment**:
  - `POSTGRES_USER=airflow`
  - `POSTGRES_PASSWORD=airflow`
  - `POSTGRES_DB=airflow`
- **Volumes**:
  - `postgres_airflow_data:/var/lib/postgresql/data` (persistent)
- **Healthcheck**: `pg_isready -U airflow` every 5s
- **Restart Policy**: `always`

#### 3.2 Redis

- **Container**: `redis-airflow`
- **Image**: `redis:latest`
- **Purpose**: Celery broker for distributed task execution

**Configuration**:
- **Port**: 6379:6379
- **Healthcheck**: `redis-cli ping` every 5s
- **Restart Policy**: `always`

#### 3.3 Airflow Common Configuration

**Base Configuration** (shared by all Airflow containers via `&airflow-common`):

**Image**: `apache/airflow:2.10.0-python3.12`

**Environment Variables**:
- **Airflow Core**:
  - `AIRFLOW__CORE__EXECUTOR=CeleryExecutor`
  - `AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres-airflow/airflow`
  - `AIRFLOW__CELERY__RESULT_BACKEND=db+postgresql://airflow:airflow@postgres-airflow/airflow`
  - `AIRFLOW__CELERY__BROKER_URL=redis://:@redis-airflow:6379/0`
  - `AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=true`
  - `AIRFLOW__CORE__LOAD_EXAMPLES=false`
  - `AIRFLOW__API__AUTH_BACKEND=airflow.api.auth.backend.basic_auth`

- **Project-Specific**:
  - `ENV=${ENV:-DEV}`
  - `GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT:-datascientest-460618}`
  - `GOOGLE_APPLICATION_CREDENTIALS=/opt/airflow/gcp.json`
  - `BQ_PROJECT=${BQ_PROJECT:-datascientest-460618}`
  - `BQ_RAW_DATASET=${BQ_RAW_DATASET:-bike_traffic_raw}`
  - `BQ_PREDICT_DATASET=${BQ_PREDICT_DATASET:-bike_traffic_predictions}`
  - `BQ_LOCATION=${BQ_LOCATION:-europe-west1}`
  - `GCS_BUCKET=${GCS_BUCKET:-df_traffic_cyclist1}`
  - `API_URL_DEV=${API_URL_DEV}`
  - `API_KEY_SECRET=${API_KEY_SECRET:-dev-key-unsafe}`
  - `R2_CRITICAL=${R2_CRITICAL}`
  - `R2_WARNING=${R2_WARNING}`
  - `RMSE_THRESHOLD=${RMSE_THRESHOLD}`

- **Additional Packages**:
  - `_PIP_ADDITIONAL_REQUIREMENTS=pandas scikit-learn requests google-cloud-bigquery google-cloud-storage gcsfs db-dtypes redis<5.0.0`

**Shared Volumes**:
- `./dags:/opt/airflow/dags`
- `./logs:/opt/airflow/logs`
- `./plugins:/opt/airflow/plugins`
- `./gcp.json:/opt/airflow/gcp.json:ro`
- `./data:/opt/airflow/data:ro`
- `./shared_data:/app/shared_data` (dev only)
- `./models:/app/models` (dev only)

**User**: `${AIRFLOW_UID:-50000}:${AIRFLOW_GID:-50000}`

#### 3.4 Airflow Webserver

- **Container**: `airflow-webserver`
- **Purpose**: Web UI for DAG management and monitoring

**Configuration**:
- **Port**: 8081:8080
- **Command**: `webserver`
- **Additional Environment**:
  - `DISCORD_WEBHOOK_URL=${DISCORD_WEBHOOK_URL}`
  - `_AIRFLOW_DB_MIGRATE=false`
- **Dependencies**:
  - `postgres-airflow` (healthy)
  - `redis-airflow` (healthy)
  - `airflow-init` (healthy)
  - `mlflow` (started)
  - `regmodel-backend` (started)
- **Restart Policy**: `always`

#### 3.5 Airflow Scheduler

- **Container**: `airflow-scheduler`
- **Purpose**: DAG scheduling and task orchestration

**Configuration**:
- **Command**: `scheduler`
- **Environment**:
  - `_AIRFLOW_DB_MIGRATE=false`
- **Dependencies**:
  - `airflow-init` (healthy)
  - `airflow-webserver` (started)
- **Restart Policy**: `always`

#### 3.6 Airflow Worker

- **Container**: `airflow-worker`
- **Purpose**: Celery worker for distributed task execution

**Configuration**:
- **Command**: `celery worker`
- **Environment**:
  - `_AIRFLOW_DB_MIGRATE=false`
- **Dependencies**:
  - `airflow-init` (healthy)
  - `airflow-webserver` (started)
- **Restart Policy**: `always`

#### 3.7 Airflow Init

- **Container**: `airflow-init`
- **Purpose**: Database initialization and admin user creation

**Configuration**:
- **Entrypoint**: `/bin/bash`
- **Command**: Multi-step initialization script:
  1. Wait for PostgreSQL readiness
  2. Initialize Airflow database
  3. Upgrade database schema
  4. Create admin user
  5. Keep alive for healthcheck
- **Environment**:
  - `_AIRFLOW_WWW_USER_USERNAME=${_AIRFLOW_WWW_USER_USERNAME:-admin}`
  - `_AIRFLOW_WWW_USER_PASSWORD=${_AIRFLOW_WWW_USER_PASSWORD:-admin}`
- **Healthcheck**: Check for `/tmp/airflow-init-status` file
- **Dependencies**:
  - `postgres-airflow` (healthy)
  - `redis-airflow` (healthy)

#### 3.8 Flower

- **Container**: `flower`
- **Purpose**: Celery monitoring UI

**Configuration**:
- **Port**: 5555:5555
- **Command**: `flower`
- **Environment**:
  - `_AIRFLOW_DB_MIGRATE=false`
- **Dependencies**:
  - `airflow-init` (healthy)
  - `airflow-webserver` (started)
- **Healthcheck**: `curl --fail http://localhost:5555/` every 10s
- **Restart Policy**: `always`

---

### 4. Monitoring Stack (Optional - Profile: `monitoring`)

#### 4.1 Prometheus

- **Container**: `prometheus`
- **Image**: `prom/prometheus:latest`
- **Purpose**: Time-series metrics collection and storage

**Configuration**:
- **Port**: 9090:9090
- **Volumes**:
  - `./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro`
  - `prometheus_data:/prometheus` (persistent)
- **Command**:
  - `--config.file=/etc/prometheus/prometheus.yml`
  - `--storage.tsdb.path=/prometheus`
  - `--storage.tsdb.retention.time=15d`
  - `--web.enable-admin-api`
- **Restart Policy**: `always`
- **Profile**: `monitoring`

**Scraping Targets**:
- `regmodel-api:8000/metrics` (FastAPI metrics)
- `airflow-exporter:9101/metrics` (Custom MLOps metrics)

#### 4.2 Prometheus Pushgateway

- **Container**: `pushgateway`
- **Image**: `prom/pushgateway:latest`
- **Purpose**: Push-based metrics for batch jobs and testing alerts

**Configuration**:
- **Port**: 9091:9091
- **Restart Policy**: `always`
- **Profile**: `monitoring`

#### 4.3 Grafana

- **Container**: `grafana`
- **Image**: `grafana/grafana:latest`
- **Purpose**: Visualization dashboards and alerting

**Configuration**:
- **Port**: 3000:3000
- **Environment**:
  - `GF_SECURITY_ADMIN_PASSWORD=${GF_SECURITY_ADMIN_PASSWORD}`
  - `GF_SERVER_ROOT_URL=http://localhost:3000`
  - `GF_USERS_ALLOW_SIGN_UP=false`
  - `DISCORD_WEBHOOK_URL=${DISCORD_WEBHOOK_URL}`
- **Volumes**:
  - `grafana_data:/var/lib/grafana` (persistent)
  - `./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro`
- **Dependencies**: `prometheus`
- **Restart Policy**: `always`
- **Profile**: `monitoring`

**Dashboards**:
1. **Overview** - System health, drift status, API metrics
2. **Model Performance** - R² trends, RMSE
3. **Drift Monitoring** - Feature drift evolution
4. **Training & Deployment** - Success rates, deployment decisions

#### 4.4 Airflow Exporter

- **Container**: `airflow-exporter`
- **Image**: Built from `monitoring/custom_exporters/Dockerfile`
- **Purpose**: Custom Prometheus exporter for Airflow and MLflow metrics

**Configuration**:
- **Port**: 9101:9101
- **Environment**:
  - `AIRFLOW_BASE_URL=http://airflow-webserver:8080`
  - `AIRFLOW_USERNAME=${_AIRFLOW_WWW_USER_USERNAME:-admin}`
  - `AIRFLOW_PASSWORD=${_AIRFLOW_WWW_USER_PASSWORD:-admin}`
  - `PYTHONUNBUFFERED=1`
  - `GOOGLE_APPLICATION_CREDENTIALS=/app/gcp.json`
  - `GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT}`
- **Volumes**:
  - `./gcp.json:/app/gcp.json:ro`
- **Dependencies**: `airflow-webserver`
- **Restart Policy**: `always`
- **Profile**: `monitoring`

**Exported Metrics**:
- `prediction_latency_seconds` - API response time
- `training_runs_total` - Training job count
- `predictions_total` - Prediction count
- `training_duration_seconds` - Training time
- `model_r2_score` - Model R² score
- `model_rmse` - Model RMSE
- `drift_detected` - Drift detection flag
- `drift_share` - Percentage of drifted features
- `drifted_features_count` - Count of drifted features

---

### 5. Legacy Services (Optional - Profile: `legacy`)

#### 5.1 ClassModel API

- **Container**: `classmodel-api`
- **Image**: Built from `backend/classmodel/Dockerfile`
- **Purpose**: Legacy classification model API (kept for reference)

**Configuration**:
- **Port**: 8080:8080
- **Volumes**:
  - `./backend/classmodel:/app`
  - `./backend/classmodel/gcp.json:/tmp/gcp_creds.json:ro`
- **Command**: `uvicorn app.main:app --host 0.0.0.0 --port 8080`
- **Profile**: `legacy`

---

## GCP Services

### 1. BigQuery

- **Project**: `datascientest-460618`
- **Location**: `europe-west1`

#### Datasets

**1. bike_traffic_raw**
- **Purpose**: Raw data ingestion from Paris Open Data API
- **Updated By**: `dag_daily_fetch_data.py` (daily @ 02:00)
- **Schema**:
  - `date` (TIMESTAMP)
  - `comptage_horaire` (INTEGER)
  - `nom_compteur` (STRING)
  - `id_compteur` (STRING)
  - `coord_geographique` (STRING)
  - ... (additional sensor metadata)
- **Partitioning**: By `date` (daily)
- **Clustering**: By `id_compteur`

**2. bike_traffic_predictions**
- **Purpose**: ML model predictions storage
- **Updated By**: `dag_daily_prediction.py` (daily @ 04:00)
- **Schema**:
  - `date` (TIMESTAMP)
  - `compteur_id` (STRING)
  - `predicted_count` (FLOAT)
  - `model_run_id` (STRING)
  - `prediction_timestamp` (TIMESTAMP)
- **Partitioning**: By `date` (daily)

**3. monitoring_audit.logs**
- **Purpose**: Training logs, drift metrics, deployment decisions
- **Updated By**: `dag_monitor_and_train.py` (weekly @ Sunday 00:00)
- **Schema**:
  - `timestamp` (TIMESTAMP)
  - `event_type` (STRING) - DRIFT, VALIDATION, TRAINING, DEPLOYMENT
  - `old_champion_run_id` (STRING)
  - `new_champion_run_id` (STRING)
  - `old_r2_score` (FLOAT)
  - `new_r2_score` (FLOAT)
  - `drift_share` (FLOAT)
  - `drifted_features_count` (INTEGER)
  - `deployment_decision` (STRING) - DEPLOY, SKIP, REJECT
  - `details` (JSON)

### 2. Cloud SQL PostgreSQL

- **Instance Connection**: `${MLFLOW_INSTANCE_CONNECTION}`
- **Database**: `${MLFLOW_DB_NAME}`
- **Purpose**: MLflow tracking server backend store

**Access**:
- Via `cloud-sql-proxy` container
- Service account: `./gcp.json`
- Connection string: `postgresql://${MLFLOW_DB_USER}:${MLFLOW_DB_PASSWORD}@cloud-sql-proxy:5432/${MLFLOW_DB_NAME}`

**Tables** (managed by MLflow):
- `experiments`
- `runs`
- `metrics`
- `params`
- `tags`
- `experiment_tags`

### 3. Cloud Storage (GCS)

**Bucket**: `gs://df_traffic_cyclist1/`

#### Directory Structure

```
gs://df_traffic_cyclist1/
├── mlflow-artifacts/           # MLflow experiment artifacts
│   ├── 0/                      # Experiment ID
│   │   ├── <run_id>/
│   │   │   ├── artifacts/
│   │   │   │   ├── model/      # Serialized models
│   │   │   │   ├── plots/      # Training plots
│   │   │   │   └── data/       # Preprocessed data
│   │   │   └── metrics/
│   └── ...
├── models/
│   ├── summary.json            # Model registry (champion tracking)
│   ├── random_forest/          # Model archives
│   └── ...
├── data/
│   ├── train_baseline.csv      # 660K rows (baseline training set)
│   ├── test_baseline.csv       # 181K rows (baseline test set)
│   └── ...
└── drift_reports/              # Evidently HTML reports
    ├── drift_report_2025-01-15.html
    └── ...
```

**Service Accounts**:
- `./mlflow-ui-access.json` - MLflow bucket read/write
- `./mlflow-trainer.json` - RegModel API bucket access
- `./gcp.json` - Airflow and general GCP access

---

## External Services

### 1. Paris Open Data API

- **URL**: `https://opendata.paris.fr/api/records/1.0/search/`
- **Purpose**: Real-time bike traffic data source
- **Dataset**: `comptage-velo-donnees-compteurs`

**Usage**:
- Called by: `dag_daily_fetch_data.py`
- Frequency: Daily @ 02:00
- Fields fetched: `date`, `comptage_horaire`, `nom_compteur`, `id_compteur`, `coord_geographique`

**Rate Limits**:
- No explicit limit documented
- Current usage: ~1 request/day
- Batch size: 10,000 rows per request

### 2. Discord Webhook

- **URL**: `${DISCORD_WEBHOOK_URL}`
- **Purpose**: Real-time alerting and notifications

**Triggered By**:
1. **Grafana Alerts**:
   - Critical: R² < 0.60, drift > 70%, API error > 10%
   - Warning: R² < 0.70, drift > 50%, latency > 500ms

2. **DAG 3 Events** (`dag_monitor_and_train.py`):
   - Champion promotion (with OLD/NEW run_ids + metrics)
   - Drift detection (drift_share, drifted_features_count)
   - Performance degradation (R² drop)
   - Training failures

**Message Format**:
```json
{
  "content": "🚀 **Champion Promoted**\n\n**OLD**: run_id=abc123, R²=0.68\n**NEW**: run_id=def456, R²=0.72\n\n**Improvement**: +0.04 R² (+5.88%)"
}
```

---

## Network & Ports

### External Ports (Host → Container)

| Service | Host Port | Container Port | Protocol | Purpose |
|---------|-----------|----------------|----------|---------|
| MLflow | 5000 | 5000 | HTTP | Tracking UI |
| RegModel API | 8000 | 8000 | HTTP | Prediction API |
| Airflow Webserver | 8081 | 8080 | HTTP | DAG Management |
| Redis | 6379 | 6379 | TCP | Celery Broker |
| Flower | 5555 | 5555 | HTTP | Celery Monitoring |
| Prometheus | 9090 | 9090 | HTTP | Metrics UI |
| Pushgateway | 9091 | 9091 | HTTP | Push Metrics |
| Grafana | 3000 | 3000 | HTTP | Dashboards |
| Airflow Exporter | 9101 | 9101 | HTTP | Custom Metrics |
| ClassModel (legacy) | 8080 | 8080 | HTTP | Legacy API |

### Internal Ports (Container → Container)

| Service | Port | Access | Purpose |
|---------|------|--------|---------|
| cloud-sql-proxy | 5432 | MLflow | PostgreSQL proxy |
| postgres-airflow | 5432 | Airflow stack | Metadata DB |
| redis-airflow | 6379 | Airflow stack | Celery broker |

### Firewall Rules (GCP)

**Required for Production**:
- Allow TCP 443 from Cloud Run → Cloud SQL (automatic)
- Allow TCP 5432 from authorized networks → Cloud SQL (manual)

---

## Environment Configuration

### Required Environment Variables (.env)

```bash
# === GCP Configuration ===
GOOGLE_CLOUD_PROJECT=datascientest-460618
BQ_PROJECT=datascientest-460618
BQ_RAW_DATASET=bike_traffic_raw
BQ_PREDICT_DATASET=bike_traffic_predictions
BQ_LOCATION=europe-west1
GCS_BUCKET=df_traffic_cyclist1

# === MLflow Configuration ===
MLFLOW_INSTANCE_CONNECTION=datascientest-460618:europe-west1:mlflow-db
MLFLOW_DB_USER=mlflow
MLFLOW_DB_PASSWORD=<secure-password>
MLFLOW_DB_NAME=mlflow

# === API Configuration ===
API_URL_DEV=http://regmodel-backend:8000
API_KEY_SECRET=<api-key>

# === Model Thresholds ===
R2_CRITICAL=0.60
R2_WARNING=0.70
RMSE_THRESHOLD=50.0

# === Airflow Configuration ===
AIRFLOW_UID=50000
AIRFLOW_GID=50000
_AIRFLOW_WWW_USER_USERNAME=admin
_AIRFLOW_WWW_USER_PASSWORD=<admin-password>

# === Monitoring Configuration ===
GF_SECURITY_ADMIN_PASSWORD=<grafana-password>
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/<id>/<token>

# === Environment ===
ENV=DEV  # or PROD
```

### Service Account Files

| File | Purpose | Used By |
|------|---------|---------|
| `./gcp.json` | General GCP access | Airflow, cloud-sql-proxy, airflow-exporter |
| `./mlflow-ui-access.json` | GCS read/write for artifacts | MLflow server |
| `./mlflow-trainer.json` | GCS access for model training | RegModel API |
| `./backend/classmodel/gcp.json` | Legacy service account | ClassModel API (legacy) |

**Permissions Required**:
- **BigQuery**: `roles/bigquery.dataEditor`, `roles/bigquery.jobUser`
- **Cloud Storage**: `roles/storage.objectAdmin`
- **Cloud SQL**: `roles/cloudsql.client`

---

## Resource Management

### Memory Allocation

| Service | Limit | Reservation | Notes |
|---------|-------|-------------|-------|
| regmodel-backend | 6GB | 2GB | Increased for large CSV files (724K rows) |
| airflow-worker | (default) | (default) | Auto-scaled by Docker |
| prometheus | (default) | (default) | 15-day retention (~2GB) |
| grafana | (default) | (default) | Minimal (<500MB) |

### Disk Usage (Persistent Volumes)

| Volume | Purpose | Estimated Size |
|--------|---------|----------------|
| `postgres_airflow_data` | Airflow metadata | ~500MB (grows with DAG runs) |
| `prometheus_data` | Time-series metrics | ~2GB (15-day retention) |
| `grafana_data` | Dashboards + users | ~100MB |

### CPU Allocation

- No explicit CPU limits set
- Docker auto-allocates based on host resources
- Recommended minimum for production:
  - 4 vCPUs for Airflow stack
  - 2 vCPUs for RegModel API
  - 1 vCPU each for MLflow, Prometheus, Grafana

---

## Service Dependencies

### Startup Order (docker-compose depends_on)

```
postgres-airflow (healthy)
  ↓
redis-airflow (healthy)
  ↓
airflow-init (healthy)
  ↓
┌─────────────────┬─────────────────┬─────────────────┐
│                 │                 │                 │
airflow-webserver  airflow-scheduler  airflow-worker
│                 │                 │                 │
└─────────────────┴─────────────────┴─────────────────┘
  ↓
flower

cloud-sql-proxy
  ↓
mlflow
  ↓
regmodel-backend
  ↓
(airflow-webserver waits for mlflow + regmodel-backend)

prometheus
  ↓
grafana

airflow-webserver
  ↓
airflow-exporter
```

### Runtime Dependencies

- **DAG 1** → Paris Open Data API, BigQuery
- **DAG 2** → BigQuery, RegModel API, MLflow
- **DAG 3** → BigQuery, RegModel API, MLflow, GCS, Discord Webhook
- **RegModel API** → MLflow, GCS
- **Prometheus** → RegModel API `/metrics`, airflow-exporter `/metrics`
- **Grafana** → Prometheus, Discord Webhook

---

## Profiles & Optional Services

### Default Profile (No Flag)

**Started Services**:
- cloud-sql-proxy
- mlflow
- regmodel-backend
- postgres-airflow
- redis-airflow
- airflow-webserver
- airflow-scheduler
- airflow-worker
- airflow-init
- flower

**Command**: `docker compose up -d`

### Monitoring Profile

**Additional Services**:
- prometheus
- pushgateway
- grafana
- airflow-exporter

**Command**: `docker compose --profile monitoring up -d`

### Legacy Profile

**Additional Services**:
- classmodel-backend

**Command**: `docker compose --profile legacy up -d`

### Combined Profiles

**Command**: `docker compose --profile monitoring --profile legacy up -d`

---

## Healthchecks & Restart Policies

| Service | Healthcheck | Restart Policy |
|---------|-------------|----------------|
| postgres-airflow | `pg_isready -U airflow` (5s interval) | `always` |
| redis-airflow | `redis-cli ping` (5s interval) | `always` |
| airflow-init | Check `/tmp/airflow-init-status` | (one-time) |
| flower | `curl http://localhost:5555/` (10s interval) | `always` |
| cloud-sql-proxy | (built-in with `--health-check`) | `unless-stopped` |
| mlflow | (none) | `unless-stopped` |
| regmodel-backend | (none) | (default) |
| airflow-webserver | (none) | `always` |
| airflow-scheduler | (none) | `always` |
| airflow-worker | (none) | `always` |
| prometheus | (none) | `always` |
| grafana | (none) | `always` |
| airflow-exporter | (none) | `always` |

---

## Troubleshooting

### Common Issues

**1. Cloud SQL Proxy Connection Refused**
- Check `MLFLOW_INSTANCE_CONNECTION` format: `project:region:instance`
- Verify `./gcp.json` has `roles/cloudsql.client` permission
- Ensure Cloud SQL instance is running

**2. MLflow Cannot Write Artifacts**
- Check `./mlflow-ui-access.json` has `roles/storage.objectAdmin`
- Verify bucket `gs://df_traffic_cyclist1/` exists
- Test: `gsutil ls gs://df_traffic_cyclist1/mlflow-artifacts/`

**3. Airflow Init Failed**
- Check logs: `docker logs airflow-init`
- Verify `AIRFLOW__DATABASE__SQL_ALCHEMY_CONN` is set
- Wait for `postgres-airflow` healthcheck

**4. RegModel API Out of Memory**
- Increase memory limit in docker-compose.yaml (current: 6GB)
- Check CSV file sizes in `./data/`
- Consider using GCS-backed data loading

**5. Prometheus Scraping Failures**
- Check target status: http://localhost:9090/targets
- Verify `regmodel-backend:8000/metrics` is accessible
- Check `airflow-exporter` logs for authentication issues

---

## Maintenance

### Backup Strategy

**1. PostgreSQL Databases**:
```bash
# Airflow metadata
docker exec postgres-airflow pg_dump -U airflow airflow > airflow_backup.sql

# Cloud SQL (via gcloud)
gcloud sql backups create --instance=mlflow-db
```

**2. Prometheus Data**:
```bash
docker run --rm -v prometheus_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/prometheus_backup.tar.gz /data
```

**3. Grafana Dashboards**:
```bash
docker run --rm -v grafana_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/grafana_backup.tar.gz /data
```

### Update Strategy

**Rolling Updates** (for RegModel API):
1. Build new image: `docker compose build regmodel-backend`
2. Scale up: `docker compose up -d --scale regmodel-backend=2`
3. Verify health: `curl http://localhost:8000/health`
4. Scale down old: `docker compose up -d --scale regmodel-backend=1`

**Blue-Green Deployment** (for major changes):
1. Deploy to new environment (e.g., port 8001)
2. Run integration tests
3. Switch traffic (update Airflow `API_URL_DEV`)
4. Monitor for 24h, rollback if needed

### Cleanup

```bash
# Remove stopped containers
docker compose down

# Remove volumes (⚠️ data loss)
docker compose down -v

# Remove unused images
docker image prune -a

# Clean Prometheus old data
docker exec prometheus rm -rf /prometheus/*
```

---

## Security

### Network Isolation

- No direct internet access for containers (except proxy)
- All GCP access via service accounts
- Internal communication via Docker network

### Secrets Management

- Environment variables in `.env` (Git-ignored)
- Service account keys in `*.json` (Git-ignored, read-only mounts)
- Airflow connections stored in metadata DB (encrypted with Fernet key)

### Authentication

- **Airflow**: Basic auth (`admin` / `${_AIRFLOW_WWW_USER_PASSWORD}`)
- **MLflow**: No authentication (internal network only)
- **Grafana**: Admin password (`${GF_SECURITY_ADMIN_PASSWORD}`)
- **RegModel API**: API key (`${API_KEY_SECRET}`)

### Recommendations

1. Use secret management service (e.g., GCP Secret Manager)
2. Rotate service account keys every 90 days
3. Enable VPC Service Controls for BigQuery
4. Implement API rate limiting on FastAPI
5. Enable audit logging for all GCP services

---

## Performance Optimization

### Current Bottlenecks

1. **RegModel API Training**:
   - CPU-bound (scikit-learn)
   - Current: ~5 min for 660K rows
   - Solution: Use Cloud Run Jobs with 4 vCPUs

2. **BigQuery Queries**:
   - Cost: ~$5/TB scanned
   - Current: ~100MB/day
   - Solution: Use table partitioning and clustering

3. **Prometheus Data Retention**:
   - Current: 15 days (~2GB)
   - Solution: Archive to GCS, reduce retention to 7 days

### Scaling Strategy

**Horizontal Scaling**:
- Airflow workers: Scale with `--scale airflow-worker=N`
- RegModel API: Load balancer + multiple replicas

**Vertical Scaling**:
- Increase memory for RegModel API (current: 6GB)
- Use larger Cloud SQL instance for MLflow (current: db-f1-micro)

---

**Last Updated**: 2025-11-06
**Maintained By**: MLOps Team
