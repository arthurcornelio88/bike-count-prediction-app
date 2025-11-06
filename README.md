# 🚲 Bike Traffic Prediction - MLOps Production Pipeline

[![CI Tests](https://github.com/arthurcornelio88/ds_traffic_cyclist1/actions/workflows/ci.yml/badge.svg)](https://github.com/arthurcornelio88/ds_traffic_cyclist1/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/arthurcornelio88/bike-count-prediction-app/branch/master/graph/badge.svg?token=O9XOGPGO6G)](https://codecov.io/gh/arthurcornelio88/bike-count-prediction-app)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![type: mypy](https://img.shields.io/badge/type-mypy-blue.svg)](https://github.com/python/mypy)

**Version 2.0.0** - Production-ready MLOps pipeline for predicting hourly bike traffic in Paris. Features automated data ingestion from Paris Open Data API, intelligent drift detection with sliding window training, champion/challenger model system with double evaluation, real-time monitoring via Prometheus + Grafana, and Discord alerting for critical events. Orchestrated with Airflow, tracked with MLflow (Cloud SQL backend), and deployed with FastAPI. All infrastructure runs locally via Docker Compose with 15 services. Production Kubernetes deployment under construction.

---

## 🚀 Quick Start

### Local Development (All Services)

```bash
# Start full MLOps stack (MLflow, Airflow, FastAPI, Monitoring)
./scripts/start-all.sh --with-monitoring

# Access services
open http://localhost:5000   # MLflow tracking
open http://localhost:8081   # Airflow (admin / see .env)
open http://localhost:8000   # FastAPI API docs
open http://localhost:3000   # Grafana (admin / see .env)
open http://localhost:9090   # Prometheus
```

### Trigger DAGs

```bash
# DAG 1: Ingest data from Paris Open Data API
docker exec airflow-webserver airflow dags trigger fetch_comptage_daily

# DAG 2: Generate predictions
docker exec airflow-webserver airflow dags trigger daily_prediction

# DAG 3: Monitor & train (with force flag)
docker exec airflow-webserver airflow dags trigger monitor_and_fine_tune \
  --conf '{"force_fine_tune": true, "test_mode": false}'
```

---

## 🎯 Features

### MLOps Core
- ✅ **Champion/Challenger System** - Explicit model promotion with double evaluation
- ✅ **Sliding Window Training** - Learns from fresh data (660K baseline + 1.6K current)
- ✅ **Drift Detection** - Evidently-based monitoring with hybrid retraining strategy
- ✅ **Real-time Monitoring** - Prometheus metrics + 4 Grafana dashboards
- ✅ **Discord Alerting** - Critical events, training failures, champion promotions

### Data Pipeline
- ✅ **Automated Ingestion** - Daily fetch from Paris Open Data API → BigQuery
- ✅ **Prediction Pipeline** - Daily ML predictions on last 7 days
- ✅ **Audit Logs** - All training runs, drift metrics, deployment decisions tracked

### Model Registry
- ✅ **MLflow Tracking** - Cloud SQL PostgreSQL backend + GCS artifacts
- ✅ **Custom Registry** - `summary.json` for fast model loading
- ✅ **Priority Loading** - Champion models loaded first regardless of metrics

### Quality Assurance
- ✅ **68% Code Coverage** - 47 tests across 4 suites
- ✅ **CI/CD Pipeline** - GitHub Actions with Codecov integration
- ✅ **Pre-commit Hooks** - Ruff, MyPy, Bandit, YAML validation

---

## 📊 Architecture Overview

### 3-Layer MLOps Stack

```
┌─────────────────────────────────────────────────────────┐
│  LAYER 3: MLOps (Training & Monitoring)                 │
│  • DAG 3: Monitor & Train (weekly)                      │
│  • Sliding window training (660K + 1.6K samples)        │
│  • Double evaluation (test_baseline + test_current)     │
│  • Champion promotion + Discord alerts                  │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│  LAYER 2: DataOps (Ingestion & Predictions)             │
│  • DAG 1: Daily data ingestion → BigQuery               │
│  • DAG 2: Daily predictions (last 7 days)               │
│  • 3 BigQuery datasets (raw, predictions, audit)        │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│  LAYER 1: InfraOps (Services & Storage)                 │
│  • 15 Docker services (MLflow, Airflow, FastAPI)        │
│  • GCP: BigQuery, Cloud SQL, GCS                        │
│  • Monitoring: Prometheus, Grafana, airflow-exporter    │
└─────────────────────────────────────────────────────────┘
```

### Docker Services (15 containers)

**Core Stack**:
- `mlflow` (port 5000) - Tracking server with Cloud SQL backend
- `regmodel-backend` (port 8000) - FastAPI with 5 endpoints
- `airflow-webserver` (port 8081) - DAG management UI
- `airflow-scheduler` - Task scheduling
- `airflow-worker` - Celery task execution
- `cloud-sql-proxy` - Secure Cloud SQL connection

**Monitoring Stack** (`--profile monitoring`):
- `prometheus` (port 9090) - Metrics collection
- `grafana` (port 3000) - 4 dashboards (overview, performance, drift, training)
- `airflow-exporter` (port 9101) - Custom MLOps metrics

**Supporting Services**:
- `postgres-airflow` - Airflow metadata DB
- `redis-airflow` - Celery broker
- `flower` (port 5555) - Celery monitoring

---

## 📁 Project Structure

```
├── backend/regmodel/app/       # FastAPI backend
│   ├── fastapi_app.py          # API endpoints (/train, /predict, /promote_champion)
│   ├── train.py                # Training logic (sliding window)
│   ├── model_registry_summary.py  # Custom registry (summary.json)
│   └── middleware/             # Prometheus metrics middleware
├── dags/                       # Airflow DAGs
│   ├── dag_daily_fetch_data.py      # Data ingestion (daily @ 02:00)
│   ├── dag_daily_prediction.py      # Predictions (daily @ 04:00)
│   ├── dag_monitor_and_train.py     # Monitor & train (weekly @ Sunday)
│   └── utils/discord_alerts.py      # Discord webhook integration
├── monitoring/                 # Monitoring configuration
│   ├── grafana/provisioning/   # 4 dashboards + alerting rules
│   ├── prometheus.yml          # Scrape config (3 targets)
│   └── custom_exporters/       # airflow_exporter.py (MLOps metrics)
├── scripts/                    # Utility scripts
│   ├── start-all.sh            # Start all services (with/without monitoring)
│   ├── restart-airflow.sh      # Reset Airflow password
│   └── reset-airflow-password.sh
├── data/                       # Training data (DVC tracked)
│   ├── train_baseline.csv      # 660K samples (69.7%)
│   └── test_baseline.csv       # 181K samples (30.3%)
├── docs/                       # Documentation (20+ files)
│   ├── MLOPS_ROADMAP.md        # Complete project roadmap
│   ├── training_strategy.md    # Sliding window + drift management
│   ├── ARCHITECTURE_DIAGRAM_GUIDE.md  # Excalidraw guide (3 layers)
│   ├── DEMO_SCRIPT.md          # 2-minute video demo script
│   └── monitoring/             # Monitoring docs (4 files)
├── tests/                      # Test suite (47 tests, 68% coverage)
│   ├── test_pipelines.py       # RF, NN pipeline validation
│   ├── test_preprocessing.py   # Transformer logic
│   ├── test_api_regmodel.py    # FastAPI endpoints
│   └── test_model_registry.py  # Registry logic
├── docker-compose.yaml         # 15 services (6GB memory for training)
└── .github/workflows/ci.yml    # CI/CD pipeline
```

---

## 🔧 Setup & Installation

### Prerequisites

- Docker & Docker Compose
- GCP credentials (service account JSON)
- Python 3.11+ (for local development)

### 1. Environment Configuration

Create `.env` file at project root (see [docs/secrets.md](docs/secrets.md) for production setup):

```bash
# ========================================
# Environment & GCP
# ========================================
ENV=DEV                                    # DEV or PROD
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=gcp.json

# ========================================
# BigQuery Configuration
# ========================================
BQ_PROJECT=your-project-id
BQ_RAW_DATASET=bike_traffic_raw          # Raw data from API
BQ_PREDICT_DATASET=bike_traffic_predictions  # Model predictions
BQ_LOCATION=europe-west1

# ========================================
# Google Cloud Storage
# ========================================
GCS_BUCKET=your-bucket-name              # MLflow artifacts + model registry

# ========================================
# API Configuration
# ========================================
API_URL_DEV=http://regmodel-api:8000     # Internal Docker network
API_KEY_SECRET=dev-key-unsafe            # Change for production!

# ========================================
# Model Performance Thresholds (v2.0.0)
# ========================================
R2_CRITICAL=0.45      # Below this → immediate retraining
R2_WARNING=0.55       # Below this + drift → proactive retraining
RMSE_THRESHOLD=90.0   # Above this → immediate retraining

# Note: If you change these thresholds, also update:
#   - monitoring/grafana/provisioning/dashboards/overview.json (lines 177, 181)
#   - monitoring/grafana/provisioning/dashboards/model_performance.json
#   - monitoring/grafana/provisioning/dashboards/training_deployment.json

# ========================================
# Discord Alerting (Optional)
# ========================================
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

# ========================================
# Grafana
# ========================================
GF_SECURITY_ADMIN_PASSWORD=your-strong-password

# ========================================
# MLflow Backend (Cloud SQL)
# ========================================
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_DB_USER=mlflow_user
MLFLOW_DB_PASSWORD=your-db-password
MLFLOW_DB_NAME=mlflow
MLFLOW_INSTANCE_CONNECTION=project-id:region:instance-name

# ========================================
# Airflow Configuration
# ========================================
_AIRFLOW_WWW_USER_USERNAME=admin
_AIRFLOW_WWW_USER_PASSWORD=admin
AIRFLOW_UID=50000  # Match host user for volume permissions
AIRFLOW_GID=50000
```

**Security Notes:**
- ⚠️ `.env` contains secrets - NEVER commit to Git (already in `.gitignore`)
- 🔐 For production: Use GCP Secret Manager (see [docs/secrets.md](docs/secrets.md))
- 📝 Legacy file `.env.airflow` can be deleted (all vars moved to `.env`)

### 2. Clone & Install

```bash
git clone https://github.com/arthurcornelio88/ds_traffic_cyclist1.git
cd ds_traffic_cyclist1

# Install dependencies (local dev)
uv init
uv venv
uv sync
source .venv/bin/activate
```

### 2. Configure GCP Credentials

```bash
# Place service account JSON in project root
# File: mlflow-trainer.json (for training + model upload)
```

**Required GCP Services**:
- BigQuery (3 datasets: raw, predictions, audit)
- Cloud SQL PostgreSQL (MLflow metadata)
- GCS bucket: `gs://df_traffic_cyclist1/`

See [docs/secrets.md](docs/secrets.md) for detailed setup.

### 3. Start Services

```bash
# Option 1: Core services only (MLflow, Airflow, FastAPI)
./scripts/start-all.sh

# Option 2: With monitoring (Prometheus + Grafana)
./scripts/start-all.sh --with-monitoring

# Check logs
docker compose logs -f regmodel-backend
docker compose logs -f airflow-scheduler
```

---

## 🧪 Development

### Run Tests

```bash
# All tests with coverage
uv run pytest tests/ -v --cov

# Specific test suite
uv run pytest tests/test_api_regmodel.py -v

# Generate HTML coverage report
uv run pytest tests/ --cov --cov-report=html
open htmlcov/index.html
```

### Pre-commit Hooks

```bash
# Install hooks
uv run pre-commit install

# Run manually
uv run pre-commit run --all-files
```

### Train Champion Model Locally

```bash
# Quick test (1K samples)
python backend/regmodel/app/train.py \
  --model-type rf \
  --data-source baseline \
  --model-test \
  --env dev

# Full production training (660K samples)
python backend/regmodel/app/train.py \
  --model-type rf \
  --data-source baseline \
  --env dev
```

---

## 📡 API Endpoints

### FastAPI (port 8000)

**Base URL**: `http://localhost:8000`

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/train` | POST | Train model with sliding window |
| `/predict` | POST | Generate predictions (returns champion metadata) |
| `/evaluate` | POST | Evaluate champion on test_baseline |
| `/drift` | POST | Detect data drift (Evidently) |
| `/promote_champion` | POST | Promote model to champion status |
| `/metrics` | GET | Prometheus metrics (scraped every 15s) |
| `/docs` | GET | Interactive API documentation |

**Example: Train via API**

```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "rf",
    "data_source": "baseline",
    "test_mode": false,
    "env": "dev"
  }'
```

---

## 📊 Monitoring & Dashboards

### Grafana Dashboards (4 total)

Access: `http://localhost:3000` (admin / see `.env`)

1. **MLOps - Overview**
   - Drift status (50% detected)
   - Champion R² (0.78)
   - API request rate & error rate
   - Services health

2. **MLOps - Model Performance**
   - R² trends (champion vs challenger)
   - RMSE: 32.5
   - API latency percentiles (P50/P95/P99)

3. **MLOps - Drift Monitoring**
   - Drift evolution over time
   - Drifted features count
   - R² vs drift correlation

4. **MLOps - Training & Deployment**
   - Training success rate (100%)
   - Deployment decisions (deploy/skip/reject)
   - Model improvement delta

### Prometheus Metrics (15+ custom)

**Key Metrics**:
- `bike_model_r2_champion_current` - Champion R² on recent data
- `bike_drift_detected` - Binary drift flag (0/1)
- `bike_training_runs_total` - Training runs counter
- `bike_model_deployments_total` - Deployment decisions
- `fastapi_requests_total` - API request rate
- `fastapi_request_duration_seconds` - API latency

See [docs/monitoring/03_metrics_reference.md](docs/monitoring/03_metrics_reference.md) for full catalog.

---

## 🎬 Demo Video (2 minutes)

See [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) for video presentation guide:

1. **Infrastructure startup** (0:00-0:15) - Start all services
2. **Data pipeline** (0:15-0:45) - DAG 1 & 2, Discord alerts, BigQuery
3. **MLOps pipeline** (0:45-1:30) - DAG 3 training, champion promotion
4. **Grafana dashboards** (1:30-2:00) - 4 dashboards overview

---

## 📚 Documentation

### Core Documentation
- [MLOPS_ROADMAP.md](MLOPS_ROADMAP.md) - Complete project roadmap (5 phases)
- [docs/training_strategy.md](docs/training_strategy.md) - Sliding window + drift management
- [docs/architecture.md](docs/architecture.md) - MLflow & model registry
- [docs/dags.md](docs/dags.md) - Airflow DAG reference (3 DAGs)

### Setup Guides
- [docs/mlflow_cloudsql.md](docs/mlflow_cloudsql.md) - MLflow Cloud SQL setup
- [docs/bigquery_setup.md](docs/bigquery_setup.md) - BigQuery pipeline
- [docs/secrets.md](docs/secrets.md) - GCS credentials & Secret Manager
- [docs/dvc.md](docs/dvc.md) - Data versioning

### Monitoring
- [docs/monitoring/01_architecture.md](docs/monitoring/01_architecture.md) - Monitoring overview
- [docs/monitoring/02_alerting.md](docs/monitoring/02_alerting.md) - Alert configuration
- [docs/monitoring/03_metrics_reference.md](docs/monitoring/03_metrics_reference.md) - Metrics catalog
- [docs/monitoring/04_dashboards_explained.md](docs/monitoring/04_dashboards_explained.md) - Dashboard guide

### Presentation Materials
- [docs/ARCHITECTURE_DIAGRAM_GUIDE.md](docs/ARCHITECTURE_DIAGRAM_GUIDE.md) - Excalidraw guide (3 layers)
- [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) - 2-minute video demo script

---

## 🚧 Version History

### Version 2.0.0 (Current - November 2025)

**Status**: Local development ready, production Kubernetes deployment under construction

**Major Features**:
- ✅ Production MLOps pipeline (Airflow, MLflow, FastAPI)
- ✅ Champion/Challenger system with double evaluation
- ✅ Sliding window training (660K + 1.6K samples)
- ✅ Real-time monitoring (Prometheus + Grafana)
- ✅ Discord alerting
- 🚧 Kubernetes deployment (under construction)
- 🚧 Production GCP deployment (under construction)

### Version 1.0.0 (Legacy)

**Features**:
- Streamlit frontend for manual predictions
- Basic MLflow tracking (local only)
- Single model registry (`summary.json`)
- No automated orchestration

**Note**: V1 frontend (Streamlit) is deprecated in V2. Focus shifted to automated MLOps pipeline.

---

## 🔑 Key Technical Decisions

### Data Strategy
- **Unified baseline**: 905K records from `current_api_data.csv` (2024-09-01 → 2025-10-10)
- **Temporal split**: 660K train (69.7%) + 181K test (30.3%)
- **DVC tracking**: Data versioned with GCS remote storage

### Drift Management
- **Hybrid strategy**: Proactive (preventive) + Reactive (corrective) triggers
- **Thresholds**: R² < 0.65 (critical), drift ≥ 50% (proactive)
- **Decision matrix**: 5 priority levels (force, reactive, proactive, wait, all good)

### Training Strategy
- **Sliding window**: Concatenate train_baseline (660K) + train_current (1.6K)
- **Double evaluation**: test_baseline (regression check) + test_current (improvement check)
- **Deployment logic**: REJECT (R² < 0.60) / SKIP (no improvement) / DEPLOY (R² gain > 0.02)

---

## 🐛 Troubleshooting

### Airflow password issues
```bash
./scripts/reset-airflow-password.sh
# Default: admin / admin
```

### Container memory issues
```bash
# Check memory usage
docker stats

# Restart with clean slate
docker compose down -v
./scripts/start-all.sh --with-monitoring
```

### MLflow connection issues
See [docs/mlflow_cloudsql.md](docs/mlflow_cloudsql.md) for Cloud SQL troubleshooting.

### Training failures
Check Discord alerts or Airflow logs:
```bash
docker compose logs -f airflow-scheduler
```

---

## 👥 Contributors

Built with ❤️ by:

- [Arthur Cornélio](https://github.com/arthurcornelio88)
- [Ibtihel Nemri](https://github.com/ibtihelnemri)
- [Bruno Happi](https://github.com/brunoCo-de)
- [Laurène Attia](https://github.com/laureneatt)

---

## 📄 License

This project is part of a DataScientest MLOps training program.

---

**Last Updated**: November 2025
**Version**: 2.0.0
**Status**: Local ready, Kubernetes deployment under construction
