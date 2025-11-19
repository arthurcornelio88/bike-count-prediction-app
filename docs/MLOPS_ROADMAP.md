# MLOps Roadmap — Bike Traffic Prediction

**Project Goal**: Build a production-ready MLOps pipeline for bike traffic prediction with automated monitoring, drift detection, and intelligent retraining.

**Presentation Date**: November 7, 2025
**Main Branch**: `feat/mlops-monitoring`

---

## Current Status

**All Phases Complete** ✅

- ✅ ML models trained (RF, NN)
- ✅ MLflow tracking + Cloud SQL backend
- ✅ Custom model registry (`summary.json` on GCS)
- ✅ FastAPI backend with training API
- ✅ Streamlit frontend deployed
- ✅ Airflow orchestration (3 DAGs operational)
- ✅ Prometheus + Grafana monitoring
- ✅ Champion/Challenger model system
- ✅ Discord alerting for critical events

---

## Phase 0-1: Foundation ✅

**Status**: Complete

### Deliverables

- ML pipelines with custom transformers (RFPipeline, NNPipeline)
- MLflow tracking (local dev + Cloud SQL prod backend)
- Custom model registry via `summary.json` on GCS
- FastAPI backend deployed on Cloud Run
- Streamlit frontend
- Docker + docker-compose for local development
- Dev/prod environment separation

### Key Achievements

- **Models**: Random Forest (R²≈0.79), Neural Network (R²≈0.72)
- **MLflow Integration**: Centralized tracking with Cloud SQL PostgreSQL backend
- **Artifact Storage**: GCS bucket (`gs://df_traffic_cyclist1/mlflow-artifacts/`)
- **Model Registry**: Append-only `summary.json` for programmatic access

### Documentation

- [architecture.md](docs/architecture.md) — MLflow & model registry architecture
- [mlflow_cloudsql.md](docs/mlflow_cloudsql.md) — Cloud SQL setup and troubleshooting

---

## Phase 2: Tests, CI & Data Versioning ✅

**Status**: Complete

### 2.1 Data Versioning with DVC ✅

**Deliverables**:

- Temporal split: train_baseline (660K rows, 69.7%) + test_baseline (132K rows)
- DVC tracking with GCS remote storage
- Automated split script (`scripts/split_data_temporal.py`)

**Documentation**: [docs/dvc.md](docs/dvc.md)

### 2.2 Tests & CI ✅

**Deliverables**:

- **47 tests** passing across 4 suites
- **68% code coverage** (Codecov integration active)
- GitHub Actions CI with UV package manager
- Coverage artifacts (HTML reports, 30-day retention)

**Test Suites**:

- `test_pipelines.py` (13 tests) — RF, NN pipeline validation
- `test_preprocessing.py` (17 tests) — Transformer logic
- `test_api_regmodel.py` (11 tests) — FastAPI endpoints
- `test_model_registry.py` (6 tests) — Registry logic

**Documentation**:

- [docs/pytest.md](docs/pytest.md) — Test suite guide
- [docs/ci.md](docs/ci.md) — CI/CD configuration

### 2.3 Backend API `/train` + MLflow Integration ✅

**Deliverables**:

- Unified `train_model()` function supporting all model types
- FastAPI `/train` endpoint for remote training
- MLflow Cloud SQL backend for team collaboration
- Docker Compose stack (MLflow + Cloud SQL Proxy)
- DVC dataset support (baseline/current/reference)
- Automatic GCS upload + `summary.json` update

**Supported Models**: `rf` (Random Forest), `nn` (Neural Network)
**Architecture**:

```text
MLflow Server (port 5000)
├─ Backend Store: Cloud SQL PostgreSQL (metadata)
├─ Artifact Store: gs://df_traffic_cyclist1/mlflow-artifacts/
└─ Cloud SQL Proxy: Secure connection to europe-west3
```

**Documentation**:

- [docs/backend.md](docs/backend.md) — API reference
- [docs/training_strategy.md](docs/training_strategy.md) — Training workflow

---

## Phase 3: Airflow Orchestration + Production Monitoring ✅

**Status**: Complete (All 3 DAGs Operational)

### Deliverables

**Airflow DAGs**:

1. **`dag_daily_fetch_data.py`** ✅ — Daily data ingestion from Paris Open Data API
   - Deduplication logic
   - BigQuery partitioned table (`comptage_velo`)
   - Schema normalization

2. **`dag_daily_prediction.py`** ✅ — Daily ML predictions
   - Handles unknown compteurs via `handle_unknown='ignore'`
   - Stores predictions in BigQuery (`bike_traffic_predictions`)
   - Drift-aware prediction pipeline

3. **`dag_monitor_and_train.py`** ✅ — Intelligent monitoring & retraining
   - Evidently drift detection
   - Hybrid strategy (proactive + reactive triggers)
   - Champion/Challenger evaluation
   - Automated fine-tuning via `/train` endpoint

**BigQuery Datasets**:

- `bike_traffic_raw` — Raw ingested data
- `bike_traffic_predictions` — ML predictions
- `monitoring_audit` — Training logs, drift metrics, deployment decisions

**Key Features**:

- **Sliding Window Training**: Combines train_baseline (660K) + train_current (1.6K) to learn new compteurs
- **Double Evaluation**: test_baseline (detect regression) + test_current (measure improvement)
- **Hybrid Drift Strategy**: Balances cost vs. performance with proactive/reactive triggers

**Documentation**:

- [docs/dags.md](docs/dags.md) — Complete DAG reference
- [docs/training_strategy.md](docs/training_strategy.md) — Sliding window & drift management
- [docs/bigquery_setup.md](docs/bigquery_setup.md) — Data pipeline architecture

---

## Phase 4: Prometheus + Grafana Monitoring ✅

**Status**: Complete

### Deliverables

**Prometheus Integration**:

- Custom Airflow exporter (`airflow_exporter.py`) exposing MLOps metrics
- FastAPI middleware for HTTP-level metrics (requests, errors, latency)
- Pushgateway for alert testing

**Metrics Exposed**:

- **Model Performance**: `bike_model_r2_champion_current`, `bike_model_r2_challenger_baseline`, `bike_model_rmse_production`
- **Drift**: `bike_drift_detected`, `bike_drift_share`, `bike_drifted_features_count`
- **Training**: `bike_training_runs_total`, `bike_model_deployments_total`, `bike_model_improvement_delta`
- **Data Pipeline**: `bike_records_ingested_total`, `bike_predictions_generated_total`
- **API**: `fastapi_requests_total`, `fastapi_errors_total`, `fastapi_request_duration_seconds`

**Grafana Dashboards** (4 dashboards):

1. **MLOps - Overview** — System health, drift status, API metrics
2. **MLOps - Model Performance** — R² trends (champion vs challenger), RMSE, latency percentiles
3. **MLOps - Drift Monitoring** — Drift evolution, drifted features count, R² vs drift correlation
4. **MLOps - Training & Deployment** — Training success rate, deployment decisions, model improvement

**Alerting Rules**:

- **Critical**: R² < 0.60, drift > 70% + R² declining, API error rate > 10%
- **Warning**: R² < 0.70, drift > 50%, training failures
- **Info**: Successful deployments, champion promotions

**Documentation**:

- [docs/monitoring/01_architecture.md](docs/monitoring/01_architecture.md) — Monitoring overview
- [docs/monitoring/02_alerting.md](docs/monitoring/02_alerting.md) — Alert configuration
- [docs/monitoring/03_metrics_reference.md](docs/monitoring/03_metrics_reference.md) — Complete metrics catalog
- [docs/monitoring/04_dashboards_explained.md](docs/monitoring/04_dashboards_explained.md) — Dashboard guide
- [docs/monitoring/TESTING_ALERTS.md](docs/monitoring/TESTING_ALERTS.md) — Testing guide

---

## Phase 5: Champion Model Tracking ✅

**Status**: Complete

### Deliverables

**Champion/Challenger System**:

- Explicit champion designation via `is_champion` flag in `summary.json`
- `/promote_champion` API endpoint for model promotion
- Priority loading: Champion models loaded first regardless of metrics
- Metadata caching in FastAPI (run_id, is_champion, R², RMSE)

**Double Evaluation Strategy**:

- **test_baseline** (181K samples, fixed reference) — Detect regression (R² >= 0.60 threshold)
- **test_current** (20% of fresh data) — Measure improvement vs. champion

**Deployment Decision Logic**:

```text
IF r2_baseline < 0.60:
   → REJECT (baseline regression)
ELIF r2_current > champion_r2 + 0.02:
   → DEPLOY (improved on current distribution)
ELSE:
   → SKIP (no significant improvement)
```

**Discord Alerting**:

- Champion promotions (with R² metrics, improvement delta)
- Training failures (with error details, dag_run_id)
- Critical performance degradation
- Infrastructure alerts (service down, high API error rate)

**Documentation**:

- [docs/monitoring/AUDIT_DOUBLE_EVAL.md](docs/monitoring/AUDIT_DOUBLE_EVAL.md) — Historical audit
- [docs/monitoring/IMPLEMENTATION_DOUBLE_EVAL.md](docs/monitoring/IMPLEMENTATION_DOUBLE_EVAL.md) — Implementation guide

---

## Key Technical Decisions

### Data Strategy

**Final Approach**: Unified baseline from `current_api_data.csv` (905K records, 2024-09-01 → 2025-10-10)

**Rationale**: All data sources (reference, current, API) are from same origin with perfect correlation (r=1.0)

### Drift Management

**Hybrid Strategy**: Proactive (preventive) + Reactive (corrective) triggers

**Decision Matrix**:

| R² Score | Drift Share | Decision | Rationale |
|----------|-------------|----------|-----------|
| < 0.65 | Any | **RETRAIN (Reactive)** | Critical performance issue |
| 0.65-0.70 | ≥ 50% | **RETRAIN (Proactive)** | High drift + declining metrics |
| 0.65-0.70 | 30-50% | **WAIT** | Moderate drift, metrics acceptable |
| ≥ 0.70 | ≥ 30% | **WAIT** | Model handles drift well |
| ≥ 0.70 | < 30% | **ALL GOOD** | Continue monitoring |

**Thresholds**:

- R2_CRITICAL = 0.65 (reactive trigger)
- R2_WARNING = 0.70 (proactive trigger)
- DRIFT_CRITICAL = 0.5 (50%+ drift share)
- DRIFT_WARNING = 0.3 (30%+ drift share)

### Sliding Window Training

**Problem Solved**: Model wasn't learning from new compteurs (bike counters)

**Solution**: Concatenate train_baseline (660K samples) + train_current (80% of fresh data) for combined training

**Benefits**:

- New compteurs integrated into model weights
- Temporal pattern adaptation
- Prevents catastrophic forgetting
- Balances stability vs. flexibility

---

## Architecture Overview

### Services

```text
┌─────────────────────────────────────────────────────────┐
│                   MLOps Pipeline                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Data Ingestion (Airflow DAG 1)                         │
│  ├─ Paris Open Data API → BigQuery                      │
│  └─ Deduplication + Schema normalization                │
│                                                          │
│  Predictions (Airflow DAG 2)                            │
│  ├─ Fetch data → FastAPI /predict → BigQuery            │
│  └─ Handle unknown compteurs gracefully                 │
│                                                          │
│  Monitoring & Training (Airflow DAG 3)                  │
│  ├─ Evidently drift detection                           │
│  ├─ Validate champion performance                       │
│  ├─ Fine-tune via /train endpoint (if needed)           │
│  ├─ Double evaluation (baseline + current)              │
│  └─ Promote champion or keep current                    │
│                                                          │
│  Model Tracking                                          │
│  ├─ MLflow (Cloud SQL backend)                          │
│  ├─ GCS artifacts storage                               │
│  └─ summary.json registry                               │
│                                                          │
│  Monitoring                                              │
│  ├─ Prometheus (airflow-exporter + FastAPI middleware)  │
│  ├─ Grafana (4 dashboards)                              │
│  ├─ Discord alerting (critical events)                  │
│  └─ BigQuery audit logs                                 │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Docker Compose Services

- `cloud-sql-proxy` — Cloud SQL connection proxy
- `mlflow` — MLflow tracking server (port 5000)
- `regmodel-backend` — FastAPI API (port 8000)
- `airflow-webserver` — Airflow UI (port 8081)
- `airflow-scheduler` — DAG scheduler
- `airflow-worker` — Celery worker
- `airflow-init` — Database initialization
- `flower` — Celery monitoring (port 5555)
- `postgres-airflow` — Airflow metadata database
- `redis-airflow` — Celery broker
- `prometheus` — Metrics collection (port 9090) [monitoring profile]
- `pushgateway` — Pushgateway for testing (port 9091) [monitoring profile]
- `grafana` — Dashboards (port 3000) [monitoring profile]
- `airflow-exporter` — Custom MLOps exporter (port 9101) [monitoring profile]

### Key Endpoints

**FastAPI (port 8000)**:

- `POST /train` — Train model with sliding window
- `POST /predict` — Generate predictions (returns champion metadata)
- `POST /evaluate` — Evaluate champion on test_baseline
- `POST /drift` — Detect data drift with Evidently
- `POST /promote_champion` — Promote model to champion status

**MLflow (port 5000)**:

- UI for experiment tracking, model comparison, artifact inspection

**Airflow (port 8081)**:

- UI for DAG monitoring, task logs, manual triggers

**Grafana (port 3000)**:

- 4 dashboards for real-time MLOps monitoring

**Prometheus (port 9090)**:

- Metrics query interface, target health

---

## Quick Start

### Local Development

```bash
# Start all services
docker compose up -d

# Start with monitoring
docker compose --profile monitoring up -d

# View logs
docker compose logs -f regmodel-backend
docker compose logs -f airflow-scheduler

# Trigger training DAG
docker exec airflow-webserver airflow dags trigger monitor_and_fine_tune \
  --conf '{"force_fine_tune": true, "test_mode": false}'

# Access services
open http://localhost:5000  # MLflow
open http://localhost:8081  # Airflow
open http://localhost:3000  # Grafana (admin / see .env)
open http://localhost:9090  # Prometheus
```

### Training Commands

```bash
# Train champion locally (full data)
python backend/regmodel/app/train.py \
  --model-type rf \
  --data-source baseline \
  --env dev

# Quick test (1K samples)
python backend/regmodel/app/train.py \
  --model-type rf \
  --data-source baseline \
  --model-test \
  --env dev

# Train via API
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

## Documentation Index

### Core Documentation

- [architecture.md](docs/architecture.md) — MLflow & registry architecture
- [training_strategy.md](docs/training_strategy.md) — Training workflow (sliding window, drift management, double evaluation)
- [dags.md](docs/dags.md) — Airflow DAG reference
- [backend.md](docs/backend.md) — FastAPI API documentation

### Setup & Configuration

- [mlflow_cloudsql.md](docs/mlflow_cloudsql.md) — MLflow Cloud SQL setup
- [bigquery_setup.md](docs/bigquery_setup.md) — BigQuery pipeline
- [secrets.md](docs/secrets.md) — GCS credentials & Secret Manager
- [dvc.md](docs/dvc.md) — Data versioning

### Testing & CI

- [pytest.md](docs/pytest.md) — Test suite guide
- [ci.md](docs/ci.md) — GitHub Actions CI/CD

### Monitoring

- [docs/monitoring/01_architecture.md](docs/monitoring/01_architecture.md) — Monitoring overview
- [docs/monitoring/02_alerting.md](docs/monitoring/02_alerting.md) — Alert configuration
- [docs/monitoring/03_metrics_reference.md](docs/monitoring/03_metrics_reference.md) — Metrics catalog
- [docs/monitoring/04_dashboards_explained.md](docs/monitoring/04_dashboards_explained.md) — Dashboard guide
- [docs/monitoring/TESTING_ALERTS.md](docs/monitoring/TESTING_ALERTS.md) — Testing alerts
- [docs/monitoring/AUDIT_DOUBLE_EVAL.md](docs/monitoring/AUDIT_DOUBLE_EVAL.md) — Double evaluation audit
- [docs/monitoring/IMPLEMENTATION_DOUBLE_EVAL.md](docs/monitoring/IMPLEMENTATION_DOUBLE_EVAL.md) — Implementation guide

---

## Project Statistics

- **Code Coverage**: 68% (Codecov integrated)
- **Tests**: 47 passing across 4 suites
- **Docker Services**: 15 containers (8 core + 7 optional monitoring)
- **Airflow DAGs**: 3 operational (ingestion, prediction, monitoring)
- **Grafana Dashboards**: 4 dashboards, 50+ panels
- **Prometheus Metrics**: 15+ custom MLOps metrics
- **Documentation**: 20+ markdown files

---

## Future Improvements

### Short Term

- [ ] Implement rolling window training (last 6 months only)
- [ ] Add SMOTE/class weighting for rare compteurs
- [ ] Make train/test split ratio configurable
- [ ] Add cost-aware retraining decision logic

### Long Term

- [ ] Better unknown compteur handling (target encoding, geographic clustering)
- [ ] Adaptive thresholds learned from historical data
- [ ] Multi-model ensemble predictions
- [ ] A/B testing framework for model deployment

---

**Maintainer**: Arthur Cornélio & Bruno (^^)
**Last Updated**: 2025-11-06
**Version**: 5.0 (Champion Tracking Complete)
