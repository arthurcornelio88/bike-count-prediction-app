# Architecture Diagram Guide (Excalidraw)

**Purpose**: Simplified guide to draw the MLOps architecture in 3 layers (infraops, dataops, mlops)

**Approach**: Layered diagram with clear separation of concerns

---

## Overview: 3-Layer Architecture

```
┌───────────────────────────────────────────────────┐
│         LAYER 3: MLOps (Top)                      │
│  Training, Monitoring, Model Management           │
└───────────────────────────────────────────────────┘
┌───────────────────────────────────────────────────┐
│         LAYER 2: DataOps (Middle)                 │
│  Data Ingestion, Storage, Predictions             │
└───────────────────────────────────────────────────┘
┌───────────────────────────────────────────────────┐
│         LAYER 1: InfraOps (Bottom)                │
│  Docker Services, Databases, Cloud Resources      │
└───────────────────────────────────────────────────┘
```

---

## LAYER 1: InfraOps (Infrastructure)

**Purpose**: Foundation services that everything runs on

### Cloud Services (GCP)

- **BigQuery** (3 datasets: raw, predictions, audit)
- **Cloud SQL PostgreSQL** (MLflow metadata)
- **GCS Bucket** `gs://df_traffic_cyclist1/` (artifacts, models, data)

### Docker Services - Core Stack

**MLflow Stack**:
- `cloud-sql-proxy` → port 5432 (secure connection to Cloud SQL)
- `mlflow` → port 5000 (tracking server)

**FastAPI Backend**:
- `regmodel-backend` → port 8000 (6GB memory, API endpoints)

**Airflow Stack**:
- `postgres-airflow` → port 5432 (Airflow metadata DB)
- `redis-airflow` → port 6379 (Celery broker)
- `airflow-webserver` → port 8081 (DAG management UI)
- `airflow-scheduler` (DAG scheduling)
- `airflow-worker` (task execution)

### Docker Services - Monitoring Stack (--profile monitoring)

- `prometheus` → port 9090 (metrics collection)
- `grafana` → port 3000 (4 dashboards)
- `airflow-exporter` → port 9101 (custom MLOps metrics)

### External Services

- **Paris Open Data API** (bike traffic source)
- **Discord Webhook** (alerting)

---

## LAYER 2: DataOps (Data Pipeline)

**Purpose**: Data flow from ingestion to storage

### DAG 1: Data Ingestion (`dag_daily_fetch_data.py`)

**Schedule**: Daily @ 02:00

**Flow**:
```
Paris Open Data API
  → fetch_data
  → deduplicate + normalize
  → BigQuery (bike_traffic_raw)
```

### DAG 2: Predictions (`dag_daily_prediction.py`)

**Schedule**: Daily @ 04:00

**Flow**:
```
BigQuery (last 7 days)
  → prepare_data
  → POST /predict (FastAPI)
  → store_predictions
  → BigQuery (bike_traffic_predictions)
```

### BigQuery Datasets

1. **bike_traffic_raw** - Raw sensor data (ingestion)
2. **bike_traffic_predictions** - ML predictions (daily)
3. **monitoring_audit** - Training logs, drift metrics, deployment decisions

### GCS Data Storage

- `/data/train_baseline.csv` (660K rows)
- `/data/test_baseline.csv` (181K rows)
- `/drift_reports/` (Evidently HTML reports)

---

## LAYER 3: MLOps (Model Lifecycle)

**Purpose**: Model training, evaluation, monitoring, deployment

### DAG 3: Monitoring & Training (`dag_monitor_and_train.py`)

**Schedule**: Weekly @ Sunday 00:00

**Simplified Flow**:
```
1. Validate Champion (R² check)
   ↓
2. Detect Drift (Evidently)
   ↓
3. Decision: WAIT or RETRAIN?
   ↓ (if RETRAIN)
4. Fetch last 30 days from BigQuery
   ↓
5. POST /train (sliding window: baseline + current)
   ↓
6. Double Evaluation (test_baseline + test_current)
   ↓
7. Deployment Decision (REJECT/SKIP/DEPLOY)
   ↓ (if DEPLOY)
8. POST /promote_champion
   ↓
9. Log audit → BigQuery + Discord alert
```

**Key Decision Logic**:
- R² < 0.65 OR drift ≥ 50% → RETRAIN (reactive)
- drift ≥ 50% + R² < 0.70 → RETRAIN (proactive)
- Otherwise → WAIT

### Training Pipeline (Sliding Window)

```
train_baseline.csv (660K, GCS)
  +
train_current (1.6K, BigQuery last 30 days)
  ↓
Train RandomForest (661.6K combined)
  ↓
Double Evaluation:
  ├─ test_baseline (181K) → detect regression (R² ≥ 0.60)
  └─ test_current (400) → measure improvement (R² > champion + 0.02)
  ↓
MLflow tracking → Cloud SQL + GCS artifacts
  ↓
summary.json updated (champion flag)
```

### MLflow Tracking Flow

```
train.py / POST /train
  ↓
MLflow Server (port 5000)
  ├─ Metadata → Cloud SQL (via proxy)
  └─ Artifacts → GCS (mlflow-artifacts/)
```

### Model Registry (summary.json on GCS)

- Append-only JSON with all trained models
- Champion designation via `is_champion=True`
- Priority loading: Champion > best metric

### Monitoring & Alerting

**Prometheus Scraping**:
```
airflow-exporter:9101 → Prometheus
regmodel-backend:8000/metrics → Prometheus
  ↓
Prometheus (store time series)
  ↓
Grafana (4 dashboards)
  ↓ (if alert triggered)
Discord Webhook
```

**4 Grafana Dashboards**:
1. **Overview** - System health, drift status, API metrics
2. **Model Performance** - R² trends (champion vs challenger), RMSE
3. **Drift Monitoring** - Drift evolution, features count
4. **Training & Deployment** - Success rate, deployment decisions

**Alerting Rules**:
- **Critical**: R² < 0.60, drift > 70%, API error > 10%, training failure
- **Warning**: R² < 0.70, drift > 50%, latency > 500ms
- **Info**: Successful deployment, champion promotion

---

## Key Endpoints (FastAPI :8000)

**Most Important Endpoints to Draw**:

1. **POST /train** - Train model with sliding window
   - Input: model_type, data_source, current_data, test_mode
   - Output: run_id, metrics, deployment_decision

2. **POST /predict** - Generate predictions
   - Input: records[], model_type
   - Output: predictions[], champion_metadata

3. **POST /promote_champion** - Deploy new champion
   - Input: model_type, run_id, env
   - Output: status, message

4. **GET /metrics** - Prometheus metrics (scraped every 15s)

---

## Connection Flows (Arrow Types)

### Data Flow (Solid Arrows)
- Paris API → DAG 1 → BigQuery
- BigQuery → DAG 2 → FastAPI /predict → BigQuery
- BigQuery → DAG 3 → FastAPI /train → MLflow → GCS

### Monitoring Flow (Dashed Arrows)
- FastAPI /metrics → Prometheus
- airflow-exporter → Prometheus
- Prometheus → Grafana

### Alerting Flow (Dotted Arrows)
- Grafana alerts → Discord
- DAG 3 (on champion promotion) → Discord

---

## Visual Organization Tips

### Component Placement

**Layer 1 (Bottom - InfraOps)**:
- Left: MLflow stack (cloud-sql-proxy, mlflow)
- Center: FastAPI (regmodel-backend)
- Right: Airflow stack (postgres, redis, webserver, scheduler, worker)
- Far right: Monitoring stack (prometheus, grafana, airflow-exporter)
- Top cloud: GCP services (BigQuery, Cloud SQL, GCS)
- Edges: External (Paris API, Discord)

**Layer 2 (Middle - DataOps)**:
- DAG 1 (ingestion) on left
- DAG 2 (predictions) in center
- BigQuery datasets as destinations

**Layer 3 (Top - MLOps)**:
- DAG 3 (monitor & train) as orchestrator
- Training pipeline flow (baseline + current → train → evaluate)
- Monitoring stack connections

### Color Coding

- **Blue**: Data storage (BigQuery, GCS, Cloud SQL, Postgres, Redis)
- **Green**: Core services (MLflow, FastAPI, Airflow)
- **Orange**: Orchestration (DAGs, tasks)
- **Purple**: Monitoring (Prometheus, Grafana, exporter)
- **Red**: Alerting (Discord, critical alerts)
- **Yellow**: External APIs (Paris Open Data)

### Labels to Include

- Port numbers (5000, 8000, 8081, 9090, 3000)
- Memory limits (regmodel-backend: 6GB)
- Schedules (Daily @ 02:00, Weekly @ Sunday)
- Data volumes (660K rows, 181K rows, 1.6K rows)
- Key thresholds (R² < 0.60, drift > 50%)

---

## Legend

- **Solid arrows** → Data flow
- **Dashed arrows** → Monitoring/metrics scraping
- **Dotted arrows** → Alerting/notifications
- **Bold boxes** → Critical services (FastAPI, MLflow, Airflow)
- **Thin boxes** → Supporting services (postgres, redis, proxy)
- **Rounded boxes** → External services (GCP, Paris API, Discord)

---

## Drawing Steps

1. **Draw Layer 1 (bottom)**: Place all Docker services + GCP cloud services
2. **Draw Layer 2 (middle)**: Add DAG 1 + DAG 2 with arrows to BigQuery
3. **Draw Layer 3 (top)**: Add DAG 3 + training flow + monitoring connections
4. **Connect layers**: Draw arrows showing data flow between layers
5. **Add monitoring**: Connect Prometheus to Grafana, Grafana to Discord
6. **Label everything**: Ports, schedules, data volumes, thresholds
7. **Add legend**: Bottom-right corner

---

**Recommended Tools**:
- Excalidraw: https://excalidraw.com
- Export as SVG or PNG when done
- Save to: `docs/architecture_diagram.svg`

---

**Last Updated**: 2025-11-05
