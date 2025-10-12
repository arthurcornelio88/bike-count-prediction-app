# 🎯 Training Strategy — Hybrid Architecture

**Last Updated**: 2025-10-12

---

## Overview

This document explains the **hybrid training strategy** for the bike traffic prediction models:
**champion models trained locally** on large historical datasets, **fine-tuned weekly in production**
on recent data to adapt to evolving patterns.

### Why Hybrid?

| Aspect | Local Champion | Production Fine-Tuning |
|--------|----------------|------------------------|
| **Purpose** | High-quality baseline | Adapt to recent patterns |
| **Data** | 724k historical records | Last 30 days (~2k records) |
| **Frequency** | Quarterly (or on-demand) | Weekly (if drift detected) |
| **Duration** | 15-30 minutes | 5-10 minutes |
| **Cost** | Free (local GPU) | Low (Cloud Run) |
| **Control** | Full experimentation | Automated via Airflow |

---

## Training Workflow

### Phase 1: Champion Training (Local Development)

**When:** Initial setup, quarterly refresh, or after major architecture changes

**Objective:** Create the best possible baseline model on comprehensive historical data

#### 1.1 Data Preparation

```bash
# Baseline datasets already uploaded to GCS
TRAIN_DATA_PATH=gs://df_traffic_cyclist1/data/train_baseline.csv  # 724k rows
TEST_DATA_PATH=gs://df_traffic_cyclist1/data/test_baseline.csv    # 181k rows

# These are referenced via environment variables in training
export TRAIN_DATA_PATH=gs://df_traffic_cyclist1/data/train_baseline.csv
export TEST_DATA_PATH=gs://df_traffic_cyclist1/data/test_baseline.csv
```

#### 1.2 Training Commands

**Basic Champion Training:**

```bash
# Train RandomForest champion on baseline data
python backend/regmodel/app/train.py \
    --model-type rf \
    --data-source baseline \
    --env dev

# Train Neural Network champion
python backend/regmodel/app/train.py \
    --model-type nn \
    --data-source baseline \
    --env dev
```

**CLI Parameters:**

- `--model-type`: Model architecture
  - `rf` — RandomForest (n_estimators=50, max_depth=20)
  - `nn` — Neural Network (embedding + dense layers)
  - `rf_class` — Binary classifier (high/low traffic)
- `--data-source`: Training data origin
  - `baseline` — Use train_baseline.csv from GCS (724k rows) ✅ **Recommended**
  - `reference` — Legacy reference_data.csv
  - `current` — Current production data snapshot
- `--env`: Environment mode
  - `dev` — Local development (default)
  - `prod` — Production mode
  - **Note:** Both modes upload everything to GCS (artifacts + metadata via MLflow)
- `--model-test`: Quick test mode (1000 rows sample for debugging)

**Quick Test Training:**

```bash
# Fast iteration during development (~10 seconds)
python backend/regmodel/app/train.py \
    --model-type rf \
    --data-source baseline \
    --model-test \
    --env dev
```

#### 1.3 What Happens During Training

1. **Setup:** Loads GCS credentials (`mlflow-trainer.json` for artifact upload)
2. **Data Loading:** Reads from `$TRAIN_DATA_PATH` (GCS or local fallback)
3. **Training:** Fits custom pipeline (cleaning → feature engineering → model)

   ![Training Pipeline NN](img/nn_train.png)

   *Figure 1: Neural Network training pipeline with custom transformers*

4. **Evaluation:** Computes metrics on training set (RMSE, R²)
5. **MLflow Logging:**

- Metadata uploaded to Cloud SQL PostgreSQL (experiments, runs, metrics)
- Model registered in MLflow Registry (`bike-traffic-rf` version X)

  ![MLflow UI Tracking](img/nn_mlflow.png)

  *Figure 2: MLflow UI showing experiment tracking and metrics*

- Artifacts uploaded to `gs://df_traffic_cyclist1/mlflow-artifacts/{run_id}/`

  ![MLflow Artifacts View](img/artifacts_mlflow_nn.png)

  *Figure 3: MLflow artifact storage showing model components (cleaner, model, OHE, preprocessor)*

6. **GCS Artifact Bucket (what's stored where)**

- Top-level buckets
  - `gs://df_traffic_cyclist1/mlflow-artifacts/` — MLflow artifacts (per-experiment, per-run folders)
  - `gs://df_traffic_cyclist1/models/` — exported model bundles and `summary.json`

- Typical artifact layout:

    gs://df_traffic_cyclist1/mlflow-artifacts/{experiment_id}/{run_id}/artifacts/model/

  ![GCS Artifact Storage](img/artifacts_gcs_nn.png)

  *Figure 4: GCS bucket structure with MLflow artifacts and model files*

7. **Summary Update:** `gs://df_traffic_cyclist1/models/summary.json` updated with run info for Airflow

- The `summary.json` file is appended with a new entry after each training run. It contains metadata
    used by Airflow to select and download candidate models for promotion.

- Example entry snapshot:

    ![Summary JSON](img/summary_json_example.png)

    *Figure 5: Example `summary.json` entry (timestamp, model_type, run_id, model_uri, metrics)*

#### 1.4 Expected Metrics (Baseline Data)

- **RandomForest:** RMSE ~47, R² ~0.79
- **Neural Network:** RMSE ~54, R² ~0.72

---

### Phase 2: Production Fine-Tuning (Automated Weekly)

**When:** Every week via `dag_monitor_and_train.py` if drift is detected

**Objective:** Adapt champion model to recent patterns without full retraining

#### 2.1 Fine-Tuning via API

The Airflow DAG calls the `/train` API endpoint with recent data:

```python
# dag_monitor_and_train.py
response = requests.post(
    f"{REGMODEL_API_URL}/train",
    json={
        "model_type": "rf",
        "data_source": "current",  # Last 30 days from BigQuery
        "hyperparams": {
            "learning_rate": 0.001,  # Lower than initial
            "n_estimators": 50,
            "max_depth": 20
        },
        "test_mode": False
    },
    timeout=600
)
```

**API Endpoint Parameters:**

- `model_type` — Same as CLI (`rf`, `nn`, `rf_class`)
- `data_source` — Source of training data
- `hyperparams` — Optional model-specific parameters
- `test_mode` — Boolean for quick testing

#### 2.2 Champion/Challenger Evaluation

The DAG compares the new model (challenger) against the current champion:

```python
# Evaluate on FIXED test set (test_baseline.csv from GCS)
test_baseline = pd.read_csv(os.getenv("TEST_DATA_PATH"))

challenger_rmse = evaluate(challenger_model, test_baseline)
champion_rmse = get_current_champion_rmse()

# Promotion decision (5% improvement threshold)
if challenger_rmse < champion_rmse * 0.95:
    promote_to_production(challenger_model)
    log_event("champion_promoted", challenger_rmse)
else:
    log_event("champion_kept", champion_rmse)
```

**Why 5% threshold?** Prevents noise-driven model churn while allowing genuine improvements.

---

### Phase 3: Quarterly Retrain (Local)

**When:** Every 3 months, or when performance degrades >20%

**Objective:** Incorporate accumulated production data into a new baseline

#### 3.1 Data Aggregation

```bash
# 1. Export 3 months of production data from BigQuery
bq extract --destination_format=CSV \
  'datascientest-460618:bike_traffic_raw.daily_*' \
  gs://df_traffic_cyclist1/exports/bq_export_*.csv

# 2. Download and merge with current baseline
gsutil -m cp gs://df_traffic_cyclist1/data/train_baseline.csv data/
gsutil -m cp gs://df_traffic_cyclist1/exports/bq_export_*.csv data/exports/

python scripts/merge_baseline_bq.py \
    --baseline data/train_baseline.csv \
    --bq-export data/exports/bq_export_*.csv \
    --output data/new_train_baseline.csv
```

#### 3.2 Retrain from Scratch

```bash
# Train on merged dataset
python backend/regmodel/app/train.py \
    --model-type rf \
    --data-source baseline \
    --env dev

# Evaluate against SAME test_baseline.csv
# If improved → upload as new baseline

gsutil cp data/new_train_baseline.csv \
  gs://df_traffic_cyclist1/data/train_baseline.csv

# Update environment variable reference
gcloud secrets versions add train-data-path \
  --data-file=- <<< "gs://df_traffic_cyclist1/data/train_baseline.csv"
```

---

## MLflow Integration

### Artifact Storage Architecture

The training system uses **MLflow** for experiment tracking and model registry:

```text
┌──────────────────┐        ┌─────────────────────┐
│  Training Script │───────>│  MLflow Server      │
│  (train.py)      │        │  localhost:5000     │
└──────────────────┘        └─────────────────────┘
         │                            │
         │ Artifacts ✅               │ Metadata ✅
         ↓                            ↓
┌──────────────────┐        ┌─────────────────────┐
│  GCS Bucket      │        │  Cloud SQL          │
│  mlflow-artifacts│        │  PostgreSQL         │
└──────────────────┘        └─────────────────────┘
                                      │
                            datascientest-460618:europe-west3:mlflow-metadata
```

**Key Components:**

1. **Backend Store** (metadata): ✅ **Cloud SQL PostgreSQL**
   - Instance: `mlflow-metadata` (europe-west3)
   - Database: `mlflow` (user: `mlflow_user`)
   - Connection: Via Cloud SQL Proxy in docker-compose
   - Benefits: Shared, persistent, scalable metadata storage
2. **Artifact Store** (models): `gs://df_traffic_cyclist1/mlflow-artifacts/`

- ✅ All model files (joblib, weights) are stored on GCS.
- ✅ Artifacts are accessible remotely with proper credentials.
- ✅ This bucket contains the canonical model files — treat it as the source of truth.

3. **Model Registry**: MLflow UI + `summary.json` for Airflow
   - MLflow UI: Rich versioning, lineage, comparison (metadata-driven)
   - summary.json: Programmatic access for DAGs (artifact-driven, reliable)

### Authentication Requirements

**For Training Script (Client):**

The training script (`train.py`) must authenticate to GCS before importing MLflow. This
ensures artifact uploads succeed when the script runs. The code checks the
`GOOGLE_APPLICATION_CREDENTIALS` environment variable and falls back to
`./mlflow-trainer.json` when needed (this happens at module import time).

**Service Account needed:** `mlflow-trainer@datascientest-460618.iam.gserviceaccount.com`
with `roles/storage.objectAdmin` for GCS bucket access.

**For MLflow Server (Docker):**

The MLflow stack is configured in `docker-compose.yaml` with two service accounts:

1. **Cloud SQL Proxy** (`gcp.json`):
   - Service Account: `streamlit-models@datascientest-460618.iam.gserviceaccount.com`
   - Role Required: `roles/cloudsql.client`
   - Purpose: Connect to Cloud SQL PostgreSQL instance for metadata storage

2. **MLflow Server** (`mlflow-ui-access.json`):
   - Service Account: `mlflow-ui-access@datascientest-460618.iam.gserviceaccount.com`
   - Role Required: `roles/storage.objectViewer`
   - Purpose: Read artifacts from GCS for UI display

**Cloud SQL Backend:**

- Instance: `mlflow-metadata` (europe-west3)
- Database: `mlflow` (user: `mlflow_user`)
- Connection: Via Cloud SQL Proxy on port 5432
- Benefits: Centralized team tracking, persistent, scalable

See [mlflow_cloudsql.md](./mlflow_cloudsql.md) for complete setup details and troubleshooting.

### Model Registry (Dual System)

**1. MLflow Registry** (Rich UI, versioning)

- URL: <http://localhost:5000\>
- Models: `bike-traffic-rf`, `bike-traffic-nn`
- Features: Version history, artifact lineage, metric comparison

**2. summary.json** (Airflow-compatible)

- Path: `gs://df_traffic_cyclist1/models/summary.json`
- Updated for **ALL** training runs (dev + prod)
- Structure:

```json
{
  "timestamp": "2025-10-12T15:37:11",
  "model_type": "rf",
  "env": "dev",
  "test_mode": false,
  "run_id": "35dc84a2ce7b427b8a3fded8435fef35",
  "model_uri": "gs://.../mlflow-artifacts/35dc84a.../artifacts/model/",
  "rmse": 47.28,
  "r2": 0.7920
}
```

**Why both?** MLflow for human exploration, summary.json for programmatic DAG access.

---

## Decision Triggers

### When to Train Champion Locally

| Trigger | Action | Priority |
|---------|--------|----------|
| **Quarterly schedule** | Merge 3 months prod data + retrain | Regular |
| **Performance drop >20%** | Investigate + retrain with fix | High |
| **New features available** | Experiment locally first | Medium |
| **Architecture change** | Full local development cycle | High |

### When to Fine-Tune in Production

| Trigger | Action | Priority |
|---------|--------|----------|
| **Weekly drift detected** | Fine-tune on last 30 days | Automated |
| **Performance drop 10-20%** | Lightweight adaptation | Medium |
| **Seasonal events** | Pre-emptive fine-tuning | Low |

### Evaluation Best Practices

✅ **DO:**

- Always evaluate on `test_baseline.csv` (fixed test set)
- Use 5% improvement threshold for promotion
- Log all decisions to `monitoring_audit.logs`

❌ **DON'T:**

- Train and evaluate on the same recent data
- Promote based on training metrics alone
- Skip test set evaluation

---

## Complete Workflow Diagram

```text
┌───────────────────────────────────────────────────────────────┐
│ LOCAL CHAMPION TRAINING (Quarterly)                          │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Merge historical + prod data → new_train_baseline.csv    │
│  2. python train.py --model-type rf --data-source baseline   │
│  3. Evaluate on test_baseline.csv from GCS                   │
│  4. Upload artifacts to gs://.../mlflow-artifacts/{run_id}/  │
│  5. Update summary.json [env=dev, test_mode=false]           │
│  6. Register in MLflow: bike-traffic-rf v{N}                 │
│                                                               │
│  Metrics: RMSE ~47, R² ~0.79 (RandomForest)                  │
│                                                               │
└───────────────────────────────────────────────────────────────┘
                            ↓
┌───────────────────────────────────────────────────────────────┐
│ PRODUCTION FINE-TUNING (Weekly via Airflow)                  │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  1. dag_monitor_and_train.py triggers on drift alert         │
│  2. POST /train API with last 30 days from BigQuery          │
│  3. Fine-tune current champion (low learning rate)           │
│  4. Evaluate challenger on test_baseline.csv                 │
│  5. Compare: challenger_rmse vs champion_rmse                │
│                                                               │
│     IF challenger < champion * 0.95:                          │
│        → Promote to production                                │
│        → Update summary.json [env=prod]                       │
│        → Log "champion_promoted" event                        │
│     ELSE:                                                     │
│        → Keep current champion                                │
│        → Log "champion_kept" event                            │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

## Environment Variables

Set these before training:

```bash
# Required for GCS data access
export TRAIN_DATA_PATH=gs://df_traffic_cyclist1/data/train_baseline.csv
export TEST_DATA_PATH=gs://df_traffic_cyclist1/data/test_baseline.csv

# Required for MLflow artifact upload
export GOOGLE_APPLICATION_CREDENTIALS=./mlflow-trainer.json

# Optional: MLflow tracking URI (defaults to localhost:5000)
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

---

## Summary

**Key Principles:**

1. **Local champion training** ensures high-quality baselines with full control
2. **Production fine-tuning** provides agility to adapt to recent patterns
3. **Fixed test set** (`test_baseline.csv`) guarantees fair model comparison
4. **Dual registry** (MLflow + summary.json) serves both humans and automation
5. **Conservative promotion** (5% threshold) prevents model churn

**Benefits:**

- ✅ No cloud costs for heavy training (local GPU)
- ✅ Fast adaptation to drift (weekly fine-tuning)
- ✅ Rigorous evaluation (fixed test set)
- ✅ Full lineage tracking (MLflow + Airflow logs)
- ✅ Rollback capability (versioned models in registry)

---

**Related Documentation:**

- [secrets.md](./secrets.md) — GCS credentials and Secret Manager setup
- [bigquery_setup.md](./bigquery_setup.md) — Production data pipeline
- [dvc.md](./dvc.md) — Data versioning strategy
- [MLOPS_ROADMAP.md](../MLOPS_ROADMAP.md) — Overall MLOps architecture
