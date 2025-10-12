# 🎯 MLOps Roadmap — Bike Traffic Prediction

**Date limite soutenance** : 7 novembre 2025
**Branche principale** : `feat/mlops-integration`

---

## 📊 État actuel (Phases 0-1 complétées ✅)

- ✅ Modèles ML entraînés (RF, NN)
- ✅ MLflow tracking opérationnel (dev/prod)
- ✅ Registry custom via `summary.json` GCS
- ✅ Backend FastAPI déployé sur Cloud Run (regmodel uniquement)
- ✅ Frontend Streamlit déployé
- ✅ Docker + docker-compose pour dev local
- ✅ Environnements dev/prod séparés

---

## 🚀 Phases MLOps à implémenter

### **Phase 2 : Tests, CI & Data Versioning**

#### **2.1 Data Versioning with DVC** (`feat/mlops-dvc-data-versioning`) ✅

**Implementation completed** ✅

📚 **Full documentation**: [docs/dvc.md](docs/dvc.md)

**Deliverables** ✅:

- ✅ Temporal split: reference (660K rows, 69.7%) + current (288K rows, 30.3%)
- ✅ DVC tracking with GCS remote storage
- ✅ `scripts/split_data_temporal.py` implemented

---

#### **2.2 Tests unitaires + CI** (`feat/mlops-tests-ci`) ✅

**Implementation completed** ✅

📚 **Full documentation**:

- [docs/pytest.md](docs/pytest.md) - Complete test suite
- [docs/ci.md](docs/ci.md) - CI/CD with GitHub Actions + Codecov

**Deliverables** ✅:

- ✅ **47 tests** passing (13 pipelines + 17 preprocessing + 11 API + 6 registry)
- ✅ **68% coverage** (app/classes: 73.42%, model_registry: 56.31%)
- ✅ GitHub Actions CI configured with **UV**
- ✅ Codecov integration active ([dashboard](https://app.codecov.io/gh/arthurcornelio88/bike-count-prediction-app))
- ✅ Coverage artifacts (HTML reports, 30 days retention)

**Files created**:

```text
tests/
├── test_pipelines.py          ✅ 13 tests (RF, NN)
├── test_preprocessing.py      ✅ 17 tests (transformers)
├── test_api_regmodel.py       ✅ 11 tests (FastAPI /predict)
├── test_model_registry.py     ✅ 6 tests (summary.json logic)
├── conftest.py                ✅ Shared fixtures
pytest.ini                     ✅ Configuration
.github/workflows/ci.yml       ✅ GitHub Actions
.coveragerc                    ✅ Coverage config
```

---

#### **2.3 Backend API `/train` + MLflow Integration** (`feat/mlops-tests-ci`) ✅

**Implementation completed** ✅

📚 **Documentation**: [docs/backend.md](docs/backend.md#train---train-and-upload-model)

**Objectifs** :

- ✅ Refactor training logic into unified `train_model()` function
- ✅ Create FastAPI `/train` endpoint for remote training
- ✅ Integrate MLflow tracking in docker-compose stack
- ✅ Support DVC-tracked datasets (reference/current)
- ✅ Automatic GCS upload + `summary.json` update

**Deliverables** ✅:

- ✅ `train_model()` function in [train.py:256](backend/regmodel/app/train.py#L256)
- ✅ `/train` endpoint in [fastapi_app.py:101](backend/regmodel/app/fastapi_app.py#L101)
- ✅ Docker Compose with RegModel API + MLflow server
- ✅ UV-optimized Dockerfile ([backend/regmodel/Dockerfile](backend/regmodel/Dockerfile))
- ✅ Dedicated pyproject.toml for RegModel service
- ✅ MLflow tracking already integrated in `train_rf()`, `train_nn()`, `train_rfc()`

**Architecture**:

```yaml
services:
  mlflow:
    - Tracking server on port 5000
    - Backend store: ./mlruns_dev
    - Artifacts: ./mlflow_artifacts
    - Healthcheck enabled

  regmodel-backend:
    - FastAPI on port 8000
    - Depends on MLflow (healthcheck)
    - Mounts: code, GCS credentials, data
    - Hot reload enabled (dev mode)
```

**API Usage**:

```bash
# Train RF model on reference data
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "rf",
    "data_source": "reference",
    "env": "prod"
  }'

# Response includes: run_id, metrics, model_uri
```

**Supported models**:

- `rf`: Random Forest regressor
- `nn`: Neural Network regressor
- `rf_class`: Random Forest classifier (affluence detection)

**Métriques trackées** (aligné avec `summary.json`) :

- **Régression (RF, NN)** : `r2_train`, `rmse_train`
- **Classification (RFC)** : `accuracy`, `precision`, `recall`, `f1_score`
- **Hyperparams** :
  - RF: `n_estimators`, `max_depth`, `random_state`
  - NN: `embedding_dim`, `batch_size`, `epochs`, `total_params`

**Validation completed** ✅:

- ✅ Full stack tested: `docker compose up` works
- ✅ MLflow UI accessible at <http://localhost:5000>
- ✅ `/train` endpoint tested with RF, NN models
- ✅ Test mode (`test_mode=true`) working with `test_sample.csv` (6s for NN, ~30s for RF)
- ✅ Metrics correctly returned in API response (RMSE, R²)
- ✅ MLflow tracking confirmed (runs, metrics, tags, artifacts)

---

### **Phase 3 : Orchestration Airflow + Monitoring Production** (`feat/mlops-airflow-pipeline`)

**Status**: 🔄 In Progress

**Objectifs unifiés** :

- 🔄 Pipeline automatisé end-to-end avec Airflow
- 📊 Monitoring avec BigQuery (raw, predictions, audit)
- 🔍 Drift detection avec Evidently
- 🎯 Réentraînement intelligent via endpoint `/train` (fine-tuning)
- 📈 Métriques API avec Prometheus + Grafana
- 🔒 Sécurité API (API Key + Rate Limiting)

**Data Strategy** (Updated 2025-10-11) ✅:

After data quality validation, we identified that all data sources (reference_data.csv,
current_data.csv, current_api_data.csv) are from the same origin (Paris Open Data historical
exports) with perfect correlation (r=1.0, MAE=0).

**Final Decision**: Use `current_api_data.csv` (905k records, 2024-09-01 → 2025-10-10) as unified baseline:

- 80% Train: ~724k records (2024-09 → 2025-08)
- 20% Test: ~181k records (2025-08 → 2025-10)
- Live API ingestion starting 2025-10-11 (cutoff date)
- Weekly drift detection + conditional fine-tuning

📚 **Full documentation**: [docs/fetch_data_strategy.md](docs/fetch_data_strategy.md)

---

#### **3.1 Data Preparation & Baseline** ✅

**Baseline Creation**:

```bash
# Split current_api_data.csv into train/test (80/20 split)
python scripts/split_data_temporal.py

# Output:
# - data/train_baseline.csv (~724k records, 2024-09-01 → 2025-08-15)
# - data/test_baseline.csv (~181k records, 2025-08-16 → 2025-10-10)
```

**GCS Upload** (baseline for champion model training):

```bash
# Upload train_baseline.csv to GCS
gsutil -m cp data/train_baseline.csv gs://<your-bucket>/data/train_baseline.csv

# Verify upload
gsutil ls -lh gs://<your-bucket>/data/
```

**DVC Tracking** (optional - for local versioning):

```bash
dvc add data/train_baseline.csv data/test_baseline.csv
dvc push
git add data/*.dvc .dvc/config
git commit -m "chore: add new baseline from current_api_data"
```

---

#### **3.1.5 Training Strategy** (Hybrid Architecture)

**Architecture**: Local champion training + Production fine-tuning

| Component | Where | When | Data Size | Duration |
|-----------|-------|------|-----------|----------|
| **Champion Training** | 💻 Local | One-time (+ quarterly) | 724k records | 15-30 min |
| **Fine-Tuning** | ☁️ Production | Weekly (if drift) | 2k records | 5-10 min |
| **Evaluation** | ☁️ Production | Weekly | 181k test set | 2-3 min |
| **Inference** | ☁️ Production | Daily | 100 records | <1 sec |

**Workflow**:

```text
┌─────────────────────────────────────────────────────────┐
│ INITIAL SETUP (Local - One Time)                       │
├─────────────────────────────────────────────────────────┤
│ 1. Train champion_v1 on train_baseline.csv (local)     │
│ 2. Evaluate on test_baseline.csv → MAE: ~12            │
│ 3. Upload to GCS + MLflow registry                     │
│ 4. Deploy to Cloud Run API                             │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ PRODUCTION (Weekly DAG)                                 │
├─────────────────────────────────────────────────────────┤
│ 1. Fetch last 7 days from BigQuery                     │
│ 2. Drift detection (Evidently vs test_baseline)        │
│ 3. If NO drift → skip, keep champion                   │
│ 4. If drift → fine-tune on last 30 days                │
│ 5. Evaluate challenger on SAME test_baseline.csv       │
│ 6. Champion/Challenger decision (5% threshold)         │
│ 7. Log metrics to monitoring_audit.logs                │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ QUARTERLY RETRAIN (Local - Every 3 months)             │
├─────────────────────────────────────────────────────────┤
│ 1. Download all BigQuery data (3 months)               │
│ 2. Merge with train_baseline.csv → new_train.csv       │
│ 3. Retrain champion_v2 locally (full training)         │
│ 4. Evaluate on SAME test_baseline.csv                  │
│ 5. If improved → deploy as new champion                │
└─────────────────────────────────────────────────────────┘
```

**Key Decisions**:

- **Local training**: Full champion model on complete baseline (724k records)
- **Production fine-tuning**: Lightweight adaptation on recent data (30 days, ~2k records)
- **Fixed test set**: Always evaluate on same test_baseline.csv for valid comparison
- **Champion/Challenger**: Promote only if 5% MAE improvement on test set

📚 **Full strategy**: [docs/training_strategy.md](docs/training_strategy.md)

---

#### **3.2 Architecture BigQuery**

**3 Datasets pour traçabilité complète** :

```yaml
# Structure BigQuery
datascientest-460618:
  bike_traffic_raw:           # Données brutes quotidiennes
    - daily_YYYYMMDD          # Tables par jour (comptage horaire)

  bike_traffic_predictions:   # Prédictions quotidiennes
    - daily_YYYYMMDD          # Prédictions + scores de confiance
    - prediction_ts           # Timestamp de prédiction

  monitoring_audit:           # Logs de monitoring et réentraînement
    - logs                    # Audit complet (drift, AUC, fine-tuning)
```

**Schema des tables** :

```python
# bike_traffic_raw.daily_YYYYMMDD
{
    "Comptage horaire": INTEGER,
    "Date et heure de comptage": TIMESTAMP,
    "Identifiant du compteur": STRING,
    "Nom du compteur": STRING,
    "Coordonnées géographiques": STRING,
    "ingestion_ts": TIMESTAMP
}

# bike_traffic_predictions.daily_YYYYMMDD
{
    "Comptage horaire": INTEGER,          # Valeur réelle (si disponible)
    "prediction": FLOAT,                   # Prédiction du modèle
    "model_type": STRING,                  # rf, nn, rf_class
    "model_version": STRING,               # Timestamp du modèle
    "prediction_ts": TIMESTAMP
}

# monitoring_audit.logs
{
    "timestamp": TIMESTAMP,
    "drift_detected": BOOLEAN,
    "rmse": FLOAT,
    "r2": FLOAT,
    "fine_tune_triggered": BOOLEAN,
    "fine_tune_success": BOOLEAN,
    "model_improvement": FLOAT,            # Δ R²
    "env": STRING,
    "error_message": STRING
}
```

---

#### **3.2 DAGs Airflow (Architecture modulaire)**

**3 DAGs séparés pour isoler les responsabilités** :

```mermaid
graph LR
    A[dag_daily_fetch_data] -->|@daily| B[BigQuery raw]
    C[dag_daily_prediction] -->|@daily| D[BigQuery predictions]
    E[dag_monitor_and_train] -->|@weekly| F{Drift?}
    F -->|Yes| G[Evaluate Model]
    G -->|Poor R²| H[Fine-tune via /train]
    G -->|Good R²| I[End]
    H --> J[Update BigQuery audit]
```

**📁 Structure des fichiers** :

```text
dags/
├── dag_daily_fetch_data.py          # Ingestion données brutes → BigQuery
├── dag_daily_prediction.py          # Prédictions via /predict → BigQuery
├── dag_monitor_and_train.py         # Drift + Eval + Fine-tuning
└── utils/
    ├── bike_helpers.py               # Fonctions BigQuery, GCS
    └── env_config.py                 # Config ENV/PROD avec Secret Manager
```

---

#### **3.3 DAG 1 : Ingestion des données** (`dag_daily_fetch_data.py`)

**Objectif** : Récupérer les données de trafic cycliste et stocker dans BigQuery

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
import pandas as pd
from google.cloud import bigquery
from utils.env_config import get_env_config

ENV_CONFIG = get_env_config()  # Gère DEV/PROD + Secret Manager

def fetch_bike_data_to_bq(**context):
    """
    Fetch latest bike traffic data from Paris Open Data API
    Store in BigQuery: bike_traffic_raw.daily_YYYYMMDD
    """
    today = datetime.utcnow().strftime("%Y%m%d")

    # Paris Open Data API (comptage vélo)
    api_url = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/comptage-velo-donnees-compteurs/records"
    params = {
        "limit": 1000,
        "order_by": "date_et_heure_de_comptage DESC"
    }

    response = requests.get(api_url, params=params)
    if response.status_code != 200:
        raise Exception(f"❌ API failed: {response.status_code}")

    data = response.json()
    df = pd.DataFrame([r['fields'] for r in data['results']])
    df["ingestion_ts"] = datetime.utcnow().isoformat()

    # Write to BigQuery
    table_id = f"{ENV_CONFIG['BQ_PROJECT']}.bike_traffic_raw.daily_{today}"
    df.to_gbq(
        destination_table=table_id,
        project_id=ENV_CONFIG['BQ_PROJECT'],
        if_exists="replace",
        location=ENV_CONFIG['BQ_LOCATION']
    )

    print(f"✅ Ingested {len(df)} records into {table_id}")

with DAG(
    dag_id="daily_fetch_bike_data",
    schedule_interval="@daily",
    start_date=datetime(2024, 10, 1),
    catchup=False,
    tags=["bike", "ingestion", "bigquery"]
) as dag:

    fetch_task = PythonOperator(
        task_id="fetch_to_bigquery",
        python_callable=fetch_bike_data_to_bq
    )
```

---

#### **3.4 DAG 2 : Prédictions quotidiennes** (`dag_daily_prediction.py`)

**Objectif** : Lire BigQuery → Prédire via `/predict` → Stocker résultats

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
import pandas as pd
from google.cloud import bigquery
from utils.env_config import get_env_config

ENV_CONFIG = get_env_config()

def run_daily_prediction(**context):
    """
    1. Read from BigQuery raw table
    2. Call /predict endpoint
    3. Store predictions in BigQuery predictions table
    """
    today = datetime.utcnow().strftime("%Y%m%d")
    bq = bigquery.Client()

    # 1️⃣ Read raw data
    raw_table = f"{ENV_CONFIG['BQ_PROJECT']}.bike_traffic_raw.daily_{today}"
    df = bq.query(f"SELECT * FROM `{raw_table}` LIMIT 500").to_dataframe()

    # 2️⃣ Call /predict endpoint
    api_url = f"{ENV_CONFIG['API_URL']}/predict"
    response = requests.post(api_url, json={
        "records": df.to_dict(orient="records"),
        "model_type": "rf",
        "metric": "r2"
    })

    if response.status_code != 200:
        raise Exception(f"❌ Prediction failed: {response.text}")

    predictions = response.json()["predictions"]
    df["prediction"] = predictions
    df["model_type"] = "rf"
    df["prediction_ts"] = datetime.utcnow().isoformat()

    # 3️⃣ Store in BigQuery
    pred_table = f"{ENV_CONFIG['BQ_PROJECT']}.bike_traffic_predictions.daily_{today}"
    df.to_gbq(
        destination_table=pred_table,
        project_id=ENV_CONFIG['BQ_PROJECT'],
        if_exists="replace",
        location=ENV_CONFIG['BQ_LOCATION']
    )

    print(f"✅ Predictions saved to {pred_table}")

with DAG(
    dag_id="daily_prediction",
    schedule_interval="@daily",
    start_date=datetime(2024, 10, 1),
    catchup=False,
    tags=["bike", "prediction", "bigquery"]
) as dag:

    predict_task = PythonOperator(
        task_id="predict_daily_data",
        python_callable=run_daily_prediction
    )
```

---

#### **3.5 DAG 3 : Monitoring + Fine-tuning** (`dag_monitor_and_train.py`)

**Objectif** : Drift detection → Validation → Fine-tuning conditionnel

```python
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime, timedelta
import requests
import pandas as pd
from google.cloud import bigquery
from utils.env_config import get_env_config

ENV_CONFIG = get_env_config()

# 1️⃣ DRIFT DETECTION
def run_drift_monitoring(**context):
    """
    Compare reference vs current data using Evidently
    Calls backend endpoint /monitor for drift detection
    """
    today = datetime.utcnow().strftime("%Y%m%d")
    bq = bigquery.Client()

    # Load current data from BigQuery
    curr_table = f"{ENV_CONFIG['BQ_PROJECT']}.bike_traffic_raw.daily_{today}"
    df_curr = bq.query(f"SELECT * FROM `{curr_table}` LIMIT 1000").to_dataframe()

    # Call /monitor endpoint with reference data from GCS
    response = requests.post(f"{ENV_CONFIG['API_URL']}/monitor", json={
        "reference_path": "gs://df_traffic_cyclist1/data/reference_data.csv",
        "current_data": df_curr.to_dict(orient="records"),
        "output_html": f"drift_report_{today}.html"
    })

    if response.status_code != 200:
        raise Exception(f"❌ Drift detection failed: {response.text}")

    result = response.json()
    drift_detected = result["drift_summary"]["drift_detected"]

    context['ti'].xcom_push(key="drift_detected", value=drift_detected)
    print(f"{'🚨 Drift detected' if drift_detected else '✅ No drift'}")

# 2️⃣ MODEL VALIDATION
def validate_model(**context):
    """
    Compare predictions vs true labels from BigQuery
    Calculate RMSE and R² for model performance
    """
    today = datetime.utcnow().strftime("%Y%m%d")
    bq = bigquery.Client()

    # Join predictions with actual values
    query = f"""
    SELECT
        p.prediction,
        r.`Comptage horaire` as true_value
    FROM `{ENV_CONFIG['BQ_PROJECT']}.bike_traffic_predictions.daily_{today}` p
    JOIN `{ENV_CONFIG['BQ_PROJECT']}.bike_traffic_raw.daily_{today}` r
    ON p.`Identifiant du compteur` = r.`Identifiant du compteur`
    """

    df = bq.query(query).to_dataframe()

    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np

    rmse = np.sqrt(mean_squared_error(df['true_value'], df['prediction']))
    r2 = r2_score(df['true_value'], df['prediction'])

    context['ti'].xcom_push(key="rmse", value=rmse)
    context['ti'].xcom_push(key="r2", value=r2)

    print(f"📊 RMSE: {rmse:.2f}, R²: {r2:.4f}")

# 3️⃣ DECISION LOGIC
def decide_if_fine_tune(**context):
    """
    Decide whether to trigger fine-tuning based on:
    - Drift detected
    - R² below threshold (0.65)
    - RMSE above threshold (60.0)
    """
    drift = context['ti'].xcom_pull(task_ids="monitor_drift", key="drift_detected")
    r2 = context['ti'].xcom_pull(task_ids="validate_model", key="r2")
    rmse = context['ti'].xcom_pull(task_ids="validate_model", key="rmse")

    R2_THRESHOLD = 0.65
    RMSE_THRESHOLD = 60.0

    if drift and (r2 < R2_THRESHOLD or rmse > RMSE_THRESHOLD):
        print(f"🚨 Fine-tuning needed: drift={drift}, R²={r2:.4f}, RMSE={rmse:.2f}")
        return "fine_tune_model"
    else:
        print(f"✅ Model OK: drift={drift}, R²={r2:.4f}, RMSE={rmse:.2f}")
        return "end_monitoring"

# 4️⃣ FINE-TUNING VIA /train ENDPOINT
def fine_tune_model(**context):
    """
    Call /train endpoint with fine_tuning=True
    Uses latest data from BigQuery for incremental learning
    """
    today = datetime.utcnow().strftime("%Y%m%d")
    bq = bigquery.Client()

    # Get fresh data from BigQuery
    table = f"{ENV_CONFIG['BQ_PROJECT']}.bike_traffic_raw.daily_{today}"
    df_fresh = bq.query(f"SELECT * FROM `{table}` LIMIT 2000").to_dataframe()

    # Call /train endpoint with fine-tuning mode
    response = requests.post(f"{ENV_CONFIG['API_URL']}/train", json={
        "model_type": "rf",
        "data_source": "bigquery",
        "data": df_fresh.to_dict(orient="records"),
        "env": ENV_CONFIG['ENV'],
        "fine_tuning": True,
        "learning_rate": 0.01,
        "epochs": 10
    }, timeout=600)

    if response.status_code != 200:
        raise Exception(f"❌ Fine-tuning failed: {response.text}")

    result = response.json()

    # Log to BigQuery audit
    audit_df = pd.DataFrame([{
        "timestamp": datetime.utcnow(),
        "drift_detected": context['ti'].xcom_pull(task_ids="monitor_drift", key="drift_detected"),
        "rmse": context['ti'].xcom_pull(task_ids="validate_model", key="rmse"),
        "r2": context['ti'].xcom_pull(task_ids="validate_model", key="r2"),
        "fine_tune_triggered": True,
        "fine_tune_success": True,
        "model_improvement": result.get("r2_improvement", 0.0),
        "env": ENV_CONFIG['ENV']
    }])

    audit_df.to_gbq(
        destination_table=f"{ENV_CONFIG['BQ_PROJECT']}.monitoring_audit.logs",
        project_id=ENV_CONFIG['BQ_PROJECT'],
        if_exists="append",
        location=ENV_CONFIG['BQ_LOCATION']
    )

    print(f"✅ Fine-tuning completed: R² improvement = {result.get('r2_improvement', 0):.4f}")

# 5️⃣ END WITHOUT TRAINING
def end_monitoring(**context):
    """Log monitoring results without training"""
    audit_df = pd.DataFrame([{
        "timestamp": datetime.utcnow(),
        "drift_detected": context['ti'].xcom_pull(task_ids="monitor_drift", key="drift_detected"),
        "rmse": context['ti'].xcom_pull(task_ids="validate_model", key="rmse"),
        "r2": context['ti'].xcom_pull(task_ids="validate_model", key="r2"),
        "fine_tune_triggered": False,
        "fine_tune_success": False,
        "model_improvement": 0.0,
        "env": ENV_CONFIG['ENV']
    }])

    audit_df.to_gbq(
        destination_table=f"{ENV_CONFIG['BQ_PROJECT']}.monitoring_audit.logs",
        project_id=ENV_CONFIG['BQ_PROJECT'],
        if_exists="append",
        location=ENV_CONFIG['BQ_LOCATION']
    )

    print("✅ Monitoring complete - no training needed")

# === DAG DEFINITION ===
with DAG(
    dag_id="monitor_and_fine_tune",
    schedule_interval="@weekly",
    start_date=datetime(2024, 10, 1),
    catchup=False,
    tags=["bike", "monitoring", "drift", "training"]
) as dag:

    monitor = PythonOperator(
        task_id="monitor_drift",
        python_callable=run_drift_monitoring
    )

    validate = PythonOperator(
        task_id="validate_model",
        python_callable=validate_model
    )

    decide = BranchPythonOperator(
        task_id="decide_fine_tune",
        python_callable=decide_if_fine_tune
    )

    fine_tune = PythonOperator(
        task_id="fine_tune_model",
        python_callable=fine_tune_model
    )

    end = PythonOperator(
        task_id="end_monitoring",
        python_callable=end_monitoring,
        trigger_rule="none_failed_min_one_success"
    )

    # Pipeline flow
    monitor >> validate >> decide
    decide >> [fine_tune, end]
```

**Visualisation du DAG** :

```text
[Monitor Drift] → [Validate Model] → [Decide]
                                        ├─→ [Fine-tune] → [End]
                                        └─→ [End (no training)]
```

---

#### **3.6 Prometheus + Grafana (Métriques API)**

**Instrumentation FastAPI** :

```python
# backend/regmodel/app/fastapi_app.py
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
from starlette.middleware.base import BaseHTTPMiddleware
import time

# Métriques custom
predictions_total = Counter('predictions_total', 'Total predictions', ['model_type'])
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency', ['model_type'])
active_models = Gauge('active_models_count', 'Cached models count')
training_total = Counter('training_total', 'Total training runs', ['model_type', 'status'])

class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start

        if request.url.path == "/predict":
            model_type = getattr(request.state, 'model_type', 'unknown')
            predictions_total.labels(model_type=model_type).inc()
            prediction_latency.labels(model_type=model_type).observe(duration)

        return response

app.add_middleware(PrometheusMiddleware)

# Endpoint métriques
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.get("/health")
def health():
    active_models.set(len(model_cache))
    return {"status": "healthy", "cached_models": len(model_cache)}
```

**Docker Compose** :

```yaml
# docker-compose.yaml (ajout)
services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=15d'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
    depends_on:
      - prometheus

volumes:
  prometheus_data:
  grafana_data:
```

**Configuration Prometheus** :

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'regmodel-api'
    static_configs:
      - targets: ['regmodel-backend:8000']
    metrics_path: '/metrics'
```

**Dashboard Grafana** :

- Requêtes/sec : `rate(predictions_total[5m])`
- Latence p50/p95/p99 : `histogram_quantile(0.95, prediction_latency_seconds)`
- Taux erreur : `rate(http_requests_total{status=~"5.."}[5m])`
- Trainings réussis : `rate(training_total{status="success"}[1h])`

---

#### **3.7 Sécurité API**

**API Key + Rate Limiting** :

```python
# backend/regmodel/app/fastapi_app.py
from fastapi import Security, HTTPException, Depends, Request
from fastapi.security.api_key import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

API_KEY = os.getenv("API_KEY_SECRET", "dev-key-unsafe")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return key

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

@app.post("/predict", dependencies=[Depends(verify_api_key)])
@limiter.limit("100/minute")
async def predict(data: PredictRequest, request: Request):
    request.state.model_type = data.model_type
    model = get_cached_model(data.model_type, data.metric)
    y_pred = model.predict_clean(pd.DataFrame(data.records))
    return {"predictions": y_pred.tolist()}

@app.post("/train", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/hour")
async def train(data: TrainRequest, request: Request):
    # Training logic with fine-tuning support
    ...
```

**Variables d'environnement** :

```bash
# backend/regmodel/.env
ENV=PROD
API_KEY_SECRET=super-secret-prod-key-2024
```

---

### **Phase 5 (Bonus) : Kubernetes** (`feat/mlops-kubernetes`)

**Si temps disponible** :

```yaml
# k8s/deployment-regmodel.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: regmodel-api
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: regmodel
        image: europe-west1-docker.pkg.dev/datascientest-460618/cloud-run-images/regmodel-api:latest
        env:
        - name: API_KEY_SECRET
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: api-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
---
apiVersion: v1
kind: Service
metadata:
  name: regmodel-service
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: regmodel-api
```

---

## 📋 Stratégie de branches

```text
feat/mlops-integration (branche principale)
├── feat/mlops-dvc-data-versioning      # Phase 2.1
├── feat/mlops-tests-ci                 # Phase 2.2
├── feat/mlops-airflow-pipeline         # Phase 3
├── feat/mlops-monitoring               # Phase 4
└── feat/mlops-kubernetes (optionnel)   # Phase 5
```

**Workflow Git** :

1. Créer branche depuis `feat/mlops-integration`
2. Développer feature
3. Tester localement
4. Push + merge dans `feat/mlops-integration`
5. À la fin : merge `feat/mlops-integration` → `master`

---

## 🏗️ Structure finale du projet

```text
ds_traffic_cycliste1/
├── .github/
│   └── workflows/
│       └── ci.yml                     # ✨ GitHub Actions
├── .dvc/
│   └── config                         # ✨ DVC config
├── data/
│   ├── reference_data.csv.dvc         # ✨ Pointer DVC (train)
│   ├── current_data.csv.dvc           # ✨ Pointer DVC (prod)
│   └── .gitignore
├── dags/
│   └── ml_pipeline_dag.py             # ✨ DAG Airflow
├── monitoring/
│   ├── prometheus.yml                 # ✨ Config Prometheus
│   ├── drift_detector.py              # ✨ Script Evidently
│   └── grafana/
│       └── provisioning/
│           └── dashboards/
│               └── api-metrics.json   # ✨ Dashboard Grafana
├── tests/                             # ✨ Tests pytest
│   ├── test_pipelines.py
│   ├── test_preprocessing.py
│   ├── test_api_regmodel.py
│   └── conftest.py
├── scripts/
│   └── split_data_temporal.py         # ✨ Split ref/current
├── backend/
│   └── regmodel/
│       └── app/
│           └── fastapi_app.py         # ✨ + Prometheus + API key
├── docker-compose.yaml                # ✨ + Airflow + Prometheus + Grafana
├── src/
│   └── train.py
├── app/
│   └── streamlit_app.py
├── docs/
│   ├── mlops-data-versioning.md       # ✨ Doc DVC
│   ├── mlops-orchestration.md         # ✨ Doc Airflow
│   └── mlops-monitoring.md            # ✨ Doc Prometheus/Evidently
├── pytest.ini                         # ✨ Config pytest
├── MLOPS_ROADMAP.md                   # ✨ Ce fichier
└── README.md                          # ✨ Mis à jour
```

---

## 📅 Timeline (jusqu'au 7 nov)

| Phase | Branche | Durée | Dates indicatives |
|-------|---------|-------|-------------------|
| 2.1 | `feat/mlops-dvc-data-versioning` | 2j | Oct 3-4 |
| 2.2 | `feat/mlops-tests-ci` | 3j | Oct 5-7 |
| 3 | `feat/mlops-airflow-pipeline` | 5j | Oct 8-12 |
| 4 | `feat/mlops-monitoring` | 6j | Oct 13-18 |
| **Buffer** | Debug, intégration | 5j | Oct 19-23 |
| **Doc finale** | README, présentation | 3j | Oct 24-26 |
| **Répétition** | Soutenance | 3j | Nov 4-6 |

---

## ✅ Checklist finale

### Technique

- [ ] DVC configuré + data reference/current versionnées
- [ ] Tests unitaires couvrent >80% du code
- [ ] CI passe sur toutes les branches
- [ ] DAG Airflow avec logique réentraînement conditionnel
- [ ] APIs sécurisées (API key + rate limit)
- [ ] Prometheus scrape métriques API
- [ ] Dashboards Grafana opérationnels
- [ ] Rapports Evidently générés automatiquement
- [ ] Docker Compose lance toute la stack

### Documentation

- [ ] README principal mis à jour
- [ ] Doc DVC (split temporel, versioning)
- [ ] Doc Airflow (DAG, branchement, scheduling)
- [ ] Doc Monitoring (Prometheus queries, dashboards Grafana)
- [ ] Doc Evidently (drift detection, alertes)

### Présentation

- [ ] Slides de présentation (15-20 slides)
- [ ] Démo vidéo de secours
- [ ] Diagramme architecture MLOps complet
- [ ] Exemples de métriques/dashboards

---

## 🎤 Structure présentation soutenance (20 min)

1. **Contexte & objectifs** (3 min)
   - Problème : prédiction trafic cycliste Paris
   - Stack technique : Streamlit + FastAPI + MLflow + Airflow

2. **Architecture MLOps** (5 min)
   - Schéma complet : Data versioning (DVC) → Training (Airflow) → Deployment (Cloud Run) → Monitoring (Prometheus/Evidently)
   - Highlight : logique réentraînement conditionnel

3. **Démo live** (8 min)
   - Trigger DAG Airflow → voir branchement retrain
   - Appel API avec métriques Prometheus
   - Dashboard Grafana en temps réel
   - Rapport Evidently drift detection

4. **Défis techniques & solutions** (3 min)
   - Split temporel data pour drift detection
   - Gestion cache modèles avec hash MD5
   - Intégration DVC + Airflow

5. **Q&A** (1 min)

---

## 📚 Ressources

- [DVC Documentation](https://dvc.org/doc)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [Prometheus Python Client](https://github.com/prometheus/client_python)
- [Evidently AI Docs](https://docs.evidentlyai.com/)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)

---

**Prochaine étape** : Créer branche `feat/mlops-dvc-data-versioning` et implémenter DVC ! 🚀
