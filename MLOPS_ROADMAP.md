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

#### **2.1 Data Versioning avec DVC** (`feat/mlops-dvc-data-versioning`)

**Objectifs** :
- Versionner les datasets avec DVC (Data Version Control)
- Split temporel : données **reference** (train/test) vs **current** (production)
- Utiliser `reference` pour l'entraînement
- Utiliser `current` pour la détection de drift

**Workflow** :
1. **Installation DVC + configuration GCS**
   ```bash
   pip install dvc[gs]
   dvc init
   dvc remote add -d gcs_storage gs://df_traffic_cyclist1/dvc-storage
   dvc remote modify gcs_storage credentialpath ./gcp.json
   ```

2. **Split temporel des données**
   ```python
   # scripts/split_data_temporal.py
   import pandas as pd
   from datetime import datetime

   # Charger données complètes
   df = pd.read_csv("data/comptage-velo-donnees-compteurs.csv", sep=";")
   df['date'] = pd.to_datetime(df['Date et heure de comptage'])

   # Split temporel : avant/après 2025-09-01
   cutoff_date = datetime(2025, 9, 1)

   df_reference = df[df['date'] < cutoff_date]  # Pour train/test
   df_current = df[df['date'] >= cutoff_date]   # Pour drift detection

   # Sauvegarder
   df_reference.to_csv("data/reference_data.csv", index=False)
   df_current.to_csv("data/current_data.csv", index=False)
   ```

3. **Versionner avec DVC**
   ```bash
   dvc add data/reference_data.csv
   dvc add data/current_data.csv
   dvc push

   git add data/reference_data.csv.dvc data/current_data.csv.dvc .dvc/config
   git commit -m "feat: add DVC data versioning with temporal split"
   ```

4. **Fichier `.dvc/config`**
   ```ini
   [core]
       remote = gcs_storage
   ['remote "gcs_storage"']
       url = gs://df_traffic_cyclist1/dvc-storage
       credentialpath = ./gcp.json
   ```

**Livrables** :
```
data/
├── reference_data.csv.dvc         # Pointer DVC
├── current_data.csv.dvc           # Pointer DVC
├── .gitignore                     # data/*.csv (sauf .dvc)
scripts/
├── split_data_temporal.py         # Script de split
.dvc/
├── config                         # Config DVC + remote GCS
└── .gitignore
```

**Intégration avec le pipeline** :
- **Entraînement** : utilise `dvc pull data/reference_data.csv.dvc`
- **Drift detection** : utilise `dvc pull data/current_data.csv.dvc`

---

#### **2.2 Tests unitaires + CI** (`feat/mlops-tests-ci`)

**Objectifs** :
- Suite de tests pytest
- GitHub Actions CI
- Coverage >80%

**Livrables** :
```
tests/
├── test_pipelines.py          # Tests RFPipeline, NNPipeline
├── test_preprocessing.py      # Tests transformers
├── test_api_regmodel.py       # Tests endpoints FastAPI
├── conftest.py                # Fixtures partagées
pytest.ini
.github/
└── workflows/
    └── ci.yml                 # GitHub Actions
```

**GitHub Actions CI** :
```yaml
# .github/workflows/ci.yml
name: MLOps CI

on:
  push:
    branches: [ feat/*, master ]
  pull_request:
    branches: [ master ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov httpx

      - name: Run tests
        run: pytest tests/ -v --cov=src --cov=app --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

**Tests clés** :
```python
# tests/test_pipelines.py
def test_rf_pipeline_fit_predict():
    X, y = load_sample_data()
    rf = RFPipeline()
    rf.fit(X, y)
    preds = rf.predict(X[:10])
    assert len(preds) == 10

# tests/test_api_regmodel.py
from fastapi.testclient import TestClient
from backend.regmodel.app.fastapi_app import app

client = TestClient(app)

def test_predict_endpoint():
    response = client.post("/predict", json={
        "records": [...],
        "model_type": "rf",
        "metric": "r2"
    })
    assert response.status_code == 200
    assert "predictions" in response.json()
```

---

### **Phase 3 : Orchestration Airflow + Réentraînement intelligent** (`feat/mlops-airflow-pipeline`)

**Objectifs** :
- Pipeline automatisé end-to-end
- Logique de réentraînement conditionnel
- Scheduling hebdomadaire

**DAG Airflow avec branchement** :

```python
# dags/ml_pipeline_dag.py

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'mlops-team',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'bike_traffic_ml_pipeline',
    default_args=default_args,
    schedule_interval='@weekly',
    start_date=datetime(2024, 10, 1),
    catchup=False
) as dag:

    # 1️⃣ Récupération données current (DVC)
    def fetch_current_data(**context):
        import subprocess
        subprocess.run(['dvc', 'pull', 'data/current_data.csv.dvc'])
        context['ti'].xcom_push(key='current_path', value='data/current_data.csv')

    fetch_data = PythonOperator(
        task_id='fetch_current_data',
        python_callable=fetch_current_data
    )

    # 2️⃣ Prédiction sur données current
    def predict_on_current(**context):
        from app.model_registry_summary import get_best_model_from_summary
        import pandas as pd

        model = get_best_model_from_summary(
            model_type="rf",
            metric="r2",
            summary_path="gs://df_traffic_cyclist1/models/summary.json"
        )

        df = pd.read_csv(context['ti'].xcom_pull(key='current_path'))
        df['prediction'] = model.predict(df)
        df.to_csv('/tmp/predictions.csv', index=False)
        context['ti'].xcom_push(key='predictions_path', value='/tmp/predictions.csv')

    predict = PythonOperator(
        task_id='predict_on_current',
        python_callable=predict_on_current
    )

    # 3️⃣ Évaluation métriques + décision
    def evaluate_and_decide(**context):
        import pandas as pd
        import numpy as np
        from sklearn.metrics import mean_squared_error, r2_score

        df = pd.read_csv(context['ti'].xcom_pull(key='predictions_path'))
        y_true = df['Comptage horaire']
        y_pred = df['prediction']

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        # Seuils de dégradation
        RMSE_THRESHOLD = 60.0
        R2_THRESHOLD = 0.65

        context['ti'].xcom_push(key='metrics', value={'rmse': rmse, 'r2': r2})

        if rmse > RMSE_THRESHOLD or r2 < R2_THRESHOLD:
            print(f"⚠️ Métriques dégradées : RMSE={rmse:.2f}, R²={r2:.4f}")
            return 'retrain_models'
        else:
            print(f"✅ Métriques OK : RMSE={rmse:.2f}, R²={r2:.4f}")
            return 'skip_training'

    evaluate = BranchPythonOperator(
        task_id='evaluate_metrics',
        python_callable=evaluate_and_decide
    )

    # 4️⃣ Réentraînement (si dégradation)
    def retrain_models(**context):
        import subprocess

        # Pull reference data (DVC)
        subprocess.run(['dvc', 'pull', 'data/reference_data.csv.dvc'])

        # Lancer train.py
        result = subprocess.run([
            'python', 'src/train.py',
            '--env', 'prod'
        ], capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"Training failed: {result.stderr}")

        print("✅ Réentraînement terminé")

    retrain = PythonOperator(
        task_id='retrain_models',
        python_callable=retrain_models
    )

    # 5️⃣ Pas de réentraînement (si OK)
    skip = EmptyOperator(task_id='skip_training')

    # 6️⃣ Refresh API (après retrain)
    def refresh_api(**context):
        import requests
        response = requests.get(
            "https://regmodel-api-467498471756.europe-west1.run.app/refresh_model"
        )
        if response.status_code != 200:
            raise Exception(f"API refresh failed: {response.text}")

    refresh = PythonOperator(
        task_id='refresh_api',
        python_callable=refresh_api
    )

    # 7️⃣ Fin du pipeline
    end = EmptyOperator(
        task_id='pipeline_complete',
        trigger_rule='none_failed_min_one_success'
    )

    # === FLUX ===
    fetch_data >> predict >> evaluate
    evaluate >> retrain >> refresh >> end
    evaluate >> skip >> end
```

**Visualisation du DAG** :
```
[Fetch Current Data] → [Predict] → [Evaluate Metrics]
                                          ├─→ [Retrain] → [Refresh API] → [End]
                                          └─→ [Skip] ──────────────────────→ [End]
```

**Docker Compose Airflow** :
```yaml
# docker-compose.yaml (ajout)
services:
  postgres-airflow:
    image: postgres:15
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    volumes:
      - postgres_airflow_data:/var/lib/postgresql/data

  airflow-webserver:
    image: apache/airflow:2.8.0-python3.12
    environment:
      - AIRFLOW__CORE__EXECUTOR=LocalExecutor
      - AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres-airflow/airflow
    volumes:
      - ./dags:/opt/airflow/dags
      - ./src:/opt/airflow/src
      - ./app:/opt/airflow/app
      - ./scripts:/opt/airflow/scripts
      - ./gcp.json:/opt/airflow/gcp.json
    ports:
      - "8081:8080"
    command: webserver
    depends_on:
      - postgres-airflow

  airflow-scheduler:
    image: apache/airflow:2.8.0-python3.12
    environment:
      - AIRFLOW__CORE__EXECUTOR=LocalExecutor
      - AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres-airflow/airflow
    volumes:
      - ./dags:/opt/airflow/dags
      - ./src:/opt/airflow/src
      - ./app:/opt/airflow/app
      - ./scripts:/opt/airflow/scripts
    command: scheduler
    depends_on:
      - postgres-airflow

volumes:
  postgres_airflow_data:
```

---

### **Phase 4 : Monitoring Production** (`feat/mlops-monitoring`)

#### **4.1 Métriques API (Prometheus + Grafana)**

**Architecture** :
- **Prometheus** : collecte métriques (TSDB local, pas Postgres)
- **Grafana** : dashboards

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

**Instrumentation API** :
```python
# backend/regmodel/app/fastapi_app.py
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
from starlette.middleware.base import BaseHTTPMiddleware
import time

# Métriques custom
predictions_total = Counter('predictions_total', 'Total predictions', ['model_type'])
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency', ['model_type'])
active_models = Gauge('active_models_count', 'Cached models count')

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

**Dashboard Grafana** :
- Requêtes/sec : `rate(predictions_total[5m])`
- Latence p50/p95/p99 : `histogram_quantile(0.95, prediction_latency_seconds)`
- Taux erreur : `rate(http_requests_total{status=~"5.."}[5m])`

---

#### **4.2 Détection de dérive (Evidently)**

**Implémentation via script Python** :

```python
# monitoring/drift_detector.py
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.metrics import ColumnDriftMetric
import pandas as pd
from google.cloud import storage
from datetime import datetime

def detect_drift(reference_csv: str, current_csv: str, output_html: str):
    """
    Compare reference (train) vs current (prod) data
    """
    ref_df = pd.read_csv(reference_csv)
    curr_df = pd.read_csv(current_csv)

    # Colonnes à surveiller
    feature_cols = ['heure', 'jour_semaine', 'latitude', 'longitude', 'mois']

    report = Report(metrics=[
        DataDriftPreset(columns=feature_cols),
        RegressionPreset(),
        ColumnDriftMetric(column_name='heure'),
        ColumnDriftMetric(column_name='latitude'),
    ])

    report.run(reference_data=ref_df, current_data=curr_df)
    report.save_html(output_html)

    # Upload GCS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gcs_path = f"monitoring/drift_report_{timestamp}.html"

    client = storage.Client()
    bucket = client.bucket("df_traffic_cyclist1")
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(output_html)

    print(f"✅ Rapport drift : gs://df_traffic_cyclist1/{gcs_path}")

    # Retourner si drift détecté
    drift_info = report.as_dict()['metrics'][0]['result']
    return drift_info.get('drift_detected', False)

if __name__ == "__main__":
    import sys
    detect_drift(sys.argv[1], sys.argv[2], sys.argv[3])
```

**Intégration DAG Airflow** :
```python
# dags/ml_pipeline_dag.py (ajout)

def check_data_drift(**context):
    from monitoring.drift_detector import detect_drift
    import subprocess

    # Pull reference data
    subprocess.run(['dvc', 'pull', 'data/reference_data.csv.dvc'])

    reference = 'data/reference_data.csv'
    current = context['ti'].xcom_pull(key='current_path')
    output = '/tmp/drift_report.html'

    drift_detected = detect_drift(reference, current, output)

    if drift_detected:
        print("⚠️ DATA DRIFT DÉTECTÉ")
        # TODO: send_slack_alert()

    context['ti'].xcom_push(key='drift_detected', value=drift_detected)

drift_task = PythonOperator(
    task_id='check_drift',
    python_callable=check_data_drift
)

# Ajout au flux
fetch_data >> drift_task >> predict >> evaluate
```

---

#### **4.3 Sécurité API**

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
    # Store model_type for Prometheus
    request.state.model_type = data.model_type

    model = get_cached_model(data.model_type, data.metric)
    y_pred = model.predict_clean(pd.DataFrame(data.records))
    return {"predictions": y_pred.tolist()}
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

```
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

```
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
