# 🗄️ BigQuery Setup

**Objectif** : Configurer BigQuery datasets et uploader les données baseline

> **Note:** Pour la configuration des secrets, voir [secrets.md](./secrets.md)

---

## 📋 Table des matières

1. [Architecture BigQuery](#architecture-bigquery)
2. [Création des Datasets BigQuery](#création-des-datasets-bigquery)
3. [Upload Baseline Data](#upload-baseline-data)
4. [Schémas des tables](#schémas-des-tables)
5. [Validation](#validation)

---

## Architecture BigQuery

**3 Datasets pour traçabilité complète** :

```text
datascientest-460618
├── bike_traffic_raw           # Données brutes quotidiennes
│   └── daily_YYYYMMDD        # Tables par jour
│
├── bike_traffic_predictions   # Prédictions quotidiennes
│   └── daily_YYYYMMDD        # Prédictions + métriques
│
└── monitoring_audit           # Logs de monitoring
    └── logs                   # Audit (drift, fine-tuning, métriques)
```

---

## Création des Datasets BigQuery

### 1. Via Console GCP

1. Aller sur [BigQuery Console](https://console.cloud.google.com/bigquery)
2. Sélectionner le projet `datascientest-460618`
3. Créer les 3 datasets :

#### Dataset 1 : `bike_traffic_raw`

```text
Dataset ID: bike_traffic_raw
Location: europe-west1 (Belgium)
Default table expiration: Never
```

#### Dataset 2 : `bike_traffic_predictions`

```text
Dataset ID: bike_traffic_predictions
Location: europe-west1 (Belgium)
Default table expiration: Never
```

#### Dataset 3 : `monitoring_audit`

```text
Dataset ID: monitoring_audit
Location: europe-west1 (Belgium)
Default table expiration: Never
```

### 2. Via `bq` CLI

```bash
# Authentification
gcloud auth login
gcloud config set project datascientest-460618

# Créer les datasets
bq mk --location=europe-west1 --dataset datascientest-460618:bike_traffic_raw
bq mk --location=europe-west1 --dataset datascientest-460618:bike_traffic_predictions
bq mk --location=europe-west1 --dataset datascientest-460618:monitoring_audit
```

### 3. Via Python (automatique dans DAGs)

Les DAGs créeront automatiquement les datasets et tables si nécessaires grâce à la fonction :

```python
from utils.bike_helpers import create_bq_dataset_if_not_exists, create_monitoring_table_if_needed

# Créer les datasets
create_bq_dataset_if_not_exists("datascientest-460618", "bike_traffic_raw")
create_bq_dataset_if_not_exists("datascientest-460618", "bike_traffic_predictions")

# Créer la table de monitoring avec son schéma
create_monitoring_table_if_needed("datascientest-460618")
```

---

## Upload Baseline Data

The champion model needs baseline data uploaded to GCS for training and evaluation.

### Upload train_baseline.csv

```bash
# Upload train baseline (~724k records, ~200MB)
gsutil -m cp data/train_baseline.csv \
  gs://df_traffic_cyclist1/data/train_baseline.csv

# Verify
gsutil ls -lh gs://df_traffic_cyclist1/data/
```

### Upload test_baseline.csv

```bash
# Upload test baseline (~181k records, ~50MB)
gsutil -m cp data/test_baseline.csv \
  gs://df_traffic_cyclist1/data/test_baseline.csv
```

### Environment Variables

The training script uses these paths:

```bash
# Set in .env.airflow or as environment variables
TRAIN_DATA_PATH=gs://df_traffic_cyclist1/data/train_baseline.csv
TEST_DATA_PATH=gs://df_traffic_cyclist1/data/test_baseline.csv
```

**Note:** Daily API fetch populates BigQuery for production predictions, not for training.

---

## Schémas des tables

### Table `bike_traffic_raw.daily_YYYYMMDD`

```sql
CREATE TABLE `datascientest-460618.bike_traffic_raw.daily_20251011` (
  comptage_horaire INT64,
  date_et_heure_de_comptage TIMESTAMP,
  identifiant_du_compteur STRING,
  nom_du_compteur STRING,
  coordonnees_geographiques STRING,
  ingestion_ts TIMESTAMP
);
```

### Table `bike_traffic_predictions.daily_YYYYMMDD`

```sql
CREATE TABLE `datascientest-460618.bike_traffic_predictions.daily_20251011` (
  comptage_horaire INT64,
  prediction FLOAT64,
  model_type STRING,
  model_version STRING,
  prediction_ts TIMESTAMP,
  identifiant_du_compteur STRING
);
```

### Table `monitoring_audit.logs`

```sql
CREATE TABLE `datascientest-460618.monitoring_audit.logs` (
  timestamp TIMESTAMP,
  drift_detected BOOL,
  rmse FLOAT64,
  r2 FLOAT64,
  fine_tune_triggered BOOL,
  fine_tune_success BOOL,
  model_improvement FLOAT64,
  env STRING,
  error_message STRING
);
```

---

## Validation

### Verify Datasets

```bash
# List all datasets
bq ls --project_id=datascientest-460618

# Expected: bike_traffic_raw, bike_traffic_predictions, monitoring_audit
```

### Verify GCS Uploads

```bash
# List uploaded files
gsutil ls -lh gs://df_traffic_cyclist1/data/

# Expected output:
# ~200MB  train_baseline.csv
# ~50MB   test_baseline.csv
```

### Test BigQuery Write

```python
# Quick test
import pandas as pd

df = pd.DataFrame({"test": [1, 2, 3]})
df.to_gbq(
    "bike_traffic_raw.test_table",
    project_id="datascientest-460618",
    if_exists="replace"
)
print("✅ BigQuery write successful")
```

---

## Checklist

- [ ] 3 datasets created (`bike_traffic_raw`, `bike_traffic_predictions`, `monitoring_audit`)
- [ ] `train_baseline.csv` uploaded to GCS
- [ ] `test_baseline.csv` uploaded to GCS
- [ ] Environment variables `TRAIN_DATA_PATH` and `TEST_DATA_PATH` set
- [ ] GCS bucket accessible (test with `gsutil ls`)
- [ ] BigQuery write permissions verified
- [ ] Secrets configured (see [secrets.md](./secrets.md))

---

**Next Steps:**

- Configure secrets → [secrets.md](./secrets.md)
- Review training strategy → [training_strategy.md](./training_strategy.md)
- Deploy Airflow DAGs → [architecture.md](./architecture.md)
