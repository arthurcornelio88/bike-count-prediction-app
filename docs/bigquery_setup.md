# 🗄️ BigQuery Setup & Secrets Configuration

**Objectif** : Configurer BigQuery et Google Secret Manager pour le pipeline Airflow MLOps

---

## 📋 Table des matières

1. [Architecture BigQuery](#architecture-bigquery)
2. [Création des Datasets BigQuery](#création-des-datasets-bigquery)
3. [Configuration Secret Manager](#configuration-secret-manager)
4. [Variables d'environnement locales (DEV)](#variables-denvironnement-locales-dev)
5. [Validation de la configuration](#validation-de-la-configuration)

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

## Upload Baseline Data to GCS

The DAGs need access to baseline data (train_baseline.csv) for champion model training and test_baseline.csv for drift detection.

### Upload train_baseline.csv to GCS

```bash
# Upload train baseline (~724k records) to GCS
gsutil -m cp data/train_baseline.csv gs://<your-bucket>/data/train_baseline.csv

# Verify upload
gsutil ls -lh gs://<your-bucket>/data/

# Expected output:
# ~200MB  gs://<your-bucket>/data/train_baseline.csv
```

**Note:** This is a one-time upload. The champion model will be trained on this baseline.
Daily API fetch will populate BigQuery for production predictions and weekly drift detection.

---

## Configuration Secret Manager

### Secrets requis pour PROD

| Secret ID | Description | Exemple de valeur |
|-----------|-------------|-------------------|
| `gcs-bucket-bike` | Nom du bucket GCS | `df_traffic_cyclist1` |
| `bq-project-bike` | ID du projet BigQuery | `datascientest-460618` |
| `bq-raw-dataset-bike` | Dataset données brutes | `bike_traffic_raw` |
| `bq-predict-dataset-bike` | Dataset prédictions | `bike_traffic_predictions` |
| `bq-location` | Location BigQuery | `europe-west1` |
| `prod-bike-api-url` | URL de l'API RegModel | `https://regmodel-api-467498471756.europe-west1.run.app` |
| `bike-api-key-secret` | API Key pour sécurité | `super-secret-prod-key-2024` |

### Création des secrets via Console GCP

1. Aller sur [Secret Manager](https://console.cloud.google.com/security/secret-manager)
2. Cliquer sur **"CREATE SECRET"**
3. Pour chaque secret :
   - **Name** : utiliser exactement le `Secret ID` du tableau ci-dessus
   - **Secret value** : entrer la valeur correspondante
   - **Regions** : Automatic
   - Cliquer sur **CREATE**

### Création des secrets via `gcloud` CLI

```bash
# Activer Secret Manager API
gcloud services enable secretmanager.googleapis.com --project=datascientest-460618

# Créer les secrets
echo -n "df_traffic_cyclist1" | gcloud secrets create gcs-bucket-bike \
  --data-file=- \
  --replication-policy="automatic" \
  --project=datascientest-460618

echo -n "datascientest-460618" | gcloud secrets create bq-project-bike \
  --data-file=- \
  --replication-policy="automatic" \
  --project=datascientest-460618

echo -n "bike_traffic_raw" | gcloud secrets create bq-raw-dataset-bike \
  --data-file=- \
  --replication-policy="automatic" \
  --project=datascientest-460618

echo -n "bike_traffic_predictions" | gcloud secrets create bq-predict-dataset-bike \
  --data-file=- \
  --replication-policy="automatic" \
  --project=datascientest-460618

echo -n "europe-west1" | gcloud secrets create bq-location \
  --data-file=- \
  --replication-policy="automatic" \
  --project=datascientest-460618

echo -n "https://regmodel-api-467498471756.europe-west1.run.app" | gcloud secrets create prod-bike-api-url \
  --data-file=- \
  --replication-policy="automatic" \
  --project=datascientest-460618

# Générer et créer l'API key (remplacer par une vraie clé forte)
echo -n "$(openssl rand -base64 32)" | gcloud secrets create bike-api-key-secret \
  --data-file=- \
  --replication-policy="automatic" \
  --project=datascientest-460618
```

### Donner accès aux secrets pour le Service Account

```bash
# Service Account utilisé par Cloud Run / Airflow
SERVICE_ACCOUNT="467498471756-compute@developer.gserviceaccount.com"

# Pour chaque secret, donner accès en lecture
for SECRET_ID in gcs-bucket-bike bq-project-bike bq-raw-dataset-bike bq-predict-dataset-bike bq-location prod-bike-api-url bike-api-key-secret
do
  gcloud secrets add-iam-policy-binding $SECRET_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/secretmanager.secretAccessor" \
    --project=datascientest-460618
done
```

---

## Variables d'environnement locales (DEV)

### Fichier `.env.airflow` (pour Docker Compose Airflow)

Créer le fichier `.env.airflow` à la racine du projet :

```bash
# Environment
ENV=DEV
GOOGLE_CLOUD_PROJECT=datascientest-460618

# BigQuery
BQ_PROJECT=datascientest-460618
BQ_RAW_DATASET=bike_traffic_raw
BQ_PREDICT_DATASET=bike_traffic_predictions
BQ_LOCATION=europe-west1

# GCS
GCS_BUCKET=df_traffic_cyclist1

# API
API_URL_DEV=http://regmodel-backend:8000
API_KEY_SECRET=dev-key-unsafe

# Google credentials
GOOGLE_APPLICATION_CREDENTIALS=/opt/airflow/gcp.json
```

### Fichier `.env` (pour RegModel backend)

Vérifier que `backend/regmodel/.env` contient :

```bash
ENV=DEV
API_KEY_SECRET=dev-key-unsafe
GOOGLE_APPLICATION_CREDENTIALS=/app/gcp.json
```

---

## Validation de la configuration

### Test 1 : Vérifier les secrets (PROD)

```python
# test_secrets.py
from google.cloud import secretmanager

def test_secrets():
    project_id = "datascientest-460618"
    client = secretmanager.SecretManagerServiceClient()

    secrets = [
        "gcs-bucket-bike",
        "bq-project-bike",
        "bq-raw-dataset-bike",
        "bq-predict-dataset-bike",
        "bq-location",
        "prod-bike-api-url",
        "bike-api-key-secret"
    ]

    for secret_id in secrets:
        try:
            name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
            response = client.access_secret_version(request={"name": name})
            value = response.payload.data.decode("UTF-8")
            print(f"✅ {secret_id}: {value[:20]}...")
        except Exception as e:
            print(f"❌ {secret_id}: {e}")

if __name__ == "__main__":
    test_secrets()
```

### Test 2 : Vérifier les datasets BigQuery

```bash
# Liste les datasets
bq ls --project_id=datascientest-460618

# Expected output:
#   bike_traffic_raw
#   bike_traffic_predictions
#   monitoring_audit
```

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

## Checklist finale

- [ ] 3 datasets BigQuery créés (`bike_traffic_raw`, `bike_traffic_predictions`, `monitoring_audit`)
- [ ] Train baseline uploaded to GCS (`gs://<your-bucket>/raw_data/train_baseline.csv`)
- [ ] Test baseline uploaded to GCS (`gs://<your-bucket>/raw_data/test_baseline.csv`)
- [ ] 7 secrets créés dans Secret Manager
- [ ] Service Account a accès aux secrets (rôle `secretmanager.secretAccessor`)
- [ ] Fichier `.env.airflow` créé avec variables DEV
- [ ] Fichier `gcp.json` présent et monté dans Docker
- [ ] Test de connexion BigQuery réussi
- [ ] Test de lecture Secret Manager réussi

---

**Prêt pour le déploiement des DAGs Airflow !** 🚀
