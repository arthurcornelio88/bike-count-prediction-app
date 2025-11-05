# 🔐 Secrets Management

**Objectif** : Centraliser la configuration des secrets pour DEV et PROD

---

## Architecture

| Environment | Method | Storage | Access |
|-------------|--------|---------|--------|
| **DEV** | `.env` files | Local filesystem | Direct file read |
| **PROD** | Google Secret Manager | GCP encrypted | Service Account |

---

## Secrets Liste

### Production Secrets (GCP Secret Manager)

| Secret ID | Description | Exemple |
|-----------|-------------|---------|
| `gcs-bucket-bike` | GCS bucket name | `df_traffic_cyclist1` |
| `bq-project-bike` | BigQuery project ID | `datascientest-460618` |
| `bq-raw-dataset-bike` | Raw data dataset | `bike_traffic_raw` |
| `bq-predict-dataset-bike` | Predictions dataset | `bike_traffic_predictions` |
| `bq-location` | BigQuery location | `europe-west1` |
| `prod-bike-api-url` | RegModel API URL | `https://regmodel-api-xxx.run.app` |
| `bike-api-key-secret` | API authentication key | `[generated-32-chars]` |
| `train-data-path` | Training data GCS path | `gs://bucket/data/train_baseline.csv` |
| `test-data-path` | Test data GCS path | `gs://bucket/data/test_baseline.csv` |
| **MLflow Cloud SQL** | | |
| `mlflow-db-host` | Cloud SQL instance connection | `datascientest-460618:europe-west3:mlflow-metadata` |
| `mlflow-db-name` | Database name | `mlflow` |
| `mlflow-db-user` | Database user | `mlflow_user` |
| `mlflow-db-password` | Database password | `[generated-password]` |

---

## Setup Instructions

### DEV Environment

#### 1. Airflow `.env.airflow`

```bash
# .env.airflow (root directory)
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

# Credentials
GOOGLE_APPLICATION_CREDENTIALS=/opt/airflow/gcp.json

# Airflow User
_AIRFLOW_WWW_USER_USERNAME=admin
_AIRFLOW_WWW_USER_PASSWORD=admin

# Airflow UID/GID (match host user for volume permissions)
AIRFLOW_UID=50000
AIRFLOW_GID=50000
```

#### 2. RegModel Backend `.env`

```bash
# backend/regmodel/.env
ENV=DEV
API_KEY_SECRET=dev-key-unsafe
GOOGLE_APPLICATION_CREDENTIALS=/app/gcp.json

# Training Data Paths
TRAIN_DATA_PATH=data/train_baseline.csv
TEST_DATA_PATH=data/test_baseline.csv

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# MLflow Cloud SQL (will be configured when instance is ready)
# MLFLOW_DB_HOST=datascientest-460618:europe-west3:mlflow-metadata
# MLFLOW_DB_NAME=mlflow
# MLFLOW_DB_USER=mlflow_user
# MLFLOW_DB_PASSWORD=<from-secret-manager>
```

**Note:** For PROD, use GCS paths:

```bash
TRAIN_DATA_PATH=gs://df_traffic_cyclist1/data/train_baseline.csv
TEST_DATA_PATH=gs://df_traffic_cyclist1/data/test_baseline.csv
```

#### 3. Streamlit App `secrets.toml`

```toml
# secrets.toml (root directory)
[general]
api_url = "http://localhost:8000"
api_key = "dev-key-unsafe"
```

---

### PROD Environment

#### 1. Create Secrets via `gcloud` CLI

```bash
# Enable Secret Manager API
gcloud services enable secretmanager.googleapis.com \
  --project=datascientest-460618

# Create all secrets
echo -n "df_traffic_cyclist1" | \
  gcloud secrets create gcs-bucket-bike \
  --data-file=- --replication-policy="automatic" \
  --project=datascientest-460618

echo -n "datascientest-460618" | \
  gcloud secrets create bq-project-bike \
  --data-file=- --replication-policy="automatic" \
  --project=datascientest-460618

echo -n "bike_traffic_raw" | \
  gcloud secrets create bq-raw-dataset-bike \
  --data-file=- --replication-policy="automatic" \
  --project=datascientest-460618

echo -n "bike_traffic_predictions" | \
  gcloud secrets create bq-predict-dataset-bike \
  --data-file=- --replication-policy="automatic" \
  --project=datascientest-460618

echo -n "europe-west1" | \
  gcloud secrets create bq-location \
  --data-file=- --replication-policy="automatic" \
  --project=datascientest-460618

echo -n "https://regmodel-api-467498471756.europe-west1.run.app" | \
  gcloud secrets create prod-bike-api-url \
  --data-file=- --replication-policy="automatic" \
  --project=datascientest-460618

# Generate secure API key
echo -n "$(openssl rand -base64 32)" | \
  gcloud secrets create bike-api-key-secret \
  --data-file=- --replication-policy="automatic" \
  --project=datascientest-460618

# Training data paths (GCS)
echo -n "gs://df_traffic_cyclist1/data/train_baseline.csv" | \
  gcloud secrets create train-data-path \
  --data-file=- --replication-policy="automatic" \
  --project=datascientest-460618

echo -n "gs://df_traffic_cyclist1/data/test_baseline.csv" | \
  gcloud secrets create test-data-path \
  --data-file=- --replication-policy="automatic" \
  --project=datascientest-460618

# MLflow Cloud SQL Configuration
echo -n "datascientest-460618:europe-west3:mlflow-metadata" | \
  gcloud secrets create mlflow-db-host \
  --data-file=- --replication-policy="automatic" \
  --project=datascientest-460618

echo -n "mlflow" | \
  gcloud secrets create mlflow-db-name \
  --data-file=- --replication-policy="automatic" \
  --project=datascientest-460618

echo -n "mlflow_user" | \
  gcloud secrets create mlflow-db-user \
  --data-file=- --replication-policy="automatic" \
  --project=datascientest-460618

# Password will be generated by setup_mlflow_db.sh script
# Run: ./scripts/setup_mlflow_db.sh
# Then save output password:
# echo -n "PASSWORD_FROM_SCRIPT" | \
#   gcloud secrets create mlflow-db-password \
#   --data-file=- --replication-policy="automatic" \
#   --project=datascientest-460618
```

#### 2. Grant Access to Service Account

```bash
SERVICE_ACCOUNT="467498471756-compute@developer.gserviceaccount.com"

for SECRET_ID in gcs-bucket-bike bq-project-bike \
  bq-raw-dataset-bike bq-predict-dataset-bike \
  bq-location prod-bike-api-url bike-api-key-secret \
  train-data-path test-data-path \
  mlflow-db-host mlflow-db-name mlflow-db-user mlflow-db-password
do
  gcloud secrets add-iam-policy-binding $SECRET_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/secretmanager.secretAccessor" \
    --project=datascientest-460618
done
```

#### 3. Verify Access

```bash
# List all secrets
gcloud secrets list --project=datascientest-460618

# Test access to one secret
gcloud secrets versions access latest \
  --secret="gcs-bucket-bike" \
  --project=datascientest-460618
```

---

## Code Usage

### Python (Airflow DAGs)

```python
from utils.env_config import get_env_config

# Automatically loads DEV (.env) or PROD (Secret Manager)
config = get_env_config()

print(config["GCS_BUCKET"])      # df_traffic_cyclist1
print(config["BQ_PROJECT"])      # datascientest-460618
print(config["API_URL"])         # DEV: http://localhost:8000 | PROD: https://...
```

### Python (Direct Secret Manager)

```python
from google.cloud import secretmanager

def get_secret(secret_id: str, project_id: str = "datascientest-460618") -> str:
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

# Usage
bucket_name = get_secret("gcs-bucket-bike")
```

---

## Validation Scripts

### Test All Secrets (Local)

```python
# scripts/test_secrets.py
from google.cloud import secretmanager

def test_all_secrets():
    project_id = "datascientest-460618"
    client = secretmanager.SecretManagerServiceClient()

    secrets = [
        "gcs-bucket-bike",
        "bq-project-bike",
        "bq-raw-dataset-bike",
        "bq-predict-dataset-bike",
        "bq-location",
        "prod-bike-api-url",
        "bike-api-key-secret",
        "train-data-path",
        "test-data-path",
        "mlflow-db-host",
        "mlflow-db-name",
        "mlflow-db-user",
        "mlflow-db-password"
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
    test_all_secrets()
```

Run test:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=./gcp.json
python scripts/test_secrets.py
```

---

## Security Best Practices

### ✅ DO

- Use Secret Manager in PROD (encrypted at rest)
- Rotate API keys quarterly
- Grant minimal IAM permissions (`secretAccessor` only)
- Use service accounts (never user credentials in production)
- Add secrets to `.gitignore` (`.env*`, `secrets.toml`, `*.json`)

### ❌ DON'T

- Commit secrets to Git
- Hardcode secrets in code
- Share secrets via Slack/email
- Use same secrets for DEV and PROD
- Give broad IAM roles (`owner`, `editor`)

---

## Troubleshooting

### Error: "Permission denied on secret"

```bash
# Check IAM permissions
gcloud secrets get-iam-policy gcs-bucket-bike \
  --project=datascientest-460618

# Expected output should include your service account
# with role: roles/secretmanager.secretAccessor
```

**Fix:**

```bash
gcloud secrets add-iam-policy-binding gcs-bucket-bike \
  --member="serviceAccount:YOUR-SA@PROJECT.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

### Error: "Secret not found"

```bash
# Verify secret exists
gcloud secrets describe gcs-bucket-bike \
  --project=datascientest-460618
```

**Fix:** Create the secret (see setup instructions above)

### Error: "Application Default Credentials not found"

```bash
# Set credentials path
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/gcp.json

# Or authenticate locally
gcloud auth application-default login
```

---

## Checklist

- [ ] Secret Manager API enabled (`secretmanager.googleapis.com`)
- [ ] 13 secrets created in GCP Secret Manager (including train/test paths + MLflow DB)
- [ ] Service Account has `secretAccessor` role on all secrets
- [ ] `.env.airflow` configured for DEV
- [ ] `backend/regmodel/.env` configured for DEV
- [ ] `secrets.toml` configured for Streamlit
- [ ] `gcp.json` service account key downloaded
- [ ] Secrets validation script passes
- [ ] All secret files in `.gitignore`
- [ ] Cloud SQL instance `mlflow-metadata` created
- [ ] MLflow database and user configured via `setup_mlflow_db.sh`

---

**Next:** See [bigquery_setup.md](./bigquery_setup.md) for BigQuery configuration
