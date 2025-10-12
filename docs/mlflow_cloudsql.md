# MLflow Cloud SQL Setup

## Overview

MLflow backend store migrated from local filesystem to **Cloud SQL PostgreSQL** for:

- ✅ Persistent metadata (experiments, runs, metrics)
- ✅ Shared across environments
- ✅ Automatic backups
- ✅ Scalable and reliable

## Infrastructure

**Instance:** `mlflow-metadata`
**Region:** `europe-west3` (Frankfurt)
**Database:** `mlflow`
**User:** `mlflow_user`

## Setup Steps

### 1. Create Cloud SQL Instance (DONE)

```bash
gcloud sql instances create mlflow-metadata \
    --database-version=POSTGRES_14 \
    --tier=db-f1-micro \
    --region=europe-west3 \
    --storage-type=SSD \
    --storage-size=10GB \
    --backup \
    --backup-start-time=03:00
```

**Status:** Check with `gcloud sql instances list`

### 2. Create Database and User

Once instance is `RUNNABLE`, run:

```bash
./scripts/setup_mlflow_db.sh
```

This will:

- Create `mlflow` database
- Create `mlflow_user` with strong password
- Save password to Secret Manager
- Output connection string

### 3. Update Docker Compose

The `docker-compose.yaml` will be updated to use Cloud SQL Proxy:

```yaml
mlflow:
  image: ghcr.io/mlflow/mlflow:v2.22.0
  depends_on:
    - cloud-sql-proxy
  environment:
    - MLFLOW_BACKEND_STORE_URI=postgresql://mlflow_user:PASSWORD@cloud-sql-proxy:5432/mlflow
    - GOOGLE_APPLICATION_CREDENTIALS=/mlflow/gcp.json
  entrypoint: >
    /bin/bash -c "pip install google-cloud-storage psycopg2-binary &&
    mlflow server --backend-store-uri ${MLFLOW_BACKEND_STORE_URI}
                  --default-artifact-root gs://bucket/mlflow-artifacts
                  --host 0.0.0.0 --port 5000"

cloud-sql-proxy:
  image: gcr.io/cloud-sql-connectors/cloud-sql-proxy:latest
  command:
    - "--private-ip"
    - "datascientest-460618:europe-west3:mlflow-metadata"
  volumes:
    - ./gcp.json:/config/gcp.json:ro
  environment:
    - GOOGLE_APPLICATION_CREDENTIALS=/config/gcp.json
```

### 4. Test Connection

```bash
# Get connection name
CONNECTION_NAME=$(gcloud sql instances describe mlflow-metadata --format='value(connectionName)')

# Test with Cloud SQL Proxy
cloud-sql-proxy ${CONNECTION_NAME} &
psql "host=127.0.0.1 port=5432 dbname=mlflow user=mlflow_user"
```

### 5. Migrate Existing Data (Optional)

If you have existing runs in `mlruns_dev/`:

```bash
# Export existing experiments
mlflow experiments search --view all > experiments_backup.json

# After connecting to Cloud SQL, experiments will be recreated
# Artifacts remain in GCS (no migration needed)
```

## Secrets Management

Store password in Secret Manager:

```bash
# Set password secret
gcloud secrets create mlflow-db-password \
    --data-file=- <<< "YOUR_PASSWORD"

# Grant access to service account
gcloud secrets add-iam-policy-binding mlflow-db-password \
    --member="serviceAccount:mlflow-trainer@datascientest-460618.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

## Costs

**db-f1-micro:**

- ~$7/month (shared CPU, 0.6GB RAM)
- Includes 10GB SSD storage
- Daily backups

**Optimization:**

- Consider `db-g1-small` if performance issues (~$25/month)
- Delete instance when not training to save costs

## Troubleshooting

### Connection refused

```bash
# Check instance status
gcloud sql instances describe mlflow-metadata --format="value(state)"

# Check Cloud SQL Proxy logs
docker logs cloud-sql-proxy
```

### Authentication errors

```bash
# Verify service account permissions
gcloud projects get-iam-policy datascientest-460618 \
    --flatten="bindings[].members" \
    --filter="bindings.members:mlflow-trainer@"
```

Required roles:

- `roles/cloudsql.client` (connect via proxy)
- `roles/secretmanager.secretAccessor` (read password)

## Rollback

To revert to local filesystem:

```bash
# docker-compose.yaml
--backend-store-uri file:///mlflow/mlruns

# Remove cloud-sql-proxy service
# Restart MLflow
docker compose up mlflow -d
```

## Related Docs

- [Cloud SQL Best Practices](https://cloud.google.com/sql/docs/postgres/best-practices)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html#backend-stores)
