# MLflow + Cloud SQL (concise)

Ce document explique rapidement comment utiliser Cloud SQL (Postgres) comme backend
store MLflow et GCS comme stockage d'artefacts. L'objectif : une configuration simple,
partageable et reproductible pour l'équipe.

## En bref

- Backend metadata : Cloud SQL (Postgres) — centralisé, sauvegardé, partagé
- Artefacts : Google Cloud Storage — gs://df_traffic_cyclist1/mlflow-artifacts/
- Connexion : Cloud SQL Proxy (pas d'IP publique requise)

## Principales valeurs

- Instance : `mlflow-metadata` (region `europe-west3`)
- Base : `mlflow` — utilisateur `mlflow_user`
- Variable d'environnement : `MLFLOW_INSTANCE_CONNECTION` (format : PROJECT:REGION:INSTANCE)

## Étapes rapides

1) Créer l'instance Cloud SQL (exemple minimal) :

```bash
gcloud sql instances create mlflow-metadata \
  --database-version=POSTGRES_14 \
  --tier=db-f1-micro \
  --region=europe-west3 \
  --storage-size=10GB
```

2) Créer la base et l'utilisateur (enregistrer le mot de passe de façon sécurisée) :

```bash
gcloud sql databases create mlflow --instance=mlflow-metadata
PASSWORD=$(openssl rand -base64 32)
gcloud sql users create mlflow_user --instance=mlflow-metadata --password="$PASSWORD"
# stocker $PASSWORD dans Secret Manager ou un vault
```

3) Définir les variables d'environnement (ou `.env`) :

```bash
echo "MLFLOW_DB_USER=mlflow_user" >> .env
echo "MLFLOW_DB_PASSWORD=$PASSWORD" >> .env
echo "MLFLOW_DB_NAME=mlflow" >> .env
echo "MLFLOW_INSTANCE_CONNECTION=PROJECT:REGION:mlflow-metadata" >> .env
```

4) Donner le rôle `roles/cloudsql.client` au compte de service utilisé par le proxy :

```bash
gcloud projects add-iam-policy-binding <PROJECT_ID> \
  --member="serviceAccount:YOUR_SA@<PROJECT_ID>.iam.gserviceaccount.com" \
  --role="roles/cloudsql.client"
```

5) Exemple minimal `docker-compose` (seulement les points essentiels) :

```yaml
services:
  cloud-sql-proxy:
    image: gcr.io/cloud-sql-connectors/cloud-sql-proxy:latest
    volumes:
      - ./gcp.json:/config:ro
    env_file: [ .env ]

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.22.0
    env_file: [ .env ]
    environment:
      - MLFLOW_BACKEND_STORE_URI=postgresql://${MLFLOW_DB_USER}:${MLFLOW_DB_PASSWORD}@cloud-sql-proxy:5432/${MLFLOW_DB_NAME}
    depends_on: [ cloud-sql-proxy ]
```

6) Démarrer les services :

```bash
docker compose up -d cloud-sql-proxy mlflow
```

## Coûts et optimisations

- `db-f1-micro` : faible coût pour développement (~€5–10/mois)
- Monter en gamme (`db-g1-small`) si besoin de CPU/mémoire

## Revenir au stockage local

Si nécessaire, on peut repasser à un backend local :

```yaml
mlflow:
  command: mlflow server --backend-store-uri file:///mlflow/mlruns --default-artifact-root gs://df_traffic_cyclist1/mlflow-artifacts --host 0.0.0.0 --port 5000
```

## Liens utiles

- training_strategy.md — intégration MLflow et workflow
- secrets.md — gestion des comptes de service et secrets
