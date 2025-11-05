# Troubleshooting: Airflow Initialization Fixed

## Problem

After running `docker compose up -d`, Airflow services (webserver, scheduler,
worker) were crashing with:

```text
ERROR: You need to initialize the database. Please run `airflow db init`.
Make sure the command is run using Airflow version 2.10.0.
```

## Root Causes

### 1. **Variable d'environnement dépréciée** ❌

- **Problème**: Utilisation de `AIRFLOW__CORE__SQL_ALCHEMY_CONN` (déprécié dans Airflow 2.7+)
- **Solution**: Ajout de `AIRFLOW__DATABASE__SQL_ALCHEMY_CONN` (nouvelle variable)
- **Compromis**: Garder les deux pour compatibilité avec l'entrypoint

### 2. **Vérification de DB par l'entrypoint** ❌

- **Problème**: Chaque service Airflow vérifie la DB au démarrage via son entrypoint
- **Impact**: Même si `airflow-init` initialise la DB, les autres services crashent car l'entrypoint fait sa propre vérification
- **Solution**: Ajouter `_AIRFLOW_DB_MIGRATE=false` pour désactiver la vérification

### 3. **Race conditions** ❌

- **Problème**: `docker compose up -d` démarre tous les services en parallèle
- **Impact**: Les services Airflow démarrent avant que `airflow-init` termine l'initialisation
- **Solution**: Utiliser des healthchecks et depends_on avec conditions

## Solutions Implémentées

### 1. Configuration des Variables d'Environnement

**Avant:**

```yaml
environment:
  AIRFLOW__CORE__SQL_ALCHEMY_CONN: postgresql+psycopg2://...
```

**Après:**

```yaml
environment:
  AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://...  # Nouvelle variable
  AIRFLOW__CORE__SQL_ALCHEMY_CONN: postgresql+psycopg2://...      # Garde pour compatibilité
```

### 2. Désactivation de la Vérification DB dans l'Entrypoint

Ajouté pour **tous** les services Airflow (webserver, scheduler, worker, flower):

```yaml
environment:
  - _AIRFLOW_DB_MIGRATE=false  # Skip entrypoint DB check since airflow-init handles it
```

### 3. Service `airflow-init` Robuste

```yaml
airflow-init:
  entrypoint: /bin/bash
  command:
    - -c
    - |
      # Wait for PostgreSQL
      while ! nc -z postgres-airflow 5432; do sleep 1; done

      # Verify env var
      if [[ -z "${AIRFLOW__DATABASE__SQL_ALCHEMY_CONN:-}" ]]; then
        echo "ERROR: AIRFLOW__DATABASE__SQL_ALCHEMY_CONN is not set!"
        exit 1
      fi

      # Initialize & upgrade DB
      airflow db init
      airflow db upgrade

      # Create admin user
      airflow users create --username admin ...

      # Keep container alive for healthcheck
      while true; do
        echo "Airflow DB is ready" > /tmp/airflow-init-status
        sleep 5
      done

  healthcheck:
    test: ["CMD", "test", "-f", "/tmp/airflow-init-status"]
    interval: 5s
    retries: 20
    start_period: 30s
```

### 4. Depends_on avec Conditions de Santé

```yaml
airflow-webserver:
  depends_on:
    postgres-airflow:
      condition: service_healthy
    redis-airflow:
      condition: service_healthy
    airflow-init:
      condition: service_healthy  # ⬅️ CRUCIAL
    mlflow:
      condition: service_started
```

## Ordre de Démarrage Garanti

```text
1. Infrastructure
   ├── postgres-airflow (avec healthcheck)
   ├── redis-airflow (avec healthcheck)
   └── cloud-sql-proxy

2. MLflow & RegModel API
   (attendent que l'infrastructure soit healthy)

3. Airflow Init
   ├── Attend que postgres et redis soient healthy
   ├── Exécute airflow db init
   ├── Exécute airflow db upgrade
   ├── Crée l'utilisateur admin
   └── Devient "healthy" via healthcheck

4. Airflow Services
   ├── airflow-webserver  ┐
   ├── airflow-scheduler  ├─ Attendent que airflow-init soit healthy
   ├── airflow-worker     │  + _AIRFLOW_DB_MIGRATE=false
   └── flower             ┘

5. Monitoring (optionnel)
   ├── prometheus
   ├── grafana
   └── airflow-exporter
```

## Utilisation

### Option 1: Script Automatique (Recommandé)

```bash
# Sans monitoring
./scripts/start-all.sh

# Avec monitoring
./scripts/start-all.sh --with-monitoring
```

### Option 2: Docker Compose Direct

```bash
# Sans monitoring
docker compose up -d

# Avec monitoring
docker compose --profile monitoring up -d
```

### Option 3: Redémarrer Seulement Airflow

```bash
./scripts/restart-airflow.sh
```

## Vérification

```bash
# Vérifier le statut de tous les services
docker compose ps

# Vérifier les healthchecks
docker compose ps airflow-init

# Vérifier l'API Airflow
curl http://localhost:8081/health

# Accéder à l'UI
open http://localhost:8081
# Username: admin
# Password: admin
```

## Résultat Final

✅ **airflow-init** devient "healthy" après initialisation complète
✅ **Tous les services Airflow** démarrent seulement après que init soit healthy
✅ **Pas de race conditions** grâce aux healthchecks
✅ **Idempotent** - Peut être relancé sans problème
✅ **Scalable** - S'adapte à la vitesse de chaque machine

## Logs de Succès

```bash
$ docker compose ps airflow-webserver
NAME                STATUS         PORTS
airflow-webserver   Up 2 minutes   0.0.0.0:8081->8080/tcp

$ curl -s http://localhost:8081/health | jq .metadatabase
{
  "status": "healthy"
}
```

## Fichiers Modifiés

1. [`docker-compose.yaml`](../docker-compose.yaml) - Configuration des services
2. [`scripts/start-all.sh`](../scripts/start-all.sh) - Script de démarrage automatique
3. [`scripts/restart-airflow.sh`](../scripts/restart-airflow.sh) - Script de redémarrage Airflow
4. [`STARTUP.md`](../STARTUP.md) - Guide de démarrage général

## Notes Importantes

- **Ne pas utiliser `docker compose up -d` sans attendre** - Privilégier le script `start-all.sh`
- **Les deux variables SQL_ALCHEMY_CONN sont nécessaires** - L'entrypoint utilise l'ancienne
- **`_AIRFLOW_DB_MIGRATE=false` est essentiel** - Sinon l'entrypoint fait sa propre vérification qui échoue
- **Le healthcheck de `airflow-init` doit rester actif** - Les autres services en dépendent

## Références

- [Airflow Docker Documentation](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html)
- [Airflow Configuration Reference](https://airflow.apache.org/docs/apache-airflow/stable/configurations-ref.html)
- [Docker Compose Healthchecks](https://docs.docker.com/compose/compose-file/compose-file-v3/#healthcheck)
