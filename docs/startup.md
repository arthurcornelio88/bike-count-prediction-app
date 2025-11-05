# MLOps Stack - Guide de Démarrage

## Démarrage Rapide

### Option 1: Script automatique (Recommandé)

```bash
# Démarrage sans monitoring
./scripts/start-all.sh

# Démarrage avec monitoring (Prometheus + Grafana)
./scripts/start-all.sh --with-monitoring
```

### Option 2: Docker Compose direct

```bash
# Sans monitoring
docker compose up -d

# Avec monitoring
docker compose --profile monitoring up -d
```

## Architecture du Démarrage

Le système utilise des **healthchecks** et **depends_on conditions** pour
garantir un démarrage robuste:

```text
1. Infrastructure (PostgreSQL, Redis, Cloud SQL Proxy)
   ↓ (attendre que postgres-airflow et redis-airflow soient healthy)

2. MLflow + RegModel API (démarrent en parallèle)
   ↓

3. Airflow Init (initialise la DB Airflow)
   ↓ (attendre que airflow-init soit healthy)

4. Airflow Services (webserver, scheduler, worker, flower)
   ↓ (optionnel)

5. Monitoring Stack (Prometheus, Grafana, Airflow Exporter)
```

## Services et Ports

| Service | Port | Credentials |
|---------|------|-------------|
| MLflow | 5000 | - |
| RegModel API | 8000 | - |
| Airflow UI | 8081 | admin/admin |
| Flower (Celery) | 5555 | - |
| Prometheus | 9090 | - |
| Grafana | 3000 | (voir .env) |

## Résolution des Problèmes

### Erreur: "You need to initialize the database"

**Cause:** Le service `airflow-init` n'a pas terminé l'initialisation.

**Solution:** Utilisez le script `start-all.sh` qui attend que l'initialisation soit complète.

### Vérifier les logs d'un service

```bash
# Tous les logs
docker compose logs -f

# Logs d'un service spécifique
docker compose logs -f airflow-init
docker compose logs -f airflow-webserver
docker compose logs -f mlflow
```

### Vérifier le statut des healthchecks

```bash
docker compose ps
```

### Redémarrer proprement

```bash
# Arrêter tous les services
docker compose down

# Nettoyer les volumes (ATTENTION: supprime les données)
docker compose down -v

# Redémarrer
./scripts/start-all.sh
```

### Airflow Init reste en "starting"

```bash
# Vérifier les logs
docker compose logs airflow-init

# Vérifier que PostgreSQL est accessible
docker compose exec airflow-init nc -zv postgres-airflow 5432
```

## Configuration Requise

### Fichiers .env nécessaires

1. `.env` (racine) - Configuration globale
2. `backend/regmodel/.env` - Configuration RegModel API
3. `gcp.json` - Service Account GCP
4. `mlflow-ui-access.json` - Service Account pour MLflow
5. `mlflow-trainer.json` - Service Account pour le training

### Variables d'environnement importantes

```bash
# Airflow
_AIRFLOW_WWW_USER_USERNAME=admin
_AIRFLOW_WWW_USER_PASSWORD=admin

# MLflow DB
MLFLOW_DB_USER=...
MLFLOW_DB_PASSWORD=...
MLFLOW_DB_NAME=...
MLFLOW_INSTANCE_CONNECTION=...

# Grafana
GF_SECURITY_ADMIN_PASSWORD=...

# Discord (optionnel)
DISCORD_WEBHOOK_URL=...
```

## Optimisations

### Healthchecks configurés

- **postgres-airflow**: Vérifie que PostgreSQL accepte les connexions
- **redis-airflow**: Vérifie que Redis répond au PING
- **airflow-init**: Vérifie qu'un fichier de statut existe après init complète
- **flower**: Vérifie que l'API Flower répond

### Depends_on conditions

- `service_healthy`: Attend que le healthcheck du service soit OK
- `service_started`: Attend que le service soit démarré (pas de healthcheck requis)

## Commandes Utiles

```bash
# Vérifier les containers en cours
docker compose ps

# Arrêter un service spécifique
docker compose stop airflow-webserver

# Redémarrer un service
docker compose restart airflow-webserver

# Reconstruire et redémarrer un service
docker compose up -d --build regmodel-backend

# Voir l'utilisation des ressources
docker stats
```

## Architecture Robuste

Le nouveau design garantit:

1. **PostgreSQL et Redis** démarrent en premier
2. **Airflow Init** attend que PostgreSQL soit healthy, puis:
   - Exécute `airflow db init`
   - Exécute `airflow db upgrade`
   - Crée l'utilisateur admin
   - Reste actif avec un healthcheck
3. **Tous les services Airflow** attendent que `airflow-init` soit healthy
4. **Pas de race conditions** grâce aux healthchecks
5. **Scalable**: Chaque machine attend selon ses propres healthchecks, pas de sleep arbitraires

## Support

En cas de problème persistant:

1. Vérifiez les logs: `docker compose logs -f`
2. Vérifiez les healthchecks: `docker compose ps`
3. Nettoyez et redémarrez: `docker compose down && ./scripts/start-all.sh`
