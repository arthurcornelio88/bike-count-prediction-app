# Troubleshooting WSL2 + Docker Volume Mounts

## Problème : Airflow UI montre "0 DAGs"

### Symptômes

- L'UI Airflow affiche "No results" / "0 DAGs"
- Les DAGs existent localement dans `/dags/`
- L'API Airflow retourne des DAG runs mais `/api/v1/dags` est vide
- Les logs montrent que des DAGs ont déjà tourné dans le passé

### Cause

**Bug de bind mount sur WSL2**: Le volume mount `/dags` devient vide dans le container après un
rebuild ou un redémarrage système.

### Diagnostic

```bash
# 1. Vérifier que les DAGs existent localement
ls -la ./dags/*.py

# 2. Vérifier dans le container (devrait être vide si le bug est présent)
docker exec airflow-scheduler ls -la /opt/airflow/dags/

# 3. Vérifier les mounts (devrait montrer le bon path mais 0 bytes)
docker inspect airflow-scheduler | grep -A 5 "dags"
```

### Solution : Restart les containers Airflow

```bash
# Redémarrer les containers Airflow pour forcer le remount
docker compose restart airflow-webserver airflow-scheduler airflow-worker

# Attendre 30 secondes puis vérifier
sleep 30
docker exec airflow-scheduler ls -la /opt/airflow/dags/

# Vérifier via l'API
curl -s --user admin:admin http://localhost:8081/api/v1/dags | jq '.total_entries'
```

### Vérification finale

1. Rafraîchir l'UI : <http://localhost:8081/home>
2. Les 3 DAGs devraient apparaître :
   - `daily_fetch_bike_data`
   - `daily_prediction`
   - `monitor_and_fine_tune`

---

## Problème : Permissions denied sur logs/dags

### Symptômes

```text
PermissionError: [Errno 13] Permission denied: '/opt/airflow/logs/...'
```

### Solution

```bash
# Fixer les permissions (user Airflow = 50000:50000)
sudo chown -R 50000:50000 ./dags ./logs ./plugins

# Ou utiliser votre user + chmod
sudo chown -R $USER:$USER ./dags ./logs ./plugins
chmod -R 755 ./dags ./logs ./plugins
```

---

## Problème : Docker volume mount vide sur WSL2

### Causes communes

1. **Path Windows vs WSL** : Utiliser `/home/user/...` et non `/mnt/c/Users/...`
2. **Docker Desktop settings** : Vérifier que WSL2 integration est activée
3. **Cache Docker** : Les volumes peuvent devenir stale après rebuild

### Solutions générales

#### 1. Vérifier WSL2 integration

Docker Desktop → Settings → Resources → WSL Integration → Cocher votre distro

#### 2. Rebuild complet sans cache

```bash
docker compose down -v
docker compose build --no-cache
docker compose up -d
```

#### 3. Forcer le recreate des containers

```bash
docker compose up -d --force-recreate
```

#### 4. Vérifier le path absolu

```bash
# Le path dans docker-compose.yml doit être relatif
volumes:
  - ./dags:/opt/airflow/dags   # ✅ Correct

# PAS de path absolu Windows-style
# - /mnt/c/Users/...  # ❌ Incorrect sur WSL2
```

---

## Problème : "No module named 'google.cloud'"

### Symptôme

Les DAGs échouent avec des imports manquants

### Solution

Vérifier `_PIP_ADDITIONAL_REQUIREMENTS` dans docker-compose.yml :

```yaml
environment:
  _PIP_ADDITIONAL_REQUIREMENTS: pandas scikit-learn requests google-cloud-bigquery google-cloud-storage gcsfs db-dtypes
```

Puis restart :

```bash
docker compose restart airflow-scheduler airflow-worker
```

---

## Mémo commandes utiles

### Vérifier l'état des services

```bash
docker compose ps
docker compose --profile monitoring ps
```

### Logs en temps réel

```bash
docker logs -f airflow-scheduler
docker logs -f airflow-webserver
docker logs -f airflow-exporter
```

### Restart sélectif

```bash
# Juste Airflow
docker compose restart airflow-webserver airflow-scheduler airflow-worker

# Juste monitoring
docker compose --profile monitoring restart prometheus grafana airflow-exporter

# Tout redémarrer
docker compose restart
```

### Nettoyer complètement

```bash
# ⚠️  ATTENTION : Supprime TOUTES les données (logs, DB, volumes)
docker compose down -v
docker system prune -a --volumes
```

### Vérifier les volumes Docker

```bash
docker volume ls
docker volume inspect ds_traffic_cycliste1_postgres_airflow_data
```

---

**Dernière mise à jour** : 2025-11-03
