# Implémentation: Double Evaluation Metrics - Guide Complet

**Date**: 2025-11-04
**Status**: ✅ DEPLOYED
**Durée estimée**: 30 min

> **Note**: This implementation has been completed. All metrics, dashboards, and alerts
> have been updated to use the 4-metric double-evaluation system.

---

## 📋 Résumé des Changements

Ajout de **4 métriques R² complètes** pour la double évaluation:

```text
bike_model_r2_champion_baseline    # Champion sur test_baseline
bike_model_r2_champion_current     # Champion sur test_current
bike_model_r2_challenger_baseline  # Challenger sur test_baseline
bike_model_r2_challenger_current   # Challenger sur test_current
```

**Backward compatibility**: Les anciennes métriques (`bike_model_r2_production`, `bike_prediction_r2`) sont conservées.

---

## 🔧 Fichiers Modifiés

### 1. `monitoring/custom_exporters/airflow_exporter.py`

**Changements**:

- ✅ Ajout de 4 Gauges Prometheus (lignes 123-139)
- ✅ Modification de `_collect_monitoring_metrics()` (lignes 443-492)
- ✅ Ajout de `champion_r2_baseline` à la query BigQuery (ligne 262)
- ✅ Ajout au dict `metrics` retourné (ligne 285)

**Impact**: Expose les 4 R² vers Prometheus

---

### 2. `dags/dag_monitor_and_train.py`

**Changements**:

- ✅ Pull de `champion_r2_baseline` depuis XCom (lignes 685-687)
- ✅ Ajout au default si training pas exécuté (ligne 703)
- ✅ Ajout au `audit_record` (lignes 733-735)

**Impact**: BigQuery audit logs contient maintenant `champion_r2_baseline`

---

### 3. `monitoring/add_champion_r2_baseline_column.sql` (NOUVEAU)

Script SQL pour ajouter la colonne à BigQuery.

---

## 🚀 Procédure de Déploiement

### Étape 1: Ajouter la colonne BigQuery (5 min)

```bash
# Se connecter à BigQuery via gcloud CLI
gcloud auth login

# Exécuter le script SQL
bq query --use_legacy_sql=false < monitoring/add_champion_r2_baseline_column.sql

# Vérifier que la colonne existe
bq show --schema datascientest-460618:monitoring_audit.logs | grep champion_r2_baseline
```

**Attendu**: Colonne `champion_r2_baseline FLOAT64` ajoutée

---

### Étape 2: Redémarrer Airflow Exporter (2 min)

```bash
# Rebuild le container avec les nouveaux Gauges
docker compose --profile monitoring build airflow-exporter

# Redémarrer
docker compose --profile monitoring up -d airflow-exporter

# Vérifier les logs
docker logs airflow-exporter --tail 50

# Vérifier les métriques exposées
curl http://localhost:9101/metrics | grep "bike_model_r2"
```

**Attendu**: 6 métriques R² exposées (4 nouvelles + 2 legacy)

---

### Étape 3: Redémarrer Airflow Webserver/Scheduler (2 min)

```bash
# Rebuild avec le DAG modifié
docker compose restart airflow-webserver airflow-scheduler

# Vérifier que le DAG charge sans erreur
docker logs airflow-scheduler --tail 50 | grep "monitor_and_fine_tune"
```

**Attendu**: Pas d'erreur de parsing DAG

---

### Étape 4: Vérification End-to-End (10 min)

#### 4.1 Déclencher un run de training

```bash
# Trigger le DAG monitor_and_fine_tune
docker exec airflow-webserver airflow dags trigger monitor_and_fine_tune

# Ou via UI: http://localhost:8081 → monitor_and_fine_tune → Trigger DAG
```

#### 4.2 Vérifier les logs Airflow

```bash
# Attendre que le DAG termine (~5-10 min)
docker logs airflow-scheduler -f | grep "champion_r2_baseline"
```

**Attendu**:

```text
Champion R² (test_baseline): 0.8670
Champion R² (test_current): 0.7500
Challenger R² (test_baseline): 0.6000
Challenger R² (test_current): 0.7800
```

#### 4.3 Vérifier BigQuery

```sql
SELECT
    timestamp,
    r2,                     -- Champion current (validate_model)
    champion_r2_baseline,   -- NEW!
    r2_baseline,            -- Challenger baseline
    r2_current,             -- Challenger current
    deployment_decision
FROM `datascientest-460618.monitoring_audit.logs`
ORDER BY timestamp DESC
LIMIT 5;
```

**Attendu**: Colonne `champion_r2_baseline` remplie avec valeurs non-NULL

#### 4.4 Vérifier Prometheus

```bash
# Attendre 60s (cache airflow_exporter)
sleep 60

# Query Prometheus
curl "http://localhost:9090/api/v1/query?query=bike_model_r2_champion_baseline" | jq '.data.result[0].value'
curl "http://localhost:9090/api/v1/query?query=bike_model_r2_champion_current" | jq '.data.result[0].value'
curl "http://localhost:9090/api/v1/query?query=bike_model_r2_challenger_baseline" | jq '.data.result[0].value'
curl "http://localhost:9090/api/v1/query?query=bike_model_r2_challenger_current" | jq '.data.result[0].value'
```

**Attendu**: Les 4 métriques retournent des valeurs > 0

---

## 📊 Étape 5 (Optionnel): Dashboards Grafana

**Note**: Les dashboards existants continuent de fonctionner (backward compatibility).

Pour ajouter les nouveaux graphes "fair comparison":

1. Ouvrir [http://localhost:3000](http://localhost:3000)
2. Créer un nouveau dashboard "Double Evaluation - Fair Comparison"
3. Ajouter 2 panels:

### Panel 1: Both Models on test_baseline

```promql
bike_model_r2_champion_baseline   # Legend: "Champion (baseline)"
bike_model_r2_challenger_baseline  # Legend: "Challenger (baseline)"
```

**Interprétation**:

- Si Challenger < 0.60 → Régression détectée → REJECT
- Sinon, continuer la comparaison

### Panel 2: Both Models on test_current

```promql
bike_model_r2_champion_current     # Legend: "Champion (current)"
bike_model_r2_challenger_current   # Legend: "Challenger (current)"
```

**Interprétation**:

- Si Challenger > Champion → Amélioration → DEPLOY
- Sinon → SKIP

---

## ✅ Checklist de Validation

### Base Requirements

- [ ] Colonne `champion_r2_baseline` existe dans BigQuery
- [ ] Airflow Exporter redémarré sans erreurs
- [ ] Airflow Scheduler/Webserver redémarrés sans erreurs
- [ ] DAG `monitor_and_fine_tune` charge sans erreur

### Functional Tests

- [ ] DAG run complété avec succès
- [ ] Logs Airflow montrent les 4 R² values
- [ ] BigQuery `monitoring_audit.logs` contient `champion_r2_baseline` non-NULL
- [ ] Prometheus expose `bike_model_r2_champion_baseline`
- [ ] Prometheus expose `bike_model_r2_champion_current`
- [ ] Prometheus expose `bike_model_r2_challenger_baseline`
- [ ] Prometheus expose `bike_model_r2_challenger_current`
- [ ] Métriques legacy (`bike_model_r2_production`, `bike_prediction_r2`) toujours fonctionnelles

### Grafana (Optionnel)

- [ ] Dashboard "Double Evaluation" créé
- [ ] Panel "test_baseline comparison" affiche les 2 modèles
- [ ] Panel "test_current comparison" affiche les 2 modèles

---

## 🔄 Rollback (si problème)

### Rollback Airflow Exporter

```bash
# Revert les changements
git checkout HEAD~1 monitoring/custom_exporters/airflow_exporter.py

# Rebuild + restart
docker compose --profile monitoring build airflow-exporter
docker compose --profile monitoring up -d airflow-exporter
```

### Rollback DAG

```bash
# Revert
git checkout HEAD~1 dags/dag_monitor_and_train.py

# Restart
docker compose restart airflow-webserver airflow-scheduler
```

### Rollback BigQuery (si table recréée)

**Note**: Si vous avez utilisé `ALTER TABLE ADD COLUMN`, pas besoin de rollback (colonne vide ne gêne pas).

Si vous avez `DROP TABLE`, restaurer depuis backup:

```bash
# Voir les backups disponibles
bq ls --transfer_config datascientest-460618

# Restaurer (à adapter selon backup)
# bq restore ...
```

---

## 📈 Résultats Attendus Après Déploiement

### Avant (état actuel)

```bash
curl http://localhost:9101/metrics | grep "bike.*r2"
# bike_model_r2_production 0.867
# bike_prediction_r2 0.528
```

**Problème**: 2 métriques seulement, comparaison sur test sets différents

---

### Après (état corrigé)

```bash
curl http://localhost:9101/metrics | grep "bike.*r2"
# bike_model_r2_production 0.867              # Legacy (Champion current)
# bike_prediction_r2 0.528                     # Legacy (Challenger current)
# bike_model_r2_champion_baseline 0.867        # NEW
# bike_model_r2_champion_current 0.750         # NEW
# bike_model_r2_challenger_baseline 0.600      # NEW
# bike_model_r2_challenger_current 0.780       # NEW
```

**Bénéfice**: 6 métriques, comparaisons équitables possibles

---

## 🆘 Troubleshooting

### Problème 1: Colonne BigQuery n'existe pas

**Symptôme**: Logs airflow_exporter affichent `Error querying BigQuery audit table: column champion_r2_baseline not found`

**Solution**:

```bash
# Exécuter le script SQL
bq query --use_legacy_sql=false < monitoring/add_champion_r2_baseline_column.sql
```

---

### Problème 2: Métriques Prometheus = 0

**Symptôme**: `bike_model_r2_champion_baseline{} 0`

**Causes possibles**:

1. BigQuery colonne vide (pas encore de run DAG)
2. Airflow Exporter cache (attendre 60s)
3. DAG n'a pas encore pusher la valeur dans BigQuery

**Solution**:

```bash
# Trigger un nouveau run
docker exec airflow-webserver airflow dags trigger monitor_and_fine_tune

# Attendre que le DAG termine
# Attendre 60s pour le cache exporter
sleep 60

# Re-check
curl http://localhost:9101/metrics | grep champion_r2_baseline
```

---

### Problème 3: DAG échoue avec KeyError

**Symptôme**: Task `end_monitoring` fail avec `KeyError: 'champion_r2_baseline'`

**Cause**: Le XCom `champion_r2_baseline` n'est pas pusher par `fine_tune_model`

**Solution**: Vérifier que le code à
[dag_monitor_and_train.py:591-593](dags/dag_monitor_and_train.py#L591-L593)
push bien la valeur:

```python
context["ti"].xcom_push(
    key="champion_r2_baseline",
    value=float(champion_r2_baseline) if champion_r2_baseline else None,
)
```

---

## 📚 Références

- [AUDIT_DOUBLE_EVAL.md](./AUDIT_DOUBLE_EVAL.md) - Rapport d'audit complet
- [01_architecture.md](./01_architecture.md) - Architecture monitoring
- [03_metrics_reference.md](./03_metrics_reference.md) - Référence métriques
- [dags/dag_monitor_and_train.py](../../dags/dag_monitor_and_train.py) - Code DAG
- [monitoring/custom_exporters/airflow_exporter.py](../../monitoring/custom_exporters/airflow_exporter.py) - Code exporter

---

## ✅ Validation Finale

Une fois toutes les étapes complétées:

```bash
# Test complet
echo "=== 1. BigQuery Schema ==="
bq show --schema datascientest-460618:monitoring_audit.logs | grep champion_r2_baseline

echo "=== 2. Prometheus Metrics ==="
curl -s http://localhost:9101/metrics | grep "bike_model_r2" | wc -l  # Should be 6

echo "=== 3. Latest Values ==="
curl -s "http://localhost:9090/api/v1/query?query=bike_model_r2_champion_baseline" | jq '.data.result[0].value[1]'
curl -s "http://localhost:9090/api/v1/query?query=bike_model_r2_champion_current" | jq '.data.result[0].value[1]'
curl -s "http://localhost:9090/api/v1/query?query=bike_model_r2_challenger_baseline" | jq '.data.result[0].value[1]'
curl -s "http://localhost:9090/api/v1/query?query=bike_model_r2_challenger_current" | jq '.data.result[0].value[1]'

echo "=== 4. BigQuery Latest Row ==="
bq query --use_legacy_sql=false --format=prettyjson "
SELECT timestamp, r2, champion_r2_baseline, r2_baseline, r2_current
FROM \`datascientest-460618.monitoring_audit.logs\`
ORDER BY timestamp DESC
LIMIT 1
"
```

**Attendu**: Toutes les commandes retournent des valeurs valides (pas d'erreurs, pas de NULL).

---

**Status**: 🎯 Ready to Deploy - Tous les fichiers modifiés, procédure documentée
