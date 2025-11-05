# Audit Urgent: Double Evaluation Metrics

**Date**: 2025-11-04
**Status**: 🔴 CRITICAL - Métriques manquantes
**Impact**: Monitoring incomplet, comparaisons trompeuses

---

## 🎯 Objectif de l'Audit

Vérifier que les **4 métriques R² de la double évaluation** sont correctement
exposées et monitorées:

```text
Champion (modèle en prod):
├─ r2_champion_baseline   (test_baseline: 181K samples fixes)
└─ r2_champion_current    (test_current: 20% fresh data)

Challenger (modèle retrained):
├─ r2_challenger_baseline (test_baseline: detect regression)
└─ r2_challenger_current  (test_current: measure improvement)
```

---

## ✅ Ce qui FONCTIONNE

### 1. DAG `monitor_and_fine_tune` - XCom Push

**Fichier**: [dags/dag_monitor_and_train.py:574-593](dags/dag_monitor_and_train.py#L574-L593)

```python
# ✅ Challenger metrics
context["ti"].xcom_push(key="r2_baseline", value=float(r2_baseline))  # Challenger baseline
context["ti"].xcom_push(key="r2_current", value=float(r2_current))    # Challenger current

# ✅ Champion baseline
context["ti"].xcom_push(key="champion_r2_baseline", value=float(champion_r2_baseline))

# ✅ Champion current (via validate_model task)
context["ti"].xcom_push(key="r2", value=float(r2))  # Ligne 200
```

**Verdict**: ✅ Les 4 métriques sont calculées et pushées vers XCom

---

### 2. BigQuery Audit Logs - Schema

**Fichier**: [monitoring/custom_exporters/airflow_exporter.py:237-260](monitoring/custom_exporters/airflow_exporter.py#L237-L260)

```sql
SELECT
    r2,              -- Champion (validation)
    r2_baseline,     -- Challenger baseline
    r2_current,      -- Challenger current
    ...
FROM `monitoring_audit.logs`
```

**Verdict**: ⚠️ Partial - Manque `champion_r2_current` dans le schema BigQuery

---

## 🔴 PROBLÈMES CRITIQUES

### Problème 1: Seulement 2 métriques Prometheus exposées

**Fichier**: [monitoring/custom_exporters/airflow_exporter.py:82-101](monitoring/custom_exporters/airflow_exporter.py#L82-L101)

**Métriques actuelles**:

```python
BIKE_PREDICTION_R2 = Gauge("bike_prediction_r2")           # Challenger current ✅
BIKE_MODEL_R2_PRODUCTION = Gauge("bike_model_r2_production")  # Champion baseline ✅
```

**Métriques MANQUANTES**:

```python
# ❌ Pas de métrique pour Champion current
# ❌ Pas de métrique pour Challenger baseline
```

**Impact**:

- Impossible de comparer les 2 modèles sur le **même test set**
- Dashboards montrent "apples vs oranges" (baseline vs current)
- Décisions basées sur des comparaisons non équitables

---

### Problème 2: Logique d'exposition confuse

**Fichier**: [monitoring/custom_exporters/airflow_exporter.py:436-466](monitoring/custom_exporters/airflow_exporter.py#L436-L466)

```python
if deployment_decision == "deploy":
    r2 = bq_metrics.get("r2_current")  # Challenger current
    BIKE_MODEL_R2_PRODUCTION.set(r2_float)  # ⚠️ Écrase champion!
else:
    r2 = bq_metrics.get("r2")  # Champion
    BIKE_MODEL_R2_PRODUCTION.set(r2_float)
```

**Problème**: `bike_model_r2_production` change de signification selon le déploiement!

---

### Problème 3: Dashboards trompeurs

**Fichier**: `monitoring/grafana/provisioning/dashboards/model_performance.json`

**Query actuelle**:

```promql
bike_model_r2_production   # Champion baseline (0.867)
bike_prediction_r2         # Challenger current (0.528)
```

**Légende affichée**: "Champion vs Challenger"
**Réalité**: "Champion sur baseline vs Challenger sur current" → **Comparaison invalide!**

---

## 📊 Données Manquantes

### Ce que nous AVONS

| Métrique | Source | Valeur |
|----------|--------|--------|
| Champion baseline | XCom `champion_r2_baseline` | 0.867 ✅ |
| Champion current | XCom `r2` (validate_model) | ??? ⚠️ |
| Challenger baseline | XCom `r2_baseline` | ??? ❌ |
| Challenger current | XCom `r2_current` | 0.528 ✅ |

### Ce que nous EXPOSONS à Prometheus

| Métrique Prometheus | Correspond à | Exposée? |
|---------------------|--------------|----------|
| `bike_model_r2_production` | Champion baseline | ✅ |
| `bike_prediction_r2` | Challenger current | ✅ |
| `bike_model_r2_champion_current` | Champion current | ❌ |
| `bike_model_r2_challenger_baseline` | Challenger baseline | ❌ |

---

## 🎯 Actions Correctives Requises

### Action 1: Ajouter les 2 métriques manquantes dans Airflow Exporter

**Fichier**: `monitoring/custom_exporters/airflow_exporter.py`

**Ajouter après ligne 101**:

```python
# Double evaluation - Full metrics
BIKE_MODEL_R2_CHAMPION_BASELINE = Gauge(
    "bike_model_r2_champion_baseline",
    "Champion model R² on test_baseline (fixed reference)",
)
BIKE_MODEL_R2_CHAMPION_CURRENT = Gauge(
    "bike_model_r2_champion_current",
    "Champion model R² on test_current (new distribution)",
)
BIKE_MODEL_R2_CHALLENGER_BASELINE = Gauge(
    "bike_model_r2_challenger_baseline",
    "Challenger model R² on test_baseline (regression check)",
)
BIKE_MODEL_R2_CHALLENGER_CURRENT = Gauge(
    "bike_model_r2_challenger_current",
    "Challenger model R² on test_current (improvement check)",
)
```

---

### Action 2: Modifier la logique de collection BigQuery

**Fichier**: `monitoring/custom_exporters/airflow_exporter.py:416-486`

**Remplacer la logique actuelle par**:

```python
def _collect_monitoring_metrics(self, dag_id: str, dag_run_id: str) -> None:
    # Try BigQuery first
    bq_metrics = self._get_latest_monitoring_metrics_from_bq()

    if bq_metrics:
        # Champion metrics (from validate_model task)
        champion_r2 = bq_metrics.get("r2")  # Champion current (validation)
        champion_r2_baseline = None  # TODO: Add to BigQuery schema

        # Challenger metrics (from training)
        challenger_r2_baseline = bq_metrics.get("r2_baseline")
        challenger_r2_current = bq_metrics.get("r2_current")

        # Set all 4 metrics
        if champion_r2 is not None:
            BIKE_MODEL_R2_CHAMPION_CURRENT.set(float(champion_r2))

        if champion_r2_baseline is not None:
            BIKE_MODEL_R2_CHAMPION_BASELINE.set(float(champion_r2_baseline))

        if challenger_r2_baseline is not None:
            BIKE_MODEL_R2_CHALLENGER_BASELINE.set(float(challenger_r2_baseline))

        if challenger_r2_current is not None:
            BIKE_MODEL_R2_CHALLENGER_CURRENT.set(float(challenger_r2_current))

        # Legacy metrics (keep for backward compatibility)
        BIKE_MODEL_R2_PRODUCTION.set(float(champion_r2))  # Always champion
        BIKE_PREDICTION_R2.set(float(challenger_r2_current))
```

---

### Action 3: Ajouter `champion_r2_current` au schema BigQuery

**Fichier**: `dags/dag_monitor_and_train.py:705-731`

**Ajouter le champ manquant**:

```python
audit_record = {
    # ...existing fields...
    "r2": float(r2) if r2 else 0.0,  # Champion current (validation)
    "r2_baseline": float(r2_baseline) if r2_baseline is not None else None,  # Challenger baseline
    "r2_current": float(r2_current) if r2_current is not None else None,  # Challenger current
    # NEW: Add champion evaluated on current
    "champion_r2_current": float(champion_r2_current) if champion_r2_current else None,
    "champion_r2_baseline": float(champion_r2_baseline) if champion_r2_baseline else None,
}
```

**Note**: Requiert `ALTER TABLE` sur BigQuery ou recréation de la table.

---

### Action 4: Mettre à jour les dashboards Grafana

**Fichier**: `monitoring/grafana/provisioning/dashboards/model_performance.json`

**Nouveau panel: "Fair Comparison - Both Models on test_baseline"**:

```json
{
  "title": "R² Comparison - test_baseline (Fair)",
  "targets": [
    {
      "expr": "bike_model_r2_champion_baseline",
      "legendFormat": "Champion (baseline)"
    },
    {
      "expr": "bike_model_r2_challenger_baseline",
      "legendFormat": "Challenger (baseline)"
    }
  ]
}
```

**Nouveau panel: "Fair Comparison - Both Models on test_current"**:

```json
{
  "title": "R² Comparison - test_current (Fair)",
  "targets": [
    {
      "expr": "bike_model_r2_champion_current",
      "legendFormat": "Champion (current)"
    },
    {
      "expr": "bike_model_r2_challenger_current",
      "legendFormat": "Challenger (current)"
    }
  ]
}
```

---

### Action 5: Mettre à jour les alertes Grafana

**Fichier**: `monitoring/grafana/provisioning/alerting/rules.yml`

**Problème**: Les alertes utilisent `bike_model_r2_production` qui est ambigu.

**Solution**: Créer des alertes spécifiques:

```yaml
- name: model_performance_champion_baseline
  condition: bike_model_r2_champion_baseline < 0.65

- name: model_performance_champion_current
  condition: bike_model_r2_champion_current < 0.65

- name: challenger_regression_detected
  condition: bike_model_r2_challenger_baseline < 0.60
```

---

## 📈 Bénéfices Attendus

### Avant (état actuel)

```text
Grafana Dashboard:
├─ Champion baseline: 0.867 ✅
└─ Challenger current: 0.528 ❌

❌ Comparaison invalide (test sets différents)
❌ Impossible de savoir si challenger améliore réellement
❌ Décisions basées sur des métriques trompeuses
```

### Après (état corrigé)

```text
Grafana Dashboard:
├─ Panel 1: Both models on test_baseline (fair comparison)
│   ├─ Champion baseline: 0.867
│   └─ Challenger baseline: 0.60 → REJECT (régression!)
│
└─ Panel 2: Both models on test_current (fair comparison)
    ├─ Champion current: 0.75
    └─ Challenger current: 0.78 → DEPLOY (amélioration!)

✅ Comparaisons équitables
✅ Décisions basées sur métriques correctes
✅ Monitoring complet de la double évaluation
```

---

## 🔍 Vérifications Post-Correction

### 1. Vérifier les métriques Prometheus

```bash
# Check all 4 metrics are exposed
curl http://localhost:9101/metrics | grep "bike_model_r2"

# Expected output (4 metrics):
# bike_model_r2_champion_baseline 0.867
# bike_model_r2_champion_current 0.75
# bike_model_r2_challenger_baseline 0.60
# bike_model_r2_challenger_current 0.78
```

### 2. Vérifier BigQuery audit logs

```sql
SELECT
    timestamp,
    r2,                      -- Champion current
    champion_r2_baseline,    -- Champion baseline (NEW)
    champion_r2_current,     -- Champion current (duplicate of r2)
    r2_baseline,             -- Challenger baseline
    r2_current,              -- Challenger current
    deployment_decision
FROM `monitoring_audit.logs`
ORDER BY timestamp DESC
LIMIT 5;
```

### 3. Vérifier Grafana dashboards

- Dashboard "Model Performance" doit montrer 2 nouveaux panels
- Légendes doivent indiquer explicitement le test set utilisé
- Comparaisons doivent être sur le **même test set**

---

## ⏱️ Priorité d'Implémentation

| Action | Priorité | Effort | Impact |
|--------|----------|--------|--------|
| Action 1: Ajouter métriques Prometheus | 🔴 HIGH | 30 min | Critique |
| Action 2: Modifier collection logic | 🔴 HIGH | 1h | Critique |
| Action 3: Schema BigQuery | 🟡 MEDIUM | 30 min | Important |
| Action 4: Dashboards Grafana | 🟡 MEDIUM | 1h | Important |
| Action 5: Alertes Grafana | 🟢 LOW | 30 min | Nice-to-have |

**Total effort estimé**: 3-4 heures

---

## 📚 Références

- [dags/dag_monitor_and_train.py](dags/dag_monitor_and_train.py) - Ligne 365-598
- [monitoring/custom_exporters/airflow_exporter.py](monitoring/custom_exporters/airflow_exporter.py) - Ligne 82-536
- [docs/sliding_window.md](docs/sliding_window.md) - Double evaluation strategy
- [docs/training_strategy.md](docs/training_strategy.md) - Decision logic

---

**Conclusion**: Le système calcule correctement les 4 métriques mais
**n'en expose que 2**. Les dashboards sont **trompeurs** car ils comparent
des modèles sur des test sets différents. Les actions correctives sont
**critiques** pour un monitoring MLOps fiable.
