# 🎯 Training Strategy — Hybrid Architecture

**Last Updated**: 2025-10-12

---

## Architecture Overview

**Hybrid approach**: Local champion training + Production fine-tuning

| Component | Where | When | Data Size | Duration |
|-----------|-------|------|-----------|----------|
| **Champion Training** | 💻 Local | One-time (+ quarterly) | 724k records | 15-30 min |
| **Fine-Tuning** | ☁️ Production | Weekly (if drift) | 2k records | 5-10 min |
| **Evaluation** | ☁️ Production | Weekly | 181k test set | 2-3 min |
| **Inference** | ☁️ Production | Daily | 100 records | <1 sec |

---

## Data Sources

### Baseline Data (GCS)

```bash
# Environment variables (set in .env.airflow or runtime)
TRAIN_DATA_PATH=gs://df_traffic_cyclist1/data/train_baseline.csv  # 724k rows
TEST_DATA_PATH=gs://df_traffic_cyclist1/data/test_baseline.csv    # 181k rows
```

**Upload once:**

```bash
gsutil -m cp data/train_baseline.csv gs://df_traffic_cyclist1/data/
gsutil -m cp data/test_baseline.csv gs://df_traffic_cyclist1/data/
```

### Live Data (BigQuery)

```sql
-- Daily fetch from Paris Open Data API
SELECT * FROM `bike_traffic_raw.daily_YYYYMMDD`
```

---

## Workflow

### 1. Initial Champion Training (Local)

```bash
# Train champion model on baseline data
python backend/regmodel/app/train.py \
    --model-type xgboost \
    --data-source baseline

# Script automatically uses environment variables:
# - TRAIN_DATA_PATH (from .env or export)
# - TEST_DATA_PATH (from .env or export)
```

**Output:**

- Model saved to MLflow (`models:/bike-traffic-champion/latest`)
- Metrics: MAE ~12, RMSE ~20, R² ~0.88
- Tags: `dataset=baseline`, `training_type=champion`, `data_source=gcs`

**Why local?**

- Full control over hyperparameters
- Fast iteration for debugging
- No cloud costs for heavy training
- Can use GPU if available

---

### 2. Production Fine-Tuning (Weekly DAG)

```python
# dags/dag_monitor_weekly.py

def fine_tune_on_fresh_data(**context):
    """Lightweight fine-tuning on recent data (if drift detected)"""

    # Load last 30 days from BigQuery (~2,160 records max)
    query = """
        SELECT * FROM `bike_traffic_raw.daily_*`
        WHERE _TABLE_SUFFIX >= FORMAT_DATE('%Y%m%d',
            DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY))
    """
    fresh_data = bq_client.query(query).to_dataframe()

    # Fine-tune via API (NOT full retraining)
    response = requests.post(f"{API_URL}/fine-tune", json={
        "base_model": "champion_v1",
        "data": fresh_data.to_dict(orient="records"),
        "learning_rate": 0.001,  # 10x smaller than initial
        "epochs": 5,              # Few epochs only
        "freeze_layers": True,    # Keep most weights frozen
    }, timeout=300)

    challenger_metrics = response.json()["metrics"]

    # CRITICAL: Evaluate on SAME test_baseline.csv from GCS
    test_baseline = pd.read_csv(os.getenv("TEST_DATA_PATH"))

    challenger_test_mae = evaluate_model(challenger_metrics, test_baseline)
    champion_test_mae = get_champion_metrics()["mae"]

    # Champion/Challenger decision (5% improvement threshold)
    if challenger_test_mae < champion_test_mae * 0.95:
        promote_to_champion(challenger_metrics)
        log_to_bq("champion_promoted", challenger_test_mae)
    else:
        log_to_bq("champion_kept", champion_test_mae)
```

**Why production fine-tuning?**

- Small dataset (30 days) → fast and cheap
- Adapts to recent patterns (seasonality, events)
- No need to retrain from scratch
- Cloud Run can handle it (~5-10 min)

---

### 3. Quarterly Retrain (Local)

```bash
# Download production data (quarterly)
bq extract --destination_format=CSV \
  'bike_traffic_raw.daily_*' \
  gs://df_traffic_cyclist1/exports/bq_export_*.csv

# Merge with baseline
python scripts/merge_baseline_bq.py \
    --baseline $TRAIN_DATA_PATH \
    --bq-export data/bq_export_*.csv \
    --output data/new_train_baseline.csv

# Retrain champion from scratch
python backend/regmodel/app/train.py \
    --model-type xgboost \
    --data-source custom \
    --train-path data/new_train_baseline.csv \
    --test-path $TEST_DATA_PATH

# If improved → upload new baseline to GCS
gsutil -m cp data/new_train_baseline.csv \
  gs://df_traffic_cyclist1/data/train_baseline.csv
```

---

## Key Decisions

### When to Retrain Locally?

Triggers for full local retraining:

- **Quarterly** (every 3 months) → new baseline with fresh data
- **Major drift** (>30% MAE degradation) → investigate and retrain
- **Architecture change** (new model, features) → local development
- **Bug fix** in feature engineering → local testing

### When to Fine-Tune in Production?

Triggers for production fine-tuning:

- **Weekly drift detected** (Evidently alert)
- **Performance degradation** (MAE increase >10%)
- **Seasonal adaptation** (events, holidays)

### Champion/Challenger Evaluation

```python
# GOOD: Same test set for all models
champion_v1 = train(train_baseline)
eval_champion = evaluate(champion_v1, test_baseline)  # MAE: 12

challenger_v2 = fine_tune(champion_v1, last_30_days)
eval_challenger = evaluate(challenger_v2, test_baseline)  # MAE: 11.5

# Valid comparison! Same test set
if eval_challenger["mae"] < eval_champion["mae"] * 0.95:
    promote(challenger_v2)
```

**Critical**: Always evaluate on the **same fixed test_baseline.csv** to ensure valid model comparison.

---

## Complete Workflow Diagram

```text
┌─────────────────────────────────────────────────────────┐
│ INITIAL SETUP (Local - One Time)                       │
├─────────────────────────────────────────────────────────┤
│ 1. Upload baseline to GCS                              │
│    gsutil cp train_baseline.csv gs://bucket/data/      │
│                                                         │
│ 2. Set environment variables                           │
│    export TRAIN_DATA_PATH=gs://bucket/data/train...    │
│    export TEST_DATA_PATH=gs://bucket/data/test...      │
│                                                         │
│ 3. Train champion model (local)                        │
│    python train.py --model-type xgboost \              │
│                    --data-source baseline              │
│                                                         │
│ 4. Metrics: MAE ~12, RMSE ~20, R² ~0.88                │
│                                                         │
│ 5. Model saved to MLflow registry                      │
│    models:/bike-traffic-champion/latest                │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ PRODUCTION (Weekly DAG)                                 │
├─────────────────────────────────────────────────────────┤
│ 1. Fetch last 7 days from BigQuery                     │
│ 2. Drift detection (Evidently vs test_baseline)        │
│ 3. If NO drift → skip, keep champion                   │
│ 4. If drift → fine-tune on last 30 days                │
│ 5. Evaluate challenger on test_baseline from GCS       │
│ 6. Champion/Challenger decision (5% threshold)         │
│ 7. Log metrics to monitoring_audit.logs                │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ QUARTERLY RETRAIN (Local - Every 3 months)             │
├─────────────────────────────────────────────────────────┤
│ 1. Download all BigQuery data (3 months)               │
│ 2. Merge with train_baseline → new_train.csv           │
│ 3. Update TRAIN_DATA_PATH to new baseline              │
│ 4. Retrain champion locally (full training)            │
│    python train.py --data-source custom \              │
│                    --train-path new_train.csv          │
│ 5. Evaluate on SAME test_baseline                      │
│ 6. If improved → upload to GCS as new baseline         │
└─────────────────────────────────────────────────────────┘
```

---

## Environment Variables Reference

### Required for Training

```bash
# Set in .env.airflow or shell
export TRAIN_DATA_PATH=gs://df_traffic_cyclist1/data/train_baseline.csv
export TEST_DATA_PATH=gs://df_traffic_cyclist1/data/test_baseline.csv
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/gcp.json
```

### Usage in Code

```python
# backend/regmodel/app/train.py
import os

train_path = os.getenv("TRAIN_DATA_PATH", "data/train_baseline.csv")
test_path = os.getenv("TEST_DATA_PATH", "data/test_baseline.csv")

# Load from GCS or local
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
```

---

## MLflow Tags Strategy

```python
# Champion training
mlflow.set_tag("dataset", "baseline")
mlflow.set_tag("training_type", "champion")
mlflow.set_tag("data_source", "gcs")
mlflow.set_tag("train_size", 724000)
mlflow.set_tag("test_size", 181000)

# Fine-tuning
mlflow.set_tag("dataset", "last_30_days")
mlflow.set_tag("training_type", "fine_tune")
mlflow.set_tag("data_source", "bigquery")
mlflow.set_tag("base_model", "champion_v1")
```

---

## Summary

**Hybrid approach rationale**:

- **Local champion**: Full control, GCS baseline (724k rows), ~15-30 min
- **Production fine-tune**: Lightweight, BigQuery recent data (~2k rows), ~5-10 min
- **Fixed test set**: Same `test_baseline.csv` for all evaluations
- **Environment variables**: Flexible paths for GCS or local data
- **Conservative promotion**: 5% improvement threshold for challenger

This strategy balances **production agility** (weekly fine-tuning) with **model quality**
(quarterly full retraining), ensuring the system adapts to recent patterns while maintaining
rigorous evaluation standards.

---

**Related Docs:**

- [secrets.md](./secrets.md) — Secret Manager configuration
- [bigquery_setup.md](./bigquery_setup.md) — BigQuery datasets and baseline upload
- [dvc.md](./dvc.md) — Data versioning and temporal split
