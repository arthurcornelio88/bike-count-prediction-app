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

## Workflow

### 1. Initial Setup (Local - One Time)

```bash
# Train champion model on full baseline
python scripts/train_champion.py \
    --train data/train_baseline.csv \
    --test data/test_baseline.csv \
    --model-type xgboost \
    --output models/champion_v1

# Expected: MAE ~12, RMSE ~20, R² ~0.88

# Register in MLflow
mlflow models register --name bike-traffic-champion --model-uri models/champion_v1

# Upload to GCS
gsutil -m cp -r models/champion_v1 gs://<your-bucket>/models/champion_v1
```

**Why local?**

- Full control over hyperparameters and features
- Fast iteration for debugging
- No cloud costs for initial training
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
        WHERE _TABLE_SUFFIX >= FORMAT_DATE('%Y%m%d', DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY))
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

    # CRITICAL: Evaluate on SAME test_baseline.csv
    test_baseline = load_from_gcs("gs://<your-bucket>/raw_data/test_baseline.csv")

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

### 3. Quarterly Retrain (Local - Every 3 months)

```bash
# Download production data
bq extract --destination_format=CSV \
  'bike_traffic_raw.daily_*' \
  gs://<your-bucket>/raw_data/bq_export_*.csv

# Merge with baseline
python scripts/merge_baseline_bq.py \
    --baseline data/train_baseline.csv \
    --bq-export data/bq_export_*.csv \
    --output data/new_train_baseline.csv

# Retrain champion from scratch
python scripts/train_champion.py \
    --train data/new_train_baseline.csv \
    --test data/test_baseline.csv \
    --model-type xgboost \
    --output models/champion_v2

# Evaluate on SAME test set
# If improved → deploy as new champion
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
│ 1. Train champion_v1 on train_baseline.csv (local)     │
│ 2. Evaluate on test_baseline.csv → MAE: ~12            │
│ 3. Upload to GCS + MLflow registry                     │
│ 4. Deploy to Cloud Run API                             │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ PRODUCTION (Weekly DAG)                                 │
├─────────────────────────────────────────────────────────┤
│ 1. Fetch last 7 days from BigQuery                     │
│ 2. Drift detection (Evidently vs test_baseline)        │
│ 3. If NO drift → skip, keep champion                   │
│ 4. If drift → fine-tune on last 30 days                │
│ 5. Evaluate challenger on SAME test_baseline.csv       │
│ 6. Champion/Challenger decision (5% threshold)         │
│ 7. Log metrics to monitoring_audit.logs                │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ QUARTERLY RETRAIN (Local - Every 3 months)             │
├─────────────────────────────────────────────────────────┤
│ 1. Download all BigQuery data (3 months)               │
│ 2. Merge with train_baseline.csv → new_train.csv       │
│ 3. Retrain champion_v2 locally (full training)         │
│ 4. Evaluate on SAME test_baseline.csv                  │
│ 5. If improved → deploy as new champion                │
└─────────────────────────────────────────────────────────┘
```

---

## Summary

**Hybrid approach rationale**:

- **Local**: Full control, fast iteration, no cloud costs for heavy training
- **Production**: Lightweight adaptation, real-time response to drift
- **Fixed test set**: Valid model comparison across time
- **Champion/Challenger**: Conservative promotion (5% improvement threshold)

This strategy balances **production agility** (weekly fine-tuning) with **model quality**
(quarterly full retraining), ensuring the system adapts to recent patterns while maintaining
rigorous evaluation standards.
