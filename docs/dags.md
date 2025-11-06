# DAGs Documentation

This document describes the Airflow DAGs used in the bike traffic prediction project.

## 1. Daily Fetch Bike Data DAG

**DAG ID**: `daily_fetch_bike_data`

**Schedule**: Daily at 6:00 AM UTC

**Purpose**: Fetch bike traffic data from Paris Open Data API and load it into BigQuery

### Architecture Overview

This DAG implements an idempotent data ingestion pipeline that:

- Fetches up to 1000 recent records from the Paris Open Data API
- Deduplicates data to prevent inserting existing records
- Loads new data into a partitioned BigQuery table
- Validates successful ingestion

### Tasks

#### 1.1 `fetch_to_bigquery`

Fetches bike traffic data from the Paris Open Data API and loads it into BigQuery.

**Key Features**:

- **API Pagination**: The Paris API limits responses to 100 records per request. The task automatically
  paginates to fetch up to 1000 records (10 pages).

- **Deduplication Logic**: To prevent duplicate data on subsequent runs:
  1. Queries BigQuery for the maximum existing `date_et_heure_de_comptage`
  2. Filters fetched records to only include data newer than this cutoff
  3. If no new data exists, exits early without writing to BigQuery

  Example logs showing deduplication:

  ```text
  First run:
  📊 Latest data in BigQuery: (table doesn't exist yet)
  ✅ Successfully appended 1000 records

  Second run (same data):
  📊 Latest data in BigQuery: 2025-10-26 22:00:00+00:00
  🔍 Filtered out 1000 existing records (keeping 0 new records)
  ℹ️ No new data to ingest (all data already exists in BigQuery)
  ```

- **Data Transformations**:
  - Maps API field names to standardized column names
  - Extracts `latitude` and `longitude` from nested `coordinates` dict
  - Converts date strings to TIMESTAMP for BigQuery partitioning
  - Adds `ingestion_ts` timestamp for tracking when data was loaded

- **BigQuery Table Structure**:
  - **Table**: `bike_traffic_raw.comptage_velo` (single partitioned table)
  - **Partitioning**: Daily partitions on `date_et_heure_de_comptage` field
  - **Clustering**: Clustered by `identifiant_du_compteur` for efficient queries
  - **Write Mode**: APPEND (with deduplication to prevent duplicates)

**Schema**:

```text
comptage_horaire              INTEGER   - Hourly bike count
date_et_heure_de_comptage     TIMESTAMP - Date and time of count (partition field)
identifiant_du_compteur       STRING    - Counter ID (clustering field)
nom_du_compteur               STRING    - Counter name
latitude                      FLOAT     - GPS latitude
longitude                     FLOAT     - GPS longitude
ingestion_ts                  TIMESTAMP - When record was ingested
```

**XCom Outputs**:

- `table_id`: Full BigQuery table ID
- `records_count`: Number of records ingested (0 if no new data)
- `ingestion_date`: Date of ingestion (YYYYMMDD format)
- `no_new_data`: Boolean flag (true if all data already exists)

![Fetch to BigQuery](/docs/img/fetch_to_bq.png)

#### 1.2 `validate_ingestion`

Validates that data was successfully ingested into BigQuery.

**Validation Logic**:

1. **No New Data Case**: If the fetch task found no new data, validation passes gracefully:

   ```text
   ✅ Validation passed: No new data to ingest (all data already exists)
   ```

2. **Normal Validation**: Queries BigQuery for records ingested in the last 5 minutes:
   - Checks total record count matches expected count
   - Verifies ingestion timestamp range
   - Reports data date range

   Example successful validation:

   ```text
   🔍 Validating ingestion for datascientest-460618.bike_traffic_raw.comptage_velo
   📊 Expected records: 1000
   ✅ Validation results:
      - Records ingested in last 5 min: 1000
      - First ingestion: 2025-10-28 00:09:14.838861+00:00
      - Last ingestion: 2025-10-28 00:09:14.838861+00:00
      - Data date range: 2025-10-26 to 2025-10-26
   ✅ Validation passed: 1000 records successfully ingested
   ```

**Failure Conditions**:

- Record count mismatch (expected vs actual)
- No records found with recent ingestion timestamp

![Validate Ingestion](/docs/img/validate_ingestion.png)

### Configuration

**Retry Policy**:

- **Development** (`ENV=DEV`): 1 retry, 30-second delay
- **Production** (`ENV=PROD`): 2 retries, 5-minute delay

**Environment Variables Required**:

- `BQ_PROJECT`: BigQuery project ID
- `BQ_RAW_DATASET`: Raw data dataset name (default: `bike_traffic_raw`)
- `ENV`: Environment (DEV/PROD)

### Data Flow Summary

```text
Paris Open Data API
    ↓ (pagination: 10 pages × 100 records)
Fetch Task (with deduplication)
    ↓ (filter out existing data)
BigQuery Table (partitioned + clustered)
    ↓ (validation query)
Validate Task
    ✅ Success
```

### Idempotency

This DAG is fully idempotent - it can be run multiple times per day without creating duplicate data.
The deduplication logic ensures that only new records are inserted, making the pipeline safe for manual
reruns or schedule adjustments.

---

## 2. Daily Prediction DAG

- **DAG ID**: `daily_prediction`
- **Schedule**: Daily at 7:00 AM UTC (1h after data ingestion)
- **Purpose**: Generate predictions for recent bike traffic data using ML models and store results in BigQuery

### Architecture Overview

This DAG implements a prediction pipeline that:

- Reads recent data from the partitioned raw table
- Calls the ML API `/predict` endpoint
- Handles data drift scenarios (unknown compteurs)
- Stores predictions with quality metrics in BigQuery
- Validates prediction results

### Tasks

#### 2.1 `check_raw_data`

Verifies that raw data exists before attempting predictions.

**Key Features**:

- **Partitioned Table Query**: Queries the `comptage_velo` table for recent data (last 7 days)
- **Data Availability Check**: Ensures there's sufficient data to make predictions
- **Lookback Window**: Checks last 7 days to ensure data freshness

**Validation Logic**:

```sql
SELECT COUNT(*) as row_count,
       MIN(date_et_heure_de_comptage) as min_date,
       MAX(date_et_heure_de_comptage) as max_date
FROM bike_traffic_raw.comptage_velo
WHERE date_et_heure_de_comptage >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
```

**XCom Outputs**:

- `raw_table`: Full BigQuery table ID
- `row_count`: Number of records available
- `max_date`: Latest data timestamp

**Success Example**:

```text
🔍 Checking if raw data exists in partitioned table
✅ Raw data found: 2000 rows in datascientest-460618.bike_traffic_raw.comptage_velo
📊 Data range: 2025-10-26 12:00:00+00:00 to 2025-10-28 22:00:00+00:00
```

![alt text](/docs/img/check_raw_data.png)

#### 2.2 `predict_daily_data`

Generates predictions by calling the ML API and stores results in BigQuery.

**Key Features**:

- **Recent Data Selection**: Queries last 48 hours of data for prediction (allows ingestion delays)

  ```sql
  SELECT * FROM bike_traffic_raw.comptage_velo
  WHERE date_et_heure_de_comptage >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 48 HOUR)
  ORDER BY date_et_heure_de_comptage DESC
  LIMIT 500
  ```

- **Data Transformation for API Compatibility**:
  - Reconstructs `coordonnées_géographiques` from separate `latitude`/`longitude` columns
  - Format: "lat,lon" as string (e.g., "48.8566,2.3522")
  - Converts timestamps to strings for JSON serialization

- **ML API Call**: POSTs to `/predict` endpoint with payload:

  ```json
  {
    "records": [...],
    "model_type": "rf",
    "metric": "r2"
  }
  ```

- **Data Drift Handling**: API backend handles unknown compteurs (new bike counters):
  - Maps unknown categories to fallback compteur
  - Logs warnings when data drift detected
  - Predictions for unknown compteurs are approximate

- **Prediction Metrics Calculation**:
  - RMSE (Root Mean Square Error)
  - MAE (Mean Absolute Error)
  - R² (coefficient of determination)

**Prediction Storage**:

Table: `bike_traffic_predictions.daily_YYYYMMDD` (daily tables)

Schema:

```text
comptage_horaire              INTEGER   - Actual value (if available)
prediction                    FLOAT     - Model prediction
model_type                    STRING    - Model used (rf/nn)
model_version                 STRING    - Champion model run_id from MLflow (for traceability)
prediction_ts                 TIMESTAMP - When prediction was made
identifiant_du_compteur       STRING    - Counter ID
date_et_heure_de_comptage     TIMESTAMP - Data timestamp
```

**Champion Tracking** (Phase 5 - MLOps Monitoring):

Every prediction is now tagged with the exact champion model that produced it:

- The `/predict` API returns model metadata including `run_id`, `is_champion`, `r2`, and `rmse`
- The `model_version` field stores the champion's MLflow `run_id` (instead of a timestamp)
- This enables full traceability from prediction → model version → training metrics
- Champion metadata is also pushed to XCom for audit purposes

Example API response:

```json
{
  "predictions": [123.4, 234.5, ...],
  "model_metadata": {
    "run_id": "849e4ce4402dd7dba3e94f1888663304",
    "is_champion": true,
    "model_type": "rf",
    "r2": 0.9087,
    "rmse": 32.25
  }
}
```

![alt text](/docs/img/predictions_bq.png)

**Success Example**:

```text
✅ Fetched 291 records from BigQuery
🌐 Calling prediction API: http://regmodel-backend:8000/predict
✅ Received 291 predictions from API
📤 Writing predictions to BigQuery: bike_traffic_predictions.daily_20251029
✅ Successfully stored 291 predictions

📈 Prediction Metrics:
   - RMSE: 32.70
   - MAE: 24.17
   - R²: 0.7856
```

![alt text](/docs/img/predict_daily_data.png)

**Alerting**:

![alt text](/docs/img/predictions_discord.png)

#### 2.3 `validate_predictions`

Validates that predictions were successfully stored in BigQuery.

**Validation Logic**:

- Queries BigQuery for predictions in the daily table
- Checks record count matches expected count
- Calculates prediction statistics (avg, min, max)

**Success Example**:

```text
🔍 Validating predictions for bike_traffic_predictions.daily_20251029
📊 Expected predictions: 291
✅ Validation results:
   - Actual predictions in BQ: 291
   - Average prediction: 61.05
   - Min/Max prediction: 37.97 / 401.07
✅ Validation passed
```

![alt text](/docs/img/validate_predictions.png)

**Failure Conditions**:

- Record count mismatch
- No predictions found in table
- Prediction values outside reasonable bounds

### Configuration

**Retry Policy**:

- **Development** (`ENV=DEV`): 1 retry, 30-second delay
- **Production** (`ENV=PROD`): 2 retries, 5-minute delay

**Environment Variables Required**:

- `BQ_PROJECT`: BigQuery project ID
- `BQ_RAW_DATASET`: Raw data dataset (default: `bike_traffic_raw`)
- `BQ_PREDICT_DATASET`: Predictions dataset (default: `bike_traffic_predictions`)
- `API_URL`: ML API endpoint (default: `http://regmodel-backend:8000`)
- `ENV`: Environment (DEV/PROD)

### Data Flow Summary

```text
BigQuery Raw Table (last 24h)
    ↓ (query recent data)
check_raw_data
    ↓ (verify availability)
predict_daily_data
    ↓ (transform data format)
ML API /predict
    ↓ (RF model predictions)
BigQuery Predictions Table
    ↓ (validation query)
validate_predictions
    ✅ Success
```

### Data Drift Handling

**Problem**: New bike counters appear that weren't in training data

**Solution**: Backend maps unknown compteurs to known fallback category

**Impact**:

- ✅ Pipeline continues to work (no crashes)
- ⚠️ Predictions for unknown compteurs are approximate/biased
- 📊 Logged warnings indicate need for model retraining
- 🔄 DAG 3 will detect drift and trigger retraining

**Example Warning**:

```text
⚠️ DATA DRIFT: 15 unknown compteurs (3 unique)
   Fallback: 'Totem 73 boulevard de Sébastopol'
   Unknown: ['147 avenue d'Italie [Bike]', ...]
```

### Quality Metrics

Based on recent test run (2025-10-29):

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Records processed | 291 | Last 24 hours of data |
| RMSE | 32.70 | Average error of ~33 bikes/hour |
| MAE | 24.17 | Median error of ~24 bikes/hour |
| R² | 0.7856 | 78.56% variance explained (excellent) |
| Avg prediction | 61.05 | Typical hourly bike count |
| Prediction range | 37.97 - 401.07 | Min to max predictions |

**Performance**: R² of 0.7856 indicates the model explains most of the variance in bike traffic,
which is very good for time-series prediction with external factors.

---

## 3. Monitor & Fine-Tune DAG

- **DAG ID**: `monitor_and_fine_tune`
- **Schedule**: Weekly on Monday at 08:00 UTC (configurable)
- **Purpose**: Detect data/schema drift, validate current champion model, and trigger fine-tuning when needed.

### Architecture Overview

This DAG stitches together monitoring, evaluation, and retraining:

- Loads reference vs current datasets to run schema & distribution drift checks
- Scores champion model performance on the latest BigQuery ground-truth window
- Applies decision logic (drift or degraded metrics) to decide on retraining
- Fine-tunes the regression model via the FastAPI `/train` endpoint with double evaluation
- Logs every run into BigQuery `monitoring_audit.logs` for traceability

### Tasks

#### 3.1 `monitor_drift`

Compares reference data (`data/reference_data.csv`) and current batch (last 7 days from BigQuery) via the `/monitor` API.

**Highlights**:

- Uses Evidently 0.5.0 to generate drift summary
- Treats missing common columns as schema drift (`drift_share = 1.0`)
- Outputs detailed summary to XCom (`drift_summary` & `drift_detected`)

![monitor_drift screenshot placeholder](/docs/img/dag3_monitor_drift.png)

#### 3.2 `validate_model`

Pulls 7 days of predictions vs actuals and computes RMSE/R² using BigQuery windowed join.

**Highlights**:

- Ensures current champion is still performant (default thresholds: R² ≥ 0.65, RMSE ≤ 60)
- Pushes metrics to XCom for decision step (`r2`, `rmse`, `validation_samples`)

![validate_model screenshot placeholder](/docs/img/dag3_validate_model.png)

#### 3.3 `decide_fine_tune`

BranchPythonOperator applying **Hybrid Drift Management Strategy** (combining proactive and reactive approaches):

**Decision Logic (Priority Order):**

1. **Force flag** (`dag_run.conf.force_fine_tune`): Overrides all logic for testing
2. **REACTIVE trigger** (R² < 0.65 OR RMSE > 60): Immediate retraining when metrics are critically poor
3. **PROACTIVE trigger** (drift ≥ 50% AND R² < 0.70): Preventive retraining when high drift + metrics declining
4. **WAIT** (drift ≥ 30% BUT R² ≥ 0.70): Monitor closely, no retraining yet (model handles drift via `handle_unknown='ignore'`)
5. **ALL GOOD** (drift < 30% AND good metrics): Continue monitoring, no action

**Thresholds:**

- R² critical: 0.65 (below = immediate retrain)
- R² warning: 0.70 (below + high drift = proactive retrain)
- RMSE threshold: 60.0
- Drift critical: 50% (high drift)
- Drift warning: 30% (moderate drift)

**Strategy Benefits:**

- ✅ Avoids unnecessary retraining when model handles drift well (cost efficiency)
- ✅ Catches degradation early with proactive trigger (performance)
- ✅ Responds immediately to critical performance issues (reliability)

![decide_fine_tune screenshot placeholder](/docs/img/dag3_decide_fine_tune.png)

#### 3.4 `fine_tune_model`

Invokes the FastAPI `/train` endpoint with `data_source="baseline"` and optional `current_data` payload (sliding window).

**Highlights**:

- Fetches 30 days from BigQuery and submits 2k samples (configurable)
- Sliding-window strategy: combines baseline training data with fresh samples when schemas align
- Double evaluation: champion vs new model on `test_baseline` + holdout current data
- Writes result metrics, run ids, and deployment decision back to XCom for auditing
- **Champion Promotion**: When decision="deploy", calls `/promote_champion` endpoint to update `summary.json`

**Real Production Run Results (2025-11-05, test_mode=false)**:

![fine_tune_model screenshot](/docs/img/dag3_finetuning_1.png)

![alt text](/docs/img/dag3_finetuning_2.png)

- **Baseline Test Set** (181K samples, fixed reference):
  - New model R²: **0.8179** (excellent!)
  - Old champion R²: **0.1945** (previous model trained on limited data)
  - Improvement: **+0.6234** (massive improvement)
  - RMSE: **72.86**
  - MAE: **34.01**
  - Baseline regression: ✅ NO (R² > 0.60)

- **Current Test Set** (20% of fresh data, new distribution):
  - New model R²: **0.8167**
  - Old champion R²: **0.5847**
  - Improvement: **+0.2320** (significant improvement)
  - RMSE: **47.26**
  - MAE: **36.46**

- **Training Metrics** (on merged train sets, 726K samples):
  - R²: **0.8658** (strong generalization)
  - RMSE: **64.89**

- **Deployment Decision**: ✅ **DEPLOY** - Model improved on both test sets
  - No baseline regression detected
  - Current distribution: +23.20% improvement
  - Baseline distribution: +62.34% improvement

**Champion Promotion Flow** (Phase 5 - MLOps Monitoring):

When deployment decision contains "deploy", the DAG automatically promotes the new model:

1. Calls `POST /promote_champion` with run_id, model_type, env
2. FastAPI updates `summary.json` in GCS:
   - Sets `is_champion=true` for new model
   - Sets `is_champion=false` for previous champion
3. Clears FastAPI model cache (both model and metadata) to force reload
4. Sends Discord notification with champion details (run_id, R² scores, improvement delta)
5. Next `/predict` call loads new champion via `get_best_model_from_summary()`

**Discord Notification**:

When a new champion is promoted, the team receives a Discord alert with:

- Model type and run_id (first 12 chars)
- R² scores on both test_current and test_baseline
- Improvement delta over previous champion
- RMSE metric

Example Discord notification:

![alt text](/docs/img/dag3_training_champion.png)

This ensures `dag_daily_prediction` automatically uses the promoted model without manual intervention, and the team is immediately aware of production model changes.

#### 3.5 `validate_new_champion`

**NEW TASK** (Added 2025-11-05): Re-validates the newly promoted champion model on recent production data.

**Why This Task is Critical**:

When a new model is promoted to champion, we face a **metrics staleness problem**:

1. `validate_model` (task 3.2) runs **BEFORE** training and validates the **OLD champion**
2. After promotion, BigQuery `monitoring_audit.logs` still contains the OLD champion's metrics
3. Prometheus/Grafana read from BigQuery → dashboards show **stale metrics** for a model that's no longer in production
4. Without re-validation, monitoring dashboards would display R²=0.5847 (old champion) instead of the actual new champion's performance

**Solution**: `validate_new_champion` validates the **NEW champion** after promotion and updates BigQuery with fresh metrics.

**Execution Logic**:

- **Conditional execution**: Only runs if `champion_promoted=True` (checked via XCom from `fine_tune_model`)
- If no promotion occurred, task exits early with "⏭️ No champion promotion, skipping validation"
- If promotion occurred, proceeds to validate the new champion

**Validation Process**:

1. Pulls new champion's `run_id` from XCom
2. Queries BigQuery for predictions vs actuals (last 7 days) - same logic as `validate_model`
3. Calculates RMSE, MAE, R² on production data
4. Pushes metrics to XCom: `new_champion_rmse`, `new_champion_r2`, `new_champion_mae`
5. Sends Discord notification (info level, not critical)

**Example Output**:

![alt text](/docs/img/dag3_validate_champion.png)

**Integration with end_monitoring**:

The `end_monitoring` task (3.6) checks for these new metrics:

```python
# Check if we have new champion validation metrics
new_champion_rmse = context["ti"].xcom_pull(
    task_ids="validate_new_champion", key="new_champion_rmse"
)
new_champion_r2 = context["ti"].xcom_pull(
    task_ids="validate_new_champion", key="new_champion_r2"
)

# If new champion was validated, use those metrics
if new_champion_rmse is not None and new_champion_r2 is not None:
    rmse = new_champion_rmse
    r2 = new_champion_r2
else:
    # Use old champion metrics from validate_model
    rmse = context["ti"].xcom_pull(task_ids="validate_model", key="rmse")
    r2 = context["ti"].xcom_pull(task_ids="validate_model", key="r2")
```

This ensures BigQuery receives the **correct champion metrics**, which Prometheus then exposes to Grafana.

**Impact**:

✅ Grafana dashboards immediately reflect new champion's performance
✅ Monitoring alerts trigger on actual production model metrics
✅ Team has visibility into post-promotion model behavior
✅ Audit trail tracks both pre- and post-promotion metrics

#### 3.6 `end_monitoring`

Final task that consolidates all run metadata and writes an audit record to BigQuery (`monitoring_audit.logs`).

**Audit Fields**:

- Drift detected, RMSE, R², fine-tune triggered & success flags
- Model improvement delta, model URI, run id, decision rationale
- `baseline_regression` & `double_evaluation_enabled` indicators

**Real Production Run Summary (2025-11-05)**:

The audit log captured the complete monitoring cycle showing:

- **Drift Detection**: ✅ YES (50% drift share detected)
- **Current Champion**: RMSE=78.30, R²=0.5847 (degraded performance, below critical threshold)
- **Fine-tuning**: ✅ SUCCESS with double evaluation enabled
- **Model Improvement**: +0.2320 on current test set, +0.6234 on baseline test set
- **Training Details**: 726,192 samples (724K baseline + 1.6K current via sliding window)
- **Deployment Decision**: ✅ DEPLOY - New model promoted to production
- **Model URI**: `gs://df_traffic_cyclist1/mlflow-artifacts/1/fc7d64d16b3b48aebf94512933ad92c9/artifacts/model`

This demonstrates the complete MLOps loop: drift detection → validation → training → double evaluation → deployment decision → champion promotion → audit logging.

![end_monitoring screenshot](/docs/img/dag3_end_monitoring2.png)

### Monitoring Notes

- **Schema Drift Handling**: If reference/current columns diverge (e.g., translated column names), the monitor treats it as full drift and triggers retraining.
- **Timeouts**: Fine-tune task has a 15-minute execution timeout and HTTP timeout of 10 minutes to accommodate large training sets.
- **Test Mode**: DAG accepts `dag_run.conf.test_mode` to force the backend `/train` endpoint into lightweight datasets for DEV.

### Data Flow Summary

**Updated Flow (with validate_new_champion)**:

```text
Reference CSV (data/reference_data.csv)      BigQuery current window (7 days)
                 │                                      │
                 └── monitor_drift ──── drift summary ──┘
                                   │
                      validate_model (OLD champion RMSE / R²)
                                   │
                      decide_fine_tune (branch)
                       │                                          │
         ┌─────────────┴────────────────┐                        │
         │                              │                        │
   fine_tune_model            end_monitoring (no training)       │
   (train + promote)                    │                        │
         │                              │                        │
   validate_new_champion                │                        │
   (NEW champion metrics)               │                        │
         │                              │                        │
         └──────────────┬───────────────┘                        │
                        │                                         │
                end_monitoring ─────────────────────────────────┘
                (uses NEW champion metrics if promotion,
                 else uses OLD champion metrics)
                        │
           BigQuery monitoring_audit.logs
                        │
                  Prometheus/Grafana
                  (shows correct champion R²)
```

**Key Flow Changes**:

1. `validate_model` → validates **OLD champion** (before training)
2. `fine_tune_model` → trains challenger, promotes if better
3. `validate_new_champion` → validates **NEW champion** (after promotion, conditional)
4. `end_monitoring` → writes correct metrics to BigQuery:
   - If promotion → uses `validate_new_champion` metrics
   - If no promotion → uses `validate_model` metrics
5. Prometheus reads BigQuery → Grafana displays current champion's actual performance

---
