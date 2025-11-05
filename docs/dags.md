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

- **Recent Data Selection**: Queries last 24 hours of data for prediction

  ```sql
  SELECT * FROM bike_traffic_raw.comptage_velo
  WHERE date_et_heure_de_comptage >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
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
model_version                 STRING    - Model run_id from MLflow
prediction_ts                 TIMESTAMP - When prediction was made
identifiant_du_compteur       STRING    - Counter ID
date_et_heure_de_comptage     TIMESTAMP - Data timestamp
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

**Real Production Run Results (2025-11-03, force=true, test_mode=false)**:

- **Baseline Test Set** (181K samples, fixed reference):
  - New model R²: **0.8161** (excellent!)
  - Champion R²: **0.3091** (old compteurs distribution)
  - Improvement: **+0.5070** (massive improvement)
  - Baseline regression: ✅ NO (R² > 0.60)

- **Current Test Set** (20% of fresh data, new distribution):
  - New model R²: **0.9087**
  - Champion R²: **0.7214**
  - Improvement: **+0.1872** (significant improvement)

- **Training Metrics** (on train_baseline):
  - R²: **0.8666** (strong generalization)

- **Deployment Decision**: ✅ **DEPLOY** - Model improved on both test sets
  - No baseline regression detected
  - Current distribution: +18.72% improvement
  - Baseline distribution: +50.70% improvement

**Champion Promotion Flow**:

When deployment decision contains "deploy", the DAG automatically promotes the new model:

1. Calls `POST /promote_champion` with run_id, model_type, env
2. FastAPI updates `summary.json` in GCS:
   - Sets `is_champion=true` for new model
   - Sets `is_champion=false` for previous champion
3. Clears FastAPI model cache to force reload
4. Next `/predict` call loads new champion via `get_best_model_from_summary()`

This ensures `dag_daily_prediction` automatically uses the promoted model without manual intervention.

![fine_tune_model screenshot](/docs/img/dag3_finetuning2.png)

#### 3.5 `end_monitoring`

Final task that consolidates all run metadata and writes an audit record to BigQuery (`monitoring_audit.logs`).

**Audit Fields**:

- Drift detected, RMSE, R², fine-tune triggered & success flags
- Model improvement delta, model URI, run id, decision rationale
- `baseline_regression` & `double_evaluation_enabled` indicators

**Real Production Run Summary (2025-11-03)**:

The audit log captured the complete monitoring cycle showing:

- **Drift Detection**: ✅ YES (50% drift share detected)
- **Current Champion**: RMSE=32.25, R²=0.7214 (excellent production performance)
- **Fine-tuning**: ✅ SUCCESS with double evaluation enabled
- **Model Improvement**: +0.1872 on current test set
- **Deployment Decision**: ✅ DEPLOY - New model promoted to production
- **Model URI**: `gs://df_traffic_cyclist1/mlflow-artifacts/1/849e4ce4402dd7dba3e94f1888663304/artifacts/model`

This demonstrates the complete MLOps loop: drift detection → validation → training → double evaluation → deployment decision → audit logging.

![end_monitoring screenshot](/docs/img/dag3_end_monitoring2.png)

### Monitoring Notes

- **Schema Drift Handling**: If reference/current columns diverge (e.g., translated column names), the monitor treats it as full drift and triggers retraining.
- **Timeouts**: Fine-tune task has a 15-minute execution timeout and HTTP timeout of 10 minutes to accommodate large training sets.
- **Test Mode**: DAG accepts `dag_run.conf.test_mode` to force the backend `/train` endpoint into lightweight datasets for DEV.

### Data Flow Summary

```text
Reference CSV (data/reference_data.csv)      BigQuery current window (7 days)
                 │                                      │
                 └── monitor_drift ──── drift summary ──┘
                                   │
                      validate_model (RMSE / R²)
                                   │
                      decide_fine_tune (branch)
                       │                       │
         fine_tune_model (FastAPI /train)   end_monitoring
                       │                       │
            MLflow + GCS artifacts      BigQuery monitoring_audit.logs
```

### Runbook Checklist

- [ ] Monitor drift report shows `drift_share` and column overlap status
- [ ] Validation task logs RMSE/R² and sample count
- [ ] Decision task prints branch rationale (drift, metrics, force flag)
- [ ] Fine-tuning logs include sliding-window status + double evaluation metrics
- [ ] Audit record written with `fine_tune_triggered` and `model_improvement`
