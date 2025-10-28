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
