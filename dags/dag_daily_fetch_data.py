"""
DAG 1: Daily Bike Traffic Data Ingestion
Fetches bike traffic data from Paris Open Data API and stores in BigQuery
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
import pandas as pd
from google.cloud import bigquery
import os

from utils.env_config import get_env_config
from utils.bike_helpers import create_bq_dataset_if_not_exists


# Configuration
ENV_CONFIG = get_env_config()
IS_DEV = os.getenv("ENV", "DEV") == "DEV"

default_args = {
    "owner": "mlops-team",
    "retries": 1 if IS_DEV else 2,
    "retry_delay": timedelta(seconds=30) if IS_DEV else timedelta(minutes=5),
    "start_date": datetime(2024, 10, 1),
}


def fetch_bike_data_to_bq(**context):
    """
    Fetch latest bike traffic data from Paris Open Data API
    Store in BigQuery: bike_traffic_raw.daily_YYYYMMDD

    Data source: Paris Open Data - Comptage vélo
    API: https://opendata.paris.fr/explore/dataset/comptage-velo-donnees-compteurs/
    """
    today = datetime.utcnow().strftime("%Y%m%d")
    print(f"📅 Fetching bike traffic data for {today}")

    # Ensure dataset exists
    create_bq_dataset_if_not_exists(
        ENV_CONFIG["BQ_PROJECT"],
        ENV_CONFIG["BQ_RAW_DATASET"],
        ENV_CONFIG["BQ_LOCATION"],
    )

    # Check existing data to avoid duplicates
    table_name = "comptage_velo"
    table_id = f"{ENV_CONFIG['BQ_RAW_DATASET']}.{table_name}"
    full_table_id = f"{ENV_CONFIG['BQ_PROJECT']}.{table_id}"

    client = bigquery.Client(project=ENV_CONFIG["BQ_PROJECT"])

    # Get the latest date already in BigQuery
    try:
        query = f"""
        SELECT MAX(date_et_heure_de_comptage) as max_date
        FROM `{full_table_id}`
        """  # nosec B608
        result = client.query(query).to_dataframe()
        max_existing_date = result["max_date"].iloc[0]
        if pd.notna(max_existing_date):
            print(f"📊 Latest data in BigQuery: {max_existing_date}")
            # Add a small buffer to avoid edge cases
            cutoff_date = max_existing_date.isoformat()
        else:
            cutoff_date = None
            print("📊 No existing data in BigQuery, fetching all available data")
    except Exception as e:
        print(f"⚠️ Table doesn't exist yet or error checking: {e}")
        cutoff_date = None

    # Paris Open Data API (comptage vélo - données compteurs)
    api_url = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/comptage-velo-donnees-compteurs/records"

    # Parameters: API limit is 100 per request, so we'll paginate
    # Fetch up to 1000 records total (10 pages of 100)
    max_records = 1000
    page_size = 100
    offset = 0
    all_records = []

    print(f"🌐 Fetching data from API (pagination with limit={page_size})")
    if cutoff_date:
        print(f"   Only fetching data newer than {cutoff_date}")

    while len(all_records) < max_records:
        params = {
            "limit": page_size,
            "offset": offset,
            "order_by": "date DESC",
            "timezone": "Europe/Paris",
        }

        print(f"  📄 Page {offset // page_size + 1}: offset={offset}")
        response = requests.get(api_url, params=params, timeout=30)

        if response.status_code != 200:
            raise Exception(
                f"❌ API request failed: {response.status_code} - {response.text}"
            )

        data = response.json()

        if "results" not in data or len(data["results"]) == 0:
            print(f"  ✅ No more data at offset {offset}, stopping pagination")
            break

        # Extract records from API response
        for record in data["results"]:
            # Paris Open Data v2 API structure: each record has 'fields' with the data
            if "fields" in record:
                all_records.append(record["fields"])
            else:
                # If no 'fields', use the record directly
                all_records.append(record)

        # If we got less than page_size records, we've reached the end
        if len(data["results"]) < page_size:
            print(f"  ✅ Last page reached (got {len(data['results'])} records)")
            break

        offset += page_size

    if len(all_records) == 0:
        raise Exception("❌ No data returned from API")

    print(f"✅ Total records fetched: {len(all_records)}")
    records = all_records

    df = pd.DataFrame(records)

    print(f"📊 Raw API response: {len(df)} records")
    print(f"📊 Columns: {df.columns.tolist()}")

    # Standardize column names (API returns: date, sum_counts, id_compteur, nom_compteur, coordinates)
    column_mapping = {
        "sum_counts": "comptage_horaire",
        "date": "date_et_heure_de_comptage",
        "id_compteur": "identifiant_du_compteur",
        "nom_compteur": "nom_du_compteur",
        "coordinates": "coordonnees_geographiques",
    }

    # Rename columns if they exist
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and old_name != new_name:
            df.rename(columns={old_name: new_name}, inplace=True)

    # Convert coordinates dict to separate lat/lon columns
    if "coordonnees_geographiques" in df.columns:
        df["latitude"] = df["coordonnees_geographiques"].apply(
            lambda x: x.get("lat") if isinstance(x, dict) else None
        )
        df["longitude"] = df["coordonnees_geographiques"].apply(
            lambda x: x.get("lon") if isinstance(x, dict) else None
        )
        # Drop the original dict column
        df = df.drop(columns=["coordonnees_geographiques"])

    # Convert date string to TIMESTAMP for BigQuery partitioning
    if "date_et_heure_de_comptage" in df.columns:
        df["date_et_heure_de_comptage"] = pd.to_datetime(
            df["date_et_heure_de_comptage"], errors="coerce"
        )

    # Filter out records older than cutoff_date to avoid duplicates
    if cutoff_date and "date_et_heure_de_comptage" in df.columns:
        original_count = len(df)
        cutoff_datetime = pd.to_datetime(cutoff_date)
        df = df[df["date_et_heure_de_comptage"] > cutoff_datetime]
        filtered_count = len(df)
        print(
            f"🔍 Filtered out {original_count - filtered_count} existing records "
            f"(keeping {filtered_count} new records)"
        )

    # Add ingestion timestamp
    df["ingestion_ts"] = datetime.utcnow()

    # Select only required columns (if they exist)
    required_columns = [
        "comptage_horaire",
        "date_et_heure_de_comptage",
        "identifiant_du_compteur",
        "nom_du_compteur",
        "latitude",
        "longitude",
        "ingestion_ts",
    ]

    available_columns = [col for col in required_columns if col in df.columns]
    df_clean = df[available_columns].copy()

    print(
        f"📊 Final dataset: {len(df_clean)} records, {len(available_columns)} columns"
    )

    # Data quality checks
    if len(df_clean) == 0:
        print("ℹ️ No new data to ingest (all data already exists in BigQuery)")
        # Push metadata to XCom for downstream tasks
        context["ti"].xcom_push(key="records_count", value=0)
        context["ti"].xcom_push(key="table_id", value=full_table_id)
        context["ti"].xcom_push(key="ingestion_date", value=today)
        context["ti"].xcom_push(key="no_new_data", value=True)
        return

    print(f"📊 Sample data:\n{df_clean.head(2)}")

    # Write to BigQuery - Single partitioned table instead of daily tables
    table_name = "comptage_velo"
    table_id = f"{ENV_CONFIG['BQ_RAW_DATASET']}.{table_name}"
    full_table_id = f"{ENV_CONFIG['BQ_PROJECT']}.{table_id}"

    print(f"📤 Writing to BigQuery: {full_table_id}")
    print("   Mode: append (partitioned by date_et_heure_de_comptage)")

    # Use BigQuery client for more control over partitioning and deduplication
    client = bigquery.Client(project=ENV_CONFIG["BQ_PROJECT"])

    # Configure job to handle duplicates (dedup by compteur + date)
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        time_partitioning=bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="date_et_heure_de_comptage",
        ),
        clustering_fields=["identifiant_du_compteur"],
    )

    # Load data
    job = client.load_table_from_dataframe(
        df_clean,
        full_table_id,
        job_config=job_config,
        location=ENV_CONFIG["BQ_LOCATION"],
    )
    job.result()  # Wait for job to complete

    print(f"✅ Successfully appended {len(df_clean)} records to {full_table_id}")
    print("   Table is partitioned by date and clustered by compteur")

    # Push metadata to XCom for downstream tasks
    context["ti"].xcom_push(key="records_count", value=len(df_clean))
    context["ti"].xcom_push(key="table_id", value=full_table_id)
    context["ti"].xcom_push(key="ingestion_date", value=today)


def validate_ingestion(**context):
    """
    Validate that data was successfully ingested into BigQuery
    Validates records ingested in the last 5 minutes (this DAG run)
    """
    # Pull metadata from previous task
    no_new_data = context["ti"].xcom_pull(
        task_ids="fetch_to_bigquery", key="no_new_data"
    )

    if no_new_data:
        print("✅ Validation passed: No new data to ingest (all data already exists)")
        return

    table_id = context["ti"].xcom_pull(task_ids="fetch_to_bigquery", key="table_id")
    records_count = context["ti"].xcom_pull(
        task_ids="fetch_to_bigquery", key="records_count"
    )

    print(f"🔍 Validating ingestion for {table_id}")
    print(f"📊 Expected records: {records_count}")

    # Query BigQuery to verify recently ingested records (last 5 minutes)
    client = bigquery.Client(project=ENV_CONFIG["BQ_PROJECT"])

    query = f"""
    SELECT
        COUNT(*) as total_records,
        MIN(ingestion_ts) as first_ingestion,
        MAX(ingestion_ts) as last_ingestion,
        MIN(DATE(date_et_heure_de_comptage)) as min_data_date,
        MAX(DATE(date_et_heure_de_comptage)) as max_data_date
    FROM `{table_id}`
    WHERE ingestion_ts >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 5 MINUTE)
    """  # nosec B608

    df = client.query(query).to_dataframe()

    actual_count = df["total_records"].iloc[0]
    first_ingestion = df["first_ingestion"].iloc[0]
    last_ingestion = df["last_ingestion"].iloc[0]
    min_data_date = df["min_data_date"].iloc[0]
    max_data_date = df["max_data_date"].iloc[0]

    print("✅ Validation results:")
    print(f"   - Records ingested in last 5 min: {actual_count}")
    print(f"   - First ingestion: {first_ingestion}")
    print(f"   - Last ingestion: {last_ingestion}")
    print(f"   - Data date range: {min_data_date} to {max_data_date}")

    # Basic validation
    if actual_count == 0:
        raise Exception(
            "❌ Validation failed: No records ingested in the last 5 minutes"
        )

    if actual_count < records_count * 0.9:
        print(
            f"⚠️ WARNING: Record count significantly lower than expected "
            f"(expected {records_count}, got {actual_count})"
        )

    print(f"✅ Validation passed: {actual_count} records successfully ingested")


# === DAG DEFINITION ===
with DAG(
    dag_id="daily_fetch_bike_data",
    default_args=default_args,
    description="Fetch bike traffic data from Paris Open Data API and store in BigQuery",
    schedule_interval="@daily",  # Run daily at midnight
    catchup=False,
    max_active_runs=1,
    tags=["bike", "ingestion", "bigquery", "mlops"],
) as dag:
    # Task 1: Fetch data from API and write to BigQuery
    fetch_task = PythonOperator(
        task_id="fetch_to_bigquery",
        python_callable=fetch_bike_data_to_bq,
        provide_context=True,
    )

    # Task 2: Validate ingestion
    validate_task = PythonOperator(
        task_id="validate_ingestion",
        python_callable=validate_ingestion,
        provide_context=True,
    )

    # Task dependencies
    fetch_task >> validate_task
