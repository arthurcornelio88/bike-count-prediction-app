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

    # Paris Open Data API (comptage vélo - données compteurs)
    api_url = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/comptage-velo-donnees-compteurs/records"

    # Parameters: API limit is 100 per request, so we'll paginate
    # Fetch up to 1000 records total (10 pages of 100)
    max_records = 1000
    page_size = 100
    offset = 0
    all_records = []

    print(f"🌐 Fetching data from API (pagination with limit={page_size})")

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
    print(f"📊 Sample data:\n{df_clean.head(2)}")

    # Data quality checks
    if len(df_clean) == 0:
        raise Exception("❌ No valid data after cleaning")

    # Write to BigQuery
    table_id = f"{ENV_CONFIG['BQ_RAW_DATASET']}.daily_{today}"
    full_table_id = f"{ENV_CONFIG['BQ_PROJECT']}.{table_id}"

    print(f"📤 Writing to BigQuery: {full_table_id}")

    df_clean.to_gbq(
        destination_table=table_id,
        project_id=ENV_CONFIG["BQ_PROJECT"],
        if_exists="replace",  # Replace if table already exists for today
        location=ENV_CONFIG["BQ_LOCATION"],
    )

    print(f"✅ Successfully ingested {len(df_clean)} records into {full_table_id}")

    # Push metadata to XCom for downstream tasks
    context["ti"].xcom_push(key="records_count", value=len(df_clean))
    context["ti"].xcom_push(key="table_id", value=full_table_id)
    context["ti"].xcom_push(key="ingestion_date", value=today)


def validate_ingestion(**context):
    """
    Validate that data was successfully ingested into BigQuery
    Performs basic data quality checks
    """
    # Pull metadata from previous task
    table_id = context["ti"].xcom_pull(task_ids="fetch_to_bigquery", key="table_id")
    records_count = context["ti"].xcom_pull(
        task_ids="fetch_to_bigquery", key="records_count"
    )

    print(f"🔍 Validating ingestion for {table_id}")
    print(f"📊 Expected records: {records_count}")

    # Query BigQuery to verify
    client = bigquery.Client(project=ENV_CONFIG["BQ_PROJECT"])

    query = f"""
    SELECT
        COUNT(*) as total_records,
        MIN(ingestion_ts) as first_ingestion,
        MAX(ingestion_ts) as last_ingestion
    FROM `{table_id}`
    """  # nosec B608

    df = client.query(query).to_dataframe()

    actual_count = df["total_records"].iloc[0]
    first_ingestion = df["first_ingestion"].iloc[0]
    last_ingestion = df["last_ingestion"].iloc[0]

    print("✅ Validation results:")
    print(f"   - Actual records in BQ: {actual_count}")
    print(f"   - First ingestion: {first_ingestion}")
    print(f"   - Last ingestion: {last_ingestion}")

    # Basic validation
    if actual_count != records_count:
        print(
            f"⚠️ WARNING: Record count mismatch (expected {records_count}, got {actual_count})"
        )

    if actual_count == 0:
        raise Exception("❌ Validation failed: No records in BigQuery table")

    print(f"✅ Validation passed for {table_id}")


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
