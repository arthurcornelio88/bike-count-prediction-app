"""
DAG 2: Daily Bike Traffic Predictions
Reads data from BigQuery, calls /predict endpoint, stores predictions back to BigQuery
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
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


def check_raw_data_exists(**context):
    """
    Check if raw data exists for today in BigQuery
    If not, skip prediction (no data to predict on)
    """
    today = datetime.utcnow().strftime("%Y%m%d")
    raw_table = (
        f"{ENV_CONFIG['BQ_PROJECT']}.{ENV_CONFIG['BQ_RAW_DATASET']}.daily_{today}"
    )

    print(f"🔍 Checking if raw data exists: {raw_table}")

    client = bigquery.Client(project=ENV_CONFIG["BQ_PROJECT"])

    try:
        table = client.get_table(raw_table)
        row_count = table.num_rows

        print(f"✅ Raw data found: {row_count} rows in {raw_table}")

        if row_count == 0:
            raise Exception(f"❌ Table {raw_table} exists but is empty")

        # Push to XCom
        context["ti"].xcom_push(key="raw_table", value=raw_table)
        context["ti"].xcom_push(key="row_count", value=row_count)

        return True

    except Exception as e:
        print(f"❌ Raw data not found: {e}")
        raise Exception(f"Cannot proceed with predictions: {e}")


def run_daily_prediction(**context):
    """
    1. Read from BigQuery raw table
    2. Call /predict endpoint
    3. Store predictions in BigQuery predictions table
    """
    today = datetime.utcnow().strftime("%Y%m%d")
    raw_table = context["ti"].xcom_pull(task_ids="check_raw_data", key="raw_table")

    print(f"🤖 Running predictions for {today}")
    print(f"📊 Source table: {raw_table}")

    # Ensure predictions dataset exists
    create_bq_dataset_if_not_exists(
        ENV_CONFIG["BQ_PROJECT"],
        ENV_CONFIG["BQ_PREDICT_DATASET"],
        ENV_CONFIG["BQ_LOCATION"],
    )

    # 1️⃣ Read raw data from BigQuery
    client = bigquery.Client(project=ENV_CONFIG["BQ_PROJECT"])

    query = f"""
    SELECT *
    FROM `{raw_table}`
    LIMIT 500
    """  # nosec B608

    print("📥 Reading data from BigQuery...")
    df = client.query(query).to_dataframe()

    if df.empty:
        raise Exception("❌ No data fetched from BigQuery")

    print(f"✅ Fetched {len(df)} records from BigQuery")
    print(f"📊 Columns: {df.columns.tolist()}")

    # 2️⃣ Call /predict endpoint
    api_url = f"{ENV_CONFIG['API_URL']}/predict"

    print(f"🌐 Calling prediction API: {api_url}")

    # Prepare payload
    payload = {
        "records": df.to_dict(orient="records"),
        "model_type": "rf",
        "metric": "r2",
    }

    # Add API key if configured
    headers = {}
    api_key = ENV_CONFIG.get("API_KEY_SECRET")
    if api_key and api_key != "dev-key-unsafe":
        headers["X-API-Key"] = api_key

    try:
        response = requests.post(
            api_url,
            json=payload,
            headers=headers,
            timeout=300,  # 5 minutes timeout
        )

        if response.status_code != 200:
            raise Exception(
                f"❌ Prediction API failed: {response.status_code} - {response.text}"
            )

        result = response.json()
        predictions = result.get("predictions", [])

        if not predictions:
            raise Exception("❌ No predictions returned from API")

        print(f"✅ Received {len(predictions)} predictions from API")

    except requests.exceptions.Timeout:
        raise Exception("❌ Prediction API timeout (>5 minutes)")
    except requests.exceptions.RequestException as e:
        raise Exception(f"❌ Prediction API request failed: {e}")

    # 3️⃣ Prepare predictions DataFrame
    df["prediction"] = predictions
    df["model_type"] = "rf"
    df["model_version"] = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    df["prediction_ts"] = datetime.utcnow()

    # Select relevant columns for predictions table
    pred_columns = [
        "comptage_horaire",  # Actual value (if available)
        "prediction",
        "model_type",
        "model_version",
        "prediction_ts",
        "identifiant_du_compteur",
        "date_et_heure_de_comptage",
    ]

    # Keep only columns that exist
    available_pred_columns = [col for col in pred_columns if col in df.columns]
    df_pred = df[available_pred_columns].copy()

    print(
        f"📊 Predictions dataset: {len(df_pred)} records, {len(available_pred_columns)} columns"
    )
    print(f"📊 Sample predictions:\n{df_pred.head(2)}")

    # 4️⃣ Store predictions in BigQuery
    pred_table = f"{ENV_CONFIG['BQ_PREDICT_DATASET']}.daily_{today}"
    full_pred_table = f"{ENV_CONFIG['BQ_PROJECT']}.{pred_table}"

    print(f"📤 Writing predictions to BigQuery: {full_pred_table}")

    df_pred.to_gbq(
        destination_table=pred_table,
        project_id=ENV_CONFIG["BQ_PROJECT"],
        if_exists="replace",
        location=ENV_CONFIG["BQ_LOCATION"],
    )

    print(f"✅ Successfully stored {len(df_pred)} predictions in {full_pred_table}")

    # Calculate basic statistics
    if "comptage_horaire" in df_pred.columns and "prediction" in df_pred.columns:
        df_with_actual = df_pred.dropna(subset=["comptage_horaire", "prediction"])

        if len(df_with_actual) > 0:
            from sklearn.metrics import (
                mean_squared_error,
                r2_score,
                mean_absolute_error,
            )
            import numpy as np

            y_true = df_with_actual["comptage_horaire"]
            y_pred = df_with_actual["prediction"]

            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            print("\n📈 Prediction Metrics:")
            print(f"   - RMSE: {rmse:.2f}")
            print(f"   - MAE: {mae:.2f}")
            print(f"   - R²: {r2:.4f}")

            # Push metrics to XCom for monitoring
            context["ti"].xcom_push(key="rmse", value=float(rmse))
            context["ti"].xcom_push(key="mae", value=float(mae))
            context["ti"].xcom_push(key="r2", value=float(r2))

    # Push metadata to XCom
    context["ti"].xcom_push(key="predictions_count", value=len(df_pred))
    context["ti"].xcom_push(key="pred_table", value=full_pred_table)


def validate_predictions(**context):
    """
    Validate that predictions were successfully stored in BigQuery
    Performs basic data quality checks
    """
    pred_table = context["ti"].xcom_pull(
        task_ids="predict_daily_data", key="pred_table"
    )
    predictions_count = context["ti"].xcom_pull(
        task_ids="predict_daily_data", key="predictions_count"
    )

    print(f"🔍 Validating predictions for {pred_table}")
    print(f"📊 Expected predictions: {predictions_count}")

    # Query BigQuery to verify
    client = bigquery.Client(project=ENV_CONFIG["BQ_PROJECT"])

    query = f"""
    SELECT
        COUNT(*) as total_predictions,
        AVG(prediction) as avg_prediction,
        MIN(prediction) as min_prediction,
        MAX(prediction) as max_prediction,
        MIN(prediction_ts) as first_prediction,
        MAX(prediction_ts) as last_prediction
    FROM `{pred_table}`
    """  # nosec B608

    df = client.query(query).to_dataframe()

    actual_count = df["total_predictions"].iloc[0]
    avg_pred = df["avg_prediction"].iloc[0]
    min_pred = df["min_prediction"].iloc[0]
    max_pred = df["max_prediction"].iloc[0]

    print("✅ Validation results:")
    print(f"   - Actual predictions in BQ: {actual_count}")
    print(f"   - Average prediction: {avg_pred:.2f}")
    print(f"   - Min/Max prediction: {min_pred:.2f} / {max_pred:.2f}")

    # Basic validation
    if actual_count != predictions_count:
        print(
            f"⚠️ WARNING: Prediction count mismatch (expected {predictions_count}, got {actual_count})"
        )

    if actual_count == 0:
        raise Exception("❌ Validation failed: No predictions in BigQuery table")

    # Sanity check on prediction values
    if min_pred < 0:
        print(f"⚠️ WARNING: Negative predictions detected (min={min_pred})")

    print(f"✅ Validation passed for {pred_table}")


# === DAG DEFINITION ===
with DAG(
    dag_id="daily_prediction",
    default_args=default_args,
    description="Generate daily bike traffic predictions via /predict endpoint",
    schedule_interval="@daily",  # Run daily after ingestion
    catchup=False,
    max_active_runs=1,
    tags=["bike", "prediction", "bigquery", "mlops"],
) as dag:
    # Task 1: Check if raw data exists
    check_data = PythonOperator(
        task_id="check_raw_data",
        python_callable=check_raw_data_exists,
        provide_context=True,
    )

    # Task 2: Run predictions
    predict_task = PythonOperator(
        task_id="predict_daily_data",
        python_callable=run_daily_prediction,
        provide_context=True,
    )

    # Task 3: Validate predictions
    validate_task = PythonOperator(
        task_id="validate_predictions",
        python_callable=validate_predictions,
        provide_context=True,
    )

    # Task dependencies
    check_data >> predict_task >> validate_task
