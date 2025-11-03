"""
DAG 3: Monitoring + Fine-tuning
Drift detection → Model validation → Conditional fine-tuning
Runs weekly to assess model performance and retrain if needed
"""

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime, timedelta
import requests
import pandas as pd
from google.cloud import bigquery
import numpy as np

from utils.env_config import get_env_config
from utils.bike_helpers import (
    create_monitoring_table_if_needed,
    get_reference_data_path,
)


# Configuration
ENV_CONFIG = get_env_config()

default_args = {
    "owner": "mlops-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "start_date": datetime(2024, 10, 1),
}


def run_drift_monitoring(**context):
    """
    Compare reference vs current data using Evidently
    Calls backend endpoint /monitor for drift detection
    """
    today = datetime.utcnow().strftime("%Y%m%d")
    print(f"🔍 Running drift monitoring for {today}")

    client = bigquery.Client(project=ENV_CONFIG["BQ_PROJECT"])

    # Load current data from BigQuery (last 7 days)
    query = f"""
    SELECT *
    FROM `{ENV_CONFIG['BQ_PROJECT']}.{ENV_CONFIG['BQ_RAW_DATASET']}.comptage_velo`
    WHERE date_et_heure_de_comptage >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
    LIMIT 1000
    """  # nosec B608

    print("📥 Loading current data from BigQuery (last 7 days)...")
    df_curr = client.query(query).to_dataframe()

    if df_curr.empty:
        raise Exception("❌ No current data found in BigQuery")

    print(f"✅ Loaded {len(df_curr)} current records")

    # Clean data for JSON serialization (convert Timestamps to strings)
    for col in df_curr.columns:
        if pd.api.types.is_datetime64_any_dtype(df_curr[col]):
            df_curr[col] = df_curr[col].astype(str)

    # Call /monitor endpoint with reference data from GCS
    api_url = f"{ENV_CONFIG['API_URL']}/monitor"
    reference_path = get_reference_data_path()

    print(f"🌐 Calling drift detection API: {api_url}")
    print(f"📂 Reference data path: {reference_path}")

    payload = {
        "reference_path": reference_path,
        "current_data": df_curr.to_dict(orient="records"),
        "output_html": f"drift_report_{today}.html",
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
            print(f"⚠️ Drift detection API returned non-200: {response.status_code}")
            print(f"Response: {response.text}")
            # Don't fail, continue with drift=False
            drift_detected = False
            drift_summary = {"drift_detected": False, "error": response.text}
        else:
            result = response.json()
            drift_summary = result.get("drift_summary", {})
            drift_detected = drift_summary.get("drift_detected", False)

            print(
                f"{'🚨 Drift detected!' if drift_detected else '✅ No drift detected'}"
            )
            print(f"📊 Drift summary: {drift_summary}")

    except requests.exceptions.Timeout:
        print("⚠️ Drift detection API timeout, assuming no drift")
        drift_detected = False
        drift_summary = {"drift_detected": False, "error": "timeout"}
    except Exception as e:
        print(f"⚠️ Drift detection failed: {e}, assuming no drift")
        drift_detected = False
        drift_summary = {"drift_detected": False, "error": str(e)}

    # Push to XCom
    context["ti"].xcom_push(key="drift_detected", value=drift_detected)
    context["ti"].xcom_push(key="drift_summary", value=str(drift_summary))
    context["ti"].xcom_push(key="timestamp", value=today)


def validate_model(**context):
    """
    Compare predictions vs true labels from BigQuery
    Calculate RMSE and R² for model performance
    """
    today = datetime.utcnow().strftime("%Y%m%d")
    print(f"📊 Validating model performance for {today}")

    client = bigquery.Client(project=ENV_CONFIG["BQ_PROJECT"])

    # Join predictions with actual values (last 7 days)
    query = f"""
    WITH recent_predictions AS (
        SELECT
            p.prediction,
            p.identifiant_du_compteur,
            PARSE_TIMESTAMP('%Y-%m-%d %H:%M:%S%Ez', p.date_et_heure_de_comptage) as date_et_heure_de_comptage,
            p.prediction_ts
        FROM `{ENV_CONFIG['BQ_PROJECT']}.{ENV_CONFIG['BQ_PREDICT_DATASET']}.daily_*` p
        WHERE _TABLE_SUFFIX >= FORMAT_DATE('%Y%m%d', DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY))
    ),
    recent_actuals AS (
        SELECT
            r.comptage_horaire as true_value,
            r.identifiant_du_compteur,
            r.date_et_heure_de_comptage
        FROM `{ENV_CONFIG['BQ_PROJECT']}.{ENV_CONFIG['BQ_RAW_DATASET']}.comptage_velo` r
        WHERE r.date_et_heure_de_comptage >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
    )
    SELECT
        p.prediction,
        a.true_value
    FROM recent_predictions p
    JOIN recent_actuals a
        ON p.identifiant_du_compteur = a.identifiant_du_compteur
        AND p.date_et_heure_de_comptage = a.date_et_heure_de_comptage
    WHERE a.true_value IS NOT NULL
    LIMIT 1000
    """  # nosec B608

    print("📥 Loading predictions and actuals from BigQuery...")
    df = client.query(query).to_dataframe()

    if df.empty or len(df) < 10:
        print(f"⚠️ Insufficient data for validation (only {len(df)} samples)")
        # Push default values
        context["ti"].xcom_push(key="rmse", value=999.0)
        context["ti"].xcom_push(key="r2", value=0.0)
        context["ti"].xcom_push(key="validation_samples", value=len(df))
        return

    print(f"✅ Loaded {len(df)} validation samples")

    # Calculate metrics
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    y_true = df["true_value"]
    y_pred = df["prediction"]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("\n📈 Model Validation Metrics:")
    print(f"   - RMSE: {rmse:.2f}")
    print(f"   - MAE: {mae:.2f}")
    print(f"   - R²: {r2:.4f}")
    print(f"   - Samples: {len(df)}")

    # Push metrics to XCom
    context["ti"].xcom_push(key="rmse", value=float(rmse))
    context["ti"].xcom_push(key="mae", value=float(mae))
    context["ti"].xcom_push(key="r2", value=float(r2))
    context["ti"].xcom_push(key="validation_samples", value=len(df))


def decide_if_fine_tune(**context):
    """
    Decide whether to trigger fine-tuning based on:
    - Drift detected
    - R² below threshold (0.65)
    - RMSE above threshold (60.0)
    - Force flag (for testing)
    """
    drift = context["ti"].xcom_pull(task_ids="monitor_drift", key="drift_detected")
    r2 = context["ti"].xcom_pull(task_ids="validate_model", key="r2")
    rmse = context["ti"].xcom_pull(task_ids="validate_model", key="rmse")

    # Check for force_fine_tune flag in DAG run config
    dag_run_conf = context.get("dag_run").conf or {}
    force_fine_tune = dag_run_conf.get("force_fine_tune", False)

    R2_THRESHOLD = 0.65
    RMSE_THRESHOLD = 60.0

    print("\n🎯 Decision Logic:")
    print(f"   - Drift detected: {drift}")
    print(f"   - R²: {r2:.4f} (threshold: {R2_THRESHOLD})")
    print(f"   - RMSE: {rmse:.2f} (threshold: {RMSE_THRESHOLD})")
    print(f"   - Force fine-tune: {force_fine_tune}")

    # PRIORITY: Force flag overrides all logic (for testing)
    if force_fine_tune:
        print("\n🧪 FORCE FINE-TUNE ENABLED (test mode)")
        print("   → Bypassing normal decision logic")
        return "fine_tune_model"

    # Decision: Fine-tune if drift AND poor metrics
    if drift and (r2 < R2_THRESHOLD or rmse > RMSE_THRESHOLD):
        print("\n🚨 Fine-tuning needed:")
        print(f"   → Drift detected: {drift}")
        print(f"   → R² below threshold: {r2:.4f} < {R2_THRESHOLD}")
        print(f"   → RMSE above threshold: {rmse:.2f} > {RMSE_THRESHOLD}")
        return "fine_tune_model"
    else:
        print("\n✅ Model performance OK, no fine-tuning needed:")
        print(f"   → Drift: {drift}")
        print(
            f"   → R²: {r2:.4f} >= {R2_THRESHOLD}"
            if r2 >= R2_THRESHOLD
            else f"   → R²: {r2:.4f}"
        )
        print(
            f"   → RMSE: {rmse:.2f} <= {RMSE_THRESHOLD}"
            if rmse <= RMSE_THRESHOLD
            else f"   → RMSE: {rmse:.2f}"
        )
        return "end_monitoring"


def fine_tune_model(**context):
    """
    Call /train endpoint with fine_tuning=True
    Uses latest data from BigQuery for incremental learning
    """
    today = datetime.utcnow().strftime("%Y%m%d")
    print(f"🧠 Starting fine-tuning for {today}")

    client = bigquery.Client(project=ENV_CONFIG["BQ_PROJECT"])

    # Get fresh data from BigQuery (last 30 days for training)
    query = f"""
    SELECT *
    FROM `{ENV_CONFIG['BQ_PROJECT']}.{ENV_CONFIG['BQ_RAW_DATASET']}.comptage_velo`
    WHERE date_et_heure_de_comptage >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
    LIMIT 2000
    """  # nosec B608

    print("📥 Fetching fresh training data from BigQuery...")
    df_fresh = client.query(query).to_dataframe()

    if df_fresh.empty:
        raise Exception("❌ No fresh data available for fine-tuning")

    print(f"✅ Loaded {len(df_fresh)} samples for fine-tuning")

    # Clean data for JSON serialization (convert Timestamps to strings)
    for col in df_fresh.columns:
        if pd.api.types.is_datetime64_any_dtype(df_fresh[col]):
            df_fresh[col] = df_fresh[col].astype(str)
        elif df_fresh[col].dtype in ["float64", "int64"]:
            df_fresh[col] = df_fresh[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    print("✅ Data cleaned for JSON serialization")

    # Call /train endpoint with fine-tuning mode
    api_url = f"{ENV_CONFIG['API_URL']}/train"

    # Get test_mode from DAG conf (default True for DEV)
    test_mode = context["dag_run"].conf.get("test_mode", False)

    payload = {
        "model_type": "rf",
        "data_source": "baseline",  # Use baseline for training
        "current_data": df_fresh.to_dict(orient="records"),  # For double evaluation
        "env": ENV_CONFIG["ENV"],
        "test_mode": test_mode,
    }

    # Add API key if configured
    headers = {}
    api_key = ENV_CONFIG.get("API_KEY_SECRET")
    if api_key and api_key != "dev-key-unsafe":
        headers["X-API-Key"] = api_key

    print(f"🌐 Calling training API: {api_url}")
    print(f"📊 Training parameters: fine_tuning=True, samples={len(df_fresh)}")

    try:
        response = requests.post(
            api_url,
            json=payload,
            headers=headers,
            timeout=600,  # 10 minutes timeout
        )

        if response.status_code != 200:
            error_msg = f"Training API failed: {response.status_code} - {response.text}"
            print(f"❌ {error_msg}")
            # Log failure to BigQuery
            context["ti"].xcom_push(key="fine_tune_success", value=False)
            context["ti"].xcom_push(key="error_message", value=error_msg)
            context["ti"].xcom_push(key="model_improvement", value=0.0)
            return

        result = response.json()
        print("✅ Fine-tuning completed successfully")
        print(f"📊 Training result: {result}")

        # Extract metrics
        current_r2 = context["ti"].xcom_pull(task_ids="validate_model", key="r2")
        new_r2 = result.get("r2", current_r2)
        r2_improvement = new_r2 - current_r2 if current_r2 else 0.0

        print("\n📈 Fine-tuning Results:")
        print(f"   - Previous R²: {current_r2:.4f}")
        print(f"   - New R²: {new_r2:.4f}")
        print(f"   - Improvement: {r2_improvement:+.4f}")

        # Push results to XCom
        context["ti"].xcom_push(key="fine_tune_success", value=True)
        context["ti"].xcom_push(key="model_improvement", value=float(r2_improvement))
        context["ti"].xcom_push(key="new_r2", value=float(new_r2))
        context["ti"].xcom_push(
            key="model_path", value=result.get("model_path", "unknown")
        )
        context["ti"].xcom_push(key="error_message", value="")

    except requests.exceptions.Timeout:
        error_msg = "Training API timeout (>10 minutes)"
        print(f"❌ {error_msg}")
        context["ti"].xcom_push(key="fine_tune_success", value=False)
        context["ti"].xcom_push(key="error_message", value=error_msg)
        context["ti"].xcom_push(key="model_improvement", value=0.0)
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        print(f"❌ {error_msg}")
        context["ti"].xcom_push(key="fine_tune_success", value=False)
        context["ti"].xcom_push(key="error_message", value=error_msg)
        context["ti"].xcom_push(key="model_improvement", value=0.0)


def end_monitoring(**context):
    """
    Log monitoring results to BigQuery audit table
    Called whether training happened or not
    """
    print("\n📊 Finalizing monitoring...")

    # Ensure monitoring table exists
    create_monitoring_table_if_needed(
        ENV_CONFIG["BQ_PROJECT"], "monitoring_audit", "logs", ENV_CONFIG["BQ_LOCATION"]
    )

    # Collect metrics from previous tasks
    timestamp = datetime.utcnow()
    drift_detected = context["ti"].xcom_pull(
        task_ids="monitor_drift", key="drift_detected"
    )
    rmse = context["ti"].xcom_pull(task_ids="validate_model", key="rmse")
    r2 = context["ti"].xcom_pull(task_ids="validate_model", key="r2")

    # Check if fine-tuning was executed
    fine_tune_success = context["ti"].xcom_pull(
        task_ids="fine_tune_model", key="fine_tune_success"
    )
    model_improvement = context["ti"].xcom_pull(
        task_ids="fine_tune_model", key="model_improvement"
    )
    error_message = context["ti"].xcom_pull(
        task_ids="fine_tune_model", key="error_message"
    )

    # If fine_tune task didn't run, set defaults
    if fine_tune_success is None:
        fine_tune_triggered = False
        fine_tune_success = False
        model_improvement = 0.0
        error_message = ""
    else:
        fine_tune_triggered = True

    # Create audit record
    audit_df = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "drift_detected": drift_detected
                if drift_detected is not None
                else False,
                "rmse": float(rmse) if rmse else 999.0,
                "r2": float(r2) if r2 else 0.0,
                "fine_tune_triggered": fine_tune_triggered,
                "fine_tune_success": fine_tune_success,
                "model_improvement": float(model_improvement)
                if model_improvement
                else 0.0,
                "env": ENV_CONFIG["ENV"],
                "error_message": error_message if error_message else "",
            }
        ]
    )

    # Write to BigQuery
    table_id = f"{ENV_CONFIG['BQ_PROJECT']}.monitoring_audit.logs"

    print(f"📤 Writing audit log to BigQuery: {table_id}")
    print(f"📊 Audit record:\n{audit_df.to_dict(orient='records')}")

    client = bigquery.Client(project=ENV_CONFIG["BQ_PROJECT"])
    client.load_table_from_dataframe(audit_df, table_id).result()

    print(f"✅ Audit log written to {table_id}")

    # Summary
    print("\n📋 Monitoring Summary:")
    print(f"   - Drift detected: {'🚨 YES' if drift_detected else '✅ NO'}")
    print(f"   - RMSE: {rmse:.2f}" if rmse else "   - RMSE: N/A")
    print(f"   - R²: {r2:.4f}" if r2 else "   - R²: N/A")
    print(
        f"   - Fine-tuning: {'✅ SUCCESS' if fine_tune_success else '⛔ NOT TRIGGERED' if not fine_tune_triggered else '❌ FAILED'}"
    )
    if model_improvement and model_improvement != 0:
        print(f"   - Model improvement: {model_improvement:+.4f}")
    print(f"   - Environment: {ENV_CONFIG['ENV']}")


# === DAG DEFINITION ===
with DAG(
    dag_id="monitor_and_fine_tune",
    default_args=default_args,
    description="Weekly monitoring with drift detection and conditional fine-tuning",
    schedule_interval="@weekly",  # Run weekly on Sunday
    catchup=False,
    max_active_runs=1,
    tags=["bike", "monitoring", "drift", "training", "mlops"],
) as dag:
    # Task 1: Drift monitoring
    monitor = PythonOperator(
        task_id="monitor_drift",
        python_callable=run_drift_monitoring,
        provide_context=True,
    )

    # Task 2: Model validation
    validate = PythonOperator(
        task_id="validate_model", python_callable=validate_model, provide_context=True
    )

    # Task 3: Decision - Branch operator
    decide = BranchPythonOperator(
        task_id="decide_fine_tune",
        python_callable=decide_if_fine_tune,
        provide_context=True,
    )

    # Task 4a: Fine-tuning (conditional)
    fine_tune = PythonOperator(
        task_id="fine_tune_model",
        python_callable=fine_tune_model,
        provide_context=True,
        execution_timeout=timedelta(minutes=15),
    )

    # Task 4b: End without training (alternative branch)
    end = PythonOperator(
        task_id="end_monitoring",
        python_callable=end_monitoring,
        provide_context=True,
        trigger_rule="none_failed_min_one_success",  # Run regardless of branch taken
    )

    # Pipeline flow
    monitor >> validate >> decide
    decide >> [fine_tune, end]
    fine_tune >> end
