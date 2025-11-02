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
import os

from utils.env_config import get_env_config
from utils.bike_helpers import (
    create_monitoring_table_if_needed,
    get_reference_data_path,
)


# Configuration
ENV_CONFIG = get_env_config()
IS_DEV = os.getenv("ENV", "DEV") == "DEV"

default_args = {
    "owner": "mlops-team",
    "retries": 1,
    "retry_delay": timedelta(seconds=30) if IS_DEV else timedelta(minutes=10),
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

    # Convert timestamps to strings for JSON serialization
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
    # Note: Predictions are still in daily_* tables (from DAG 2)
    # Raw data is in partitioned comptage_velo table
    query = f"""
    WITH recent_predictions AS (
        SELECT
            p.prediction,
            p.identifiant_du_compteur,
            p.date_et_heure_de_comptage,
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
    """
    drift = context["ti"].xcom_pull(task_ids="monitor_drift", key="drift_detected")
    r2 = context["ti"].xcom_pull(task_ids="validate_model", key="r2")
    rmse = context["ti"].xcom_pull(task_ids="validate_model", key="rmse")

    R2_THRESHOLD = 0.65
    RMSE_THRESHOLD = 60.0

    print("\n🎯 Decision Logic:")
    print(f"   - Drift detected: {drift}")
    print(f"   - R²: {r2:.4f} (threshold: {R2_THRESHOLD})")
    print(f"   - RMSE: {rmse:.2f} (threshold: {RMSE_THRESHOLD})")

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
    Call /train endpoint with double evaluation strategy.

    NEW: Uses double test set evaluation:
    - Passes current BigQuery data to /train endpoint
    - Evaluates on test_baseline (fixed reference) + test_current (20% of current data)
    - Makes deployment decision based on both metrics
    """
    today = datetime.utcnow().strftime("%Y%m%d")
    print(f"🧠 Starting fine-tuning with double evaluation for {today}")

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

    # Check if enough data for double evaluation (min 200 samples)
    if len(df_fresh) < 200:
        print(
            f"⚠️ Insufficient data for double evaluation: {len(df_fresh)} < 200 samples"
        )
        print("Skipping fine-tuning, continuing monitoring...")
        context["ti"].xcom_push(key="fine_tune_success", value=False)
        context["ti"].xcom_push(
            key="error_message", value="Insufficient data for double evaluation"
        )
        context["ti"].xcom_push(key="model_improvement", value=0.0)
        return

    # Convert timestamps to strings for JSON serialization
    for col in df_fresh.columns:
        if pd.api.types.is_datetime64_any_dtype(df_fresh[col]):
            df_fresh[col] = df_fresh[col].astype(str)

    # Call /train endpoint with double evaluation
    api_url = f"{ENV_CONFIG['API_URL']}/train"

    # NEW: Use baseline as data_source + pass current_data for double eval
    payload = {
        "model_type": "rf",
        "data_source": "baseline",  # Train on train_baseline.csv
        "env": ENV_CONFIG["ENV"],
        "current_data": df_fresh.to_dict(orient="records"),  # For double evaluation
        "test_mode": False,
    }

    # Add API key if configured
    headers = {}
    api_key = ENV_CONFIG.get("API_KEY_SECRET")
    if api_key and api_key != "dev-key-unsafe":
        headers["X-API-Key"] = api_key

    print(f"🌐 Calling training API: {api_url}")
    print("📊 Training parameters:")
    print("   - Model: rf")
    print("   - Data source: baseline (train_baseline.csv)")
    print(f"   - Current data: {len(df_fresh)} samples (for double evaluation)")
    print(f"   - Environment: {ENV_CONFIG['ENV']}")

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
            # Log failure
            context["ti"].xcom_push(key="fine_tune_success", value=False)
            context["ti"].xcom_push(key="error_message", value=error_msg)
            context["ti"].xcom_push(key="model_improvement", value=0.0)
            context["ti"].xcom_push(key="decision", value="error")
            return

        result = response.json()
        print("✅ Fine-tuning completed successfully")
        print(f"📊 Training result: {result}")

        # Extract double evaluation metrics
        metrics_baseline = result.get("metrics_baseline", {})
        metrics_current = result.get("metrics_current", {})
        baseline_regression = result.get("baseline_regression", False)
        double_eval_enabled = result.get("double_evaluation_enabled", False)

        if not double_eval_enabled:
            print("⚠️ Double evaluation was not enabled - possibly insufficient data")
            context["ti"].xcom_push(key="fine_tune_success", value=False)
            context["ti"].xcom_push(key="decision", value="no_double_eval")
            context["ti"].xcom_push(
                key="error_message", value="Double evaluation not enabled"
            )
            return

        # Get previous R² from validation task for comparison
        previous_r2 = context["ti"].xcom_pull(task_ids="validate_model", key="r2")
        new_r2_baseline = metrics_baseline.get("r2", 0.0)
        new_r2_current = metrics_current.get("r2", 0.0)

        print("\n" + "=" * 60)
        print("📊 DOUBLE EVALUATION RESULTS")
        print("=" * 60)
        print("📍 Test Baseline (fixed reference):")
        print(f"   - RMSE: {metrics_baseline.get('rmse', 'N/A')}")
        print(f"   - MAE:  {metrics_baseline.get('mae', 'N/A')}")
        print(f"   - R²:   {new_r2_baseline:.4f}")
        print("\n🆕 Test Current (new distribution):")
        print(f"   - RMSE: {metrics_current.get('rmse', 'N/A')}")
        print(f"   - MAE:  {metrics_current.get('mae', 'N/A')}")
        print(f"   - R²:   {new_r2_current:.4f}")
        print("\n📈 Comparison:")
        print(f"   - Previous R² (production):  {previous_r2:.4f}")
        print(f"   - New R² (test_current):     {new_r2_current:.4f}")
        print(f"   - Improvement: {(new_r2_current - previous_r2):+.4f}")
        print("=" * 60)

        # DECISION LOGIC
        BASELINE_R2_THRESHOLD = 0.60

        if baseline_regression:
            # Model regressed on baseline - reject deployment
            decision = "reject_regression"
            fine_tune_success = False
            print("\n🚨 DECISION: REJECT")
            print(
                f"   Reason: Model regressed on baseline (R² < {BASELINE_R2_THRESHOLD})"
            )
            print(f"   Baseline R²: {new_r2_baseline:.4f}")

        elif new_r2_current > previous_r2:
            # Model improved on current distribution - deploy
            decision = "deploy"
            fine_tune_success = True
            improvement = new_r2_current - previous_r2
            print("\n✅ DECISION: DEPLOY")
            print("   Reason: Improved on current distribution")
            print(f"   Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")

        else:
            # No improvement on current distribution - skip
            decision = "skip_no_improvement"
            fine_tune_success = False
            print("\n⚠️ DECISION: SKIP")
            print("   Reason: No improvement on current distribution")
            print(f"   New R²: {new_r2_current:.4f} <= Previous R²: {previous_r2:.4f}")

        # Push results to XCom for audit logging
        context["ti"].xcom_push(key="fine_tune_success", value=fine_tune_success)
        context["ti"].xcom_push(key="decision", value=decision)
        context["ti"].xcom_push(
            key="model_improvement",
            value=float(new_r2_current - previous_r2) if previous_r2 else 0.0,
        )
        context["ti"].xcom_push(key="metrics_baseline", value=str(metrics_baseline))
        context["ti"].xcom_push(key="metrics_current", value=str(metrics_current))
        context["ti"].xcom_push(key="baseline_regression", value=baseline_regression)
        context["ti"].xcom_push(key="new_r2_baseline", value=float(new_r2_baseline))
        context["ti"].xcom_push(key="new_r2_current", value=float(new_r2_current))
        context["ti"].xcom_push(key="error_message", value="")

    except requests.exceptions.Timeout:
        error_msg = "Training API timeout (>10 minutes)"
        print(f"❌ {error_msg}")
        context["ti"].xcom_push(key="fine_tune_success", value=False)
        context["ti"].xcom_push(key="error_message", value=error_msg)
        context["ti"].xcom_push(key="decision", value="timeout")
        context["ti"].xcom_push(key="model_improvement", value=0.0)
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback

        print(traceback.format_exc())
        context["ti"].xcom_push(key="fine_tune_success", value=False)
        context["ti"].xcom_push(key="error_message", value=error_msg)
        context["ti"].xcom_push(key="decision", value="error")
        context["ti"].xcom_push(key="model_improvement", value=0.0)


def end_monitoring(**context):
    """
    Log monitoring results to BigQuery audit table.
    Called whether training happened or not.

    NEW: Logs double evaluation metrics (metrics_baseline, metrics_current, decision)
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

    # NEW: Pull double evaluation metrics
    decision = context["ti"].xcom_pull(task_ids="fine_tune_model", key="decision")
    metrics_baseline = context["ti"].xcom_pull(
        task_ids="fine_tune_model", key="metrics_baseline"
    )
    metrics_current = context["ti"].xcom_pull(
        task_ids="fine_tune_model", key="metrics_current"
    )
    baseline_regression = context["ti"].xcom_pull(
        task_ids="fine_tune_model", key="baseline_regression"
    )
    new_r2_baseline = context["ti"].xcom_pull(
        task_ids="fine_tune_model", key="new_r2_baseline"
    )
    new_r2_current = context["ti"].xcom_pull(
        task_ids="fine_tune_model", key="new_r2_current"
    )

    # If fine_tune task didn't run, set defaults
    if fine_tune_success is None:
        fine_tune_triggered = False
        fine_tune_success = False
        model_improvement = 0.0
        error_message = ""
        decision = "not_triggered"
        metrics_baseline = ""
        metrics_current = ""
        baseline_regression = False
        new_r2_baseline = 0.0
        new_r2_current = 0.0
    else:
        fine_tune_triggered = True

    # Create audit record with double evaluation metrics
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
                # NEW: Double evaluation fields
                "decision": decision if decision else "not_triggered",
                "baseline_regression": baseline_regression
                if baseline_regression is not None
                else False,
                "r2_baseline": float(new_r2_baseline) if new_r2_baseline else 0.0,
                "r2_current": float(new_r2_current) if new_r2_current else 0.0,
                "metrics_baseline": str(metrics_baseline) if metrics_baseline else "",
                "metrics_current": str(metrics_current) if metrics_current else "",
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

    # Summary with double evaluation info
    print("\n" + "=" * 60)
    print("📋 MONITORING SUMMARY")
    print("=" * 60)
    print("🔍 Drift Detection:")
    print(f"   - Drift detected: {'🚨 YES' if drift_detected else '✅ NO'}")
    print("\n📊 Production Model Performance:")
    print(f"   - RMSE: {rmse:.2f}" if rmse else "   - RMSE: N/A")
    print(f"   - R²: {r2:.4f}" if r2 else "   - R²: N/A")
    print("\n🎯 Fine-tuning Status:")
    if fine_tune_triggered:
        status = "✅ SUCCESS" if fine_tune_success else "❌ FAILED"
        print(f"   - Status: {status}")
        print(f"   - Decision: {decision if decision else 'N/A'}")
        if decision and decision != "not_triggered":
            print("\n📈 Double Evaluation Results:")
            print(
                f"   - Baseline R²: {new_r2_baseline:.4f}"
                if new_r2_baseline
                else "   - Baseline R²: N/A"
            )
            print(
                f"   - Current R²: {new_r2_current:.4f}"
                if new_r2_current
                else "   - Current R²: N/A"
            )
            print(
                f"   - Regression on baseline: {'🚨 YES' if baseline_regression else '✅ NO'}"
            )
            if model_improvement and model_improvement != 0:
                print(
                    f"   - Improvement: {model_improvement:+.4f} ({model_improvement*100:+.2f}%)"
                )
    else:
        print("   - Status: ⛔ NOT TRIGGERED")
    print(f"\n🌍 Environment: {ENV_CONFIG['ENV']}")
    print("=" * 60)


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
        task_id="fine_tune_model", python_callable=fine_tune_model, provide_context=True
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
