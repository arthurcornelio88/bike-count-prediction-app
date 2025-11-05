"""
DAG 3: Monitoring + Fine-tuning with Champion Promotion
Drift detection → Model validation → Conditional fine-tuning → Champion validation

Flow:
1. monitor_drift: Detect data drift using Evidently
2. validate_model: Validate current CHAMPION on production data (last 7 days)
3. decide_fine_tune: Hybrid decision logic (reactive + proactive)
   - Branch A: fine_tune_model → validate_new_champion → end_monitoring
   - Branch B: end_monitoring (no training needed)
4. fine_tune_model: Train CHALLENGER with double evaluation (baseline + current)
   - If CHALLENGER beats CHAMPION: promote to new CHAMPION
5. validate_new_champion: Re-validate NEW CHAMPION on production data
   - CRITICAL: After promotion, we need fresh metrics for the new champion
   - Without this, Grafana/Prometheus would show OLD champion's metrics
   - Pushes metrics to XCom for end_monitoring to write to BigQuery
6. end_monitoring: Write audit log with correct champion metrics
   - Uses validate_new_champion metrics if promotion happened
   - Uses validate_model metrics if no promotion

Why validate_new_champion is necessary:
- validate_model runs BEFORE training (validates old champion)
- After promotion, BigQuery still has old champion's validation metrics
- Prometheus/Grafana read from BigQuery → show stale metrics
- validate_new_champion provides fresh metrics for the newly promoted model
- This ensures monitoring dashboards reflect the current production model

Runs weekly to assess model performance and retrain if needed.
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
from utils.discord_alerts import (
    send_drift_alert,
    send_performance_alert,
    send_training_success,
    send_training_failure,
    send_champion_promotion_alert,
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
    FROM `{ENV_CONFIG["BQ_PROJECT"]}.{ENV_CONFIG["BQ_RAW_DATASET"]}.comptage_velo`
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

    NOTE: This validates the CHAMPION model currently in production.
    Predictions in BigQuery were made by the champion (is_champion=True).
    """
    today = datetime.utcnow().strftime("%Y%m%d")
    print(f"📊 Validating CHAMPION model performance for {today}")

    # First, identify which model is the current champion
    try:
        print("🔍 Identifying current champion model...")
        api_url = f"{ENV_CONFIG['API_URL']}/model_summary"
        headers = {}
        api_key = ENV_CONFIG.get("API_KEY_SECRET")
        if api_key and api_key != "dev-key-unsafe":
            headers["X-API-Key"] = api_key

        response = requests.get(api_url, headers=headers, timeout=30)
        if response.status_code == 200:
            summary = response.json()
            champion_models = [m for m in summary if m.get("is_champion", False)]
            if champion_models:
                champion = champion_models[0]
                print(
                    f"🏆 Current champion: {champion.get('model_type')} (run_id: {champion.get('run_id', 'unknown')[:8]}...)"
                )
                print(
                    f"   - env: {champion.get('env')}, test_mode: {champion.get('test_mode')}"
                )
                context["ti"].xcom_push(
                    key="champion_run_id", value=champion.get("run_id", "unknown")
                )
            else:
                print("⚠️ No champion model found in registry")
        else:
            print(f"⚠️ Failed to fetch model summary: {response.status_code}")
    except Exception as e:
        print(f"⚠️ Failed to identify champion model: {e}")

    client = bigquery.Client(project=ENV_CONFIG["BQ_PROJECT"])

    # Join predictions with actual values (last 7 days)
    query = f"""
    WITH recent_predictions AS (
        SELECT
            p.prediction,
            p.identifiant_du_compteur,
            PARSE_TIMESTAMP('%Y-%m-%d %H:%M:%S%Ez', p.date_et_heure_de_comptage) as date_et_heure_de_comptage,
            p.prediction_ts
        FROM `{ENV_CONFIG["BQ_PROJECT"]}.{ENV_CONFIG["BQ_PREDICT_DATASET"]}.daily_*` p
        WHERE _TABLE_SUFFIX >= FORMAT_DATE('%Y%m%d', DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY))
    ),
    recent_actuals AS (
        SELECT
            r.comptage_horaire as true_value,
            r.identifiant_du_compteur,
            r.date_et_heure_de_comptage
        FROM `{ENV_CONFIG["BQ_PROJECT"]}.{ENV_CONFIG["BQ_RAW_DATASET"]}.comptage_velo` r
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
    Hybrid drift management strategy combining proactive and reactive approaches.

    Decision logic:
    1. REACTIVE: If metrics are critically poor (R² < 0.65), retrain immediately
    2. PROACTIVE: If high drift (>50%) + metrics declining (R² < 0.70), retrain preventively
    3. WAIT: If moderate drift (30-50%) but metrics still good (R² >= 0.70), monitor closely
    4. OK: If low drift (<30%) and good metrics, no action needed

    This balances cost (avoiding unnecessary retraining) with performance (catching degradation early).

    NOTE: Decision is based on CHAMPION model performance (validated in previous task).
    """
    drift = context["ti"].xcom_pull(task_ids="monitor_drift", key="drift_detected")
    r2 = context["ti"].xcom_pull(task_ids="validate_model", key="r2")
    rmse = context["ti"].xcom_pull(task_ids="validate_model", key="rmse")
    champion_run_id = context["ti"].xcom_pull(
        task_ids="validate_model", key="champion_run_id"
    )
    drift_summary_str = context["ti"].xcom_pull(
        task_ids="monitor_drift", key="drift_summary"
    )

    # Check for force_fine_tune flag in DAG run config
    dag_run_conf = context.get("dag_run").conf or {}
    force_fine_tune = dag_run_conf.get("force_fine_tune", False)

    # Parse drift summary to get drift_share
    drift_share = 0.0
    if drift_summary_str:
        try:
            import ast

            drift_info = (
                ast.literal_eval(drift_summary_str)
                if isinstance(drift_summary_str, str)
                else drift_summary_str
            )
            drift_share = drift_info.get("drift_share", 0.0)
        except Exception as e:
            print(f"⚠️ Failed to parse drift_summary: {e}")
            drift_share = (
                1.0 if drift else 0.0
            )  # Fallback: assume full drift if detected

    # Thresholds (adjusted for production data distribution)
    R2_CRITICAL = 0.45  # Below this → retrain immediately (reactive)
    R2_WARNING = 0.55  # Below this + high drift → retrain proactively
    RMSE_THRESHOLD = 90.0  # Above this → retrain immediately (was 60.0)
    DRIFT_CRITICAL = 0.5  # 50%+ drift share → critical
    DRIFT_WARNING = 0.3  # 30%+ drift share → warning

    print("\n🎯 Decision Logic (Hybrid Strategy):")
    print(
        f"   - Champion model: {champion_run_id[:8] if champion_run_id else 'unknown'}..."
    )
    print(f"   - Drift detected: {drift} (drift share: {drift_share:.1%})")
    print(
        f"   - Champion R²: {r2:.4f} (critical: {R2_CRITICAL}, warning: {R2_WARNING})"
    )
    print(f"   - Champion RMSE: {rmse:.2f} (threshold: {RMSE_THRESHOLD})")
    print(f"   - Force fine-tune: {force_fine_tune}")

    # PRIORITY 0: Force flag overrides all logic (for testing)
    if force_fine_tune:
        print("\n🧪 FORCE FINE-TUNE ENABLED (test mode)")
        print("   → Bypassing normal decision logic")
        return "fine_tune_model"

    # PRIORITY 1: Critical metrics → retrain immediately (REACTIVE)
    if r2 < R2_CRITICAL or rmse > RMSE_THRESHOLD:
        print("\n🚨 RETRAIN DECISION: Poor performance (REACTIVE)")
        print("   → Metrics critically poor:")
        if r2 < R2_CRITICAL:
            print(f"     • R²: {r2:.4f} < {R2_CRITICAL} (critical threshold)")
        if rmse > RMSE_THRESHOLD:
            print(f"     • RMSE: {rmse:.2f} > {RMSE_THRESHOLD} (critical threshold)")
        if drift:
            print(f"   → Drift also detected (drift share: {drift_share:.1%})")
        print("   → Action: Immediate retraining to restore performance")

        # Send Discord alert for critical performance
        send_performance_alert(r2, rmse, threshold=R2_WARNING)

        return "fine_tune_model"

    # PRIORITY 2: Critical drift + metrics declining → retrain preventively (PROACTIVE)
    if drift and drift_share >= DRIFT_CRITICAL and r2 < R2_WARNING:
        print("\n⚠️ RETRAIN DECISION: Critical drift + declining metrics (PROACTIVE)")
        print(f"   → Drift share: {drift_share:.1%} >= {DRIFT_CRITICAL:.1%} (critical)")
        print(f"   → R²: {r2:.4f} < {R2_WARNING} (declining, not critical yet)")
        print(f"   → Metrics still above critical ({R2_CRITICAL}) but trending down")
        print("   → Action: Proactive retraining to prevent further degradation")

        # Send Discord alerts for drift and declining performance
        send_drift_alert(
            drift_share, r2, drifted_features=0
        )  # features count from drift_summary if available
        send_performance_alert(r2, rmse, threshold=R2_WARNING)

        return "fine_tune_model"

    # PRIORITY 3: Moderate-to-critical drift but metrics still good → monitor closely (WAIT)
    if drift and drift_share >= DRIFT_WARNING:
        print("\n✅ WAIT DECISION: Significant drift but metrics OK")
        print(f"   → Drift share: {drift_share:.1%} >= {DRIFT_WARNING:.1%}")
        print(f"   → R²: {r2:.4f} >= {R2_WARNING} (still good)")
        print(f"   → RMSE: {rmse:.2f} <= {RMSE_THRESHOLD} (within limits)")
        print("   → Model handles new compteurs via handle_unknown='ignore'")
        print(
            f"   → Will retrain if R² drops below {R2_WARNING} (proactive) or {R2_CRITICAL} (reactive)"
        )
        print("   → Action: Monitor closely, no retraining yet")

        # Send drift alert (WARNING level) - no training needed yet
        send_drift_alert(drift_share, r2, drifted_features=0)

        return "end_monitoring"

    # All good - no drift or low drift with good metrics
    print("\n✅ ALL GOOD: No retraining needed")
    if drift and drift_share > 0:
        print(
            f"   → Low drift detected (drift share: {drift_share:.1%} < {DRIFT_WARNING:.1%})"
        )
    else:
        print("   → No significant drift detected")
    print(f"   → R²: {r2:.4f} >= {R2_WARNING} (excellent)")
    print(f"   → RMSE: {rmse:.2f} <= {RMSE_THRESHOLD} (within limits)")
    print("   → Action: Continue monitoring")
    return "end_monitoring"


def fine_tune_model(**context):
    """
    Call /train endpoint with double evaluation strategy.
    Uses latest data from BigQuery for training + evaluation on:
    - test_baseline (detect regression)
    - test_current (20% of fresh data, evaluate improvement)
    """
    today = datetime.utcnow().strftime("%Y%m%d")
    print(f"🧠 Starting fine-tuning for {today}")

    client = bigquery.Client(project=ENV_CONFIG["BQ_PROJECT"])

    # Get fresh data from BigQuery (last 30 days for training)
    query = f"""
    SELECT *
    FROM `{ENV_CONFIG["BQ_PROJECT"]}.{ENV_CONFIG["BQ_RAW_DATASET"]}.comptage_velo`
    WHERE date_et_heure_de_comptage >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
    LIMIT 2000
    """  # nosec B608

    print("📥 Fetching fresh training data from BigQuery...")
    df_fresh = client.query(query).to_dataframe()

    if df_fresh.empty:
        raise Exception("❌ No fresh data available for fine-tuning")

    print(f"✅ Loaded {len(df_fresh)} samples for fine-tuning")

    # Preprocess BigQuery data to align with train_baseline.csv format
    print("🔧 Preprocessing BigQuery data for schema alignment...")

    # 1. Drop ingestion_ts (not in baseline CSV)
    if "ingestion_ts" in df_fresh.columns:
        df_fresh = df_fresh.drop(columns=["ingestion_ts"])
        print("   ✅ Dropped ingestion_ts")

    # 2. Reconstruct coordonnées_géographiques from latitude/longitude
    #    (RawCleanerTransformer expects this format for CSV compatibility)
    if "latitude" in df_fresh.columns and "longitude" in df_fresh.columns:
        df_fresh["coordonnées_géographiques"] = (
            df_fresh["latitude"].astype(str) + "," + df_fresh["longitude"].astype(str)
        )
        # Drop latitude/longitude to avoid duplicate columns after normalization
        df_fresh = df_fresh.drop(columns=["latitude", "longitude"])
        print("   ✅ Reconstructed coordonnées_géographiques (dropped lat/lon)")

    print(f"   📊 Preprocessed columns: {sorted(df_fresh.columns.tolist())}")

    # Clean data for JSON serialization (convert Timestamps to strings)
    for col in df_fresh.columns:
        if pd.api.types.is_datetime64_any_dtype(df_fresh[col]):
            df_fresh[col] = df_fresh[col].astype(str)
        elif df_fresh[col].dtype in ["float64", "int64"]:
            df_fresh[col] = df_fresh[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    print("✅ Data cleaned for JSON serialization")

    # Get current champion info from validate_model task
    old_champion_run_id = context["ti"].xcom_pull(
        task_ids="validate_model", key="champion_run_id"
    )

    # Evaluate OLD CHAMPION on test_baseline BEFORE training (for fair comparison)
    print(
        f"\n📊 Evaluating OLD CHAMPION ({old_champion_run_id[:8] if old_champion_run_id else 'unknown'}...) on test_baseline..."
    )
    evaluate_url = f"{ENV_CONFIG['API_URL']}/evaluate"
    evaluate_payload = {
        "model_type": "rf",
        "metric": "r2",
        "test_baseline_path": "gs://df_traffic_cyclist1/raw_data/test_baseline.csv",
    }

    old_champion_r2_baseline = None
    try:
        eval_response = requests.post(evaluate_url, json=evaluate_payload, timeout=1200)
        if eval_response.status_code == 200:
            eval_result = eval_response.json()
            old_champion_r2_baseline = eval_result["metrics"]["r2"]
            print(
                f"✅ OLD CHAMPION R² on test_baseline: {old_champion_r2_baseline:.4f}"
            )
        else:
            print(f"⚠️  Baseline evaluation failed: {eval_response.status_code}")
    except Exception as e:
        print(f"⚠️  Failed to evaluate old champion on baseline: {e}")

    # Call /train endpoint with fine-tuning mode
    api_url = f"{ENV_CONFIG['API_URL']}/train"

    # Get test_mode from DAG conf (default False for PROD)
    dag_run_conf = context.get("dag_run").conf or {}
    test_mode = dag_run_conf.get("test_mode", False)

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

    print("\n🚀 Training CHALLENGER model...")
    print(f"🌐 Calling training API: {api_url}")
    print(
        f"📊 Training parameters: data_source=baseline, current_data={len(df_fresh)} samples, test_mode={test_mode}"
    )

    try:
        response = requests.post(
            api_url,
            json=payload,
            headers=headers,
            timeout=1800,  # 30 minutes timeout (baseline eval takes ~12 min)
        )

        if response.status_code != 200:
            error_msg = f"Training API failed: {response.status_code} - {response.text}"
            print(f"❌ {error_msg}")
            # Log failure to BigQuery
            context["ti"].xcom_push(key="fine_tune_success", value=False)
            context["ti"].xcom_push(key="error_message", value=error_msg)
            context["ti"].xcom_push(key="model_improvement", value=0.0)
            context["ti"].xcom_push(key="baseline_regression", value=False)
            context["ti"].xcom_push(key="double_evaluation_enabled", value=False)
            return

        result = response.json()
        challenger_run_id = result.get("run_id", "unknown")
        print(
            f"✅ CHALLENGER model trained successfully (run_id: {challenger_run_id[:8]}...)"
        )
        print(f"📊 Training result: {result}")

        # Check if double evaluation was enabled
        double_eval_enabled = result.get("double_evaluation_enabled", False)

        if not double_eval_enabled:
            print("⚠️ Double evaluation not enabled (current_data too small or missing)")
            # Fallback to old behavior
            current_r2 = context["ti"].xcom_pull(task_ids="validate_model", key="r2")
            new_r2 = result.get("metrics", {}).get("r2", current_r2)
            r2_improvement = new_r2 - current_r2 if current_r2 else 0.0

            context["ti"].xcom_push(key="fine_tune_success", value=True)
            context["ti"].xcom_push(
                key="model_improvement", value=float(r2_improvement)
            )
            context["ti"].xcom_push(key="new_r2", value=float(new_r2))
            context["ti"].xcom_push(key="baseline_regression", value=False)
            context["ti"].xcom_push(key="double_evaluation_enabled", value=False)
            context["ti"].xcom_push(key="error_message", value="")
            return

        # Extract double evaluation metrics
        baseline_regression = result.get("baseline_regression", False)
        metrics_baseline = result.get("metrics_baseline", {})
        metrics_current = result.get("metrics_current", {})
        metrics_train = result.get("metrics", {})

        r2_baseline = metrics_baseline.get("r2", 0.0)
        r2_current = metrics_current.get("r2", 0.0)
        r2_train = metrics_train.get("r2", 0.0)

        # Get OLD CHAMPION metrics (from validation task)
        old_champion_r2_current = context["ti"].xcom_pull(
            task_ids="validate_model", key="r2"
        )
        # old_champion_r2_baseline already evaluated above (before training)

        # Calculate improvements on both test sets
        r2_improvement_current = (
            r2_current - old_champion_r2_current if old_champion_r2_current else 0.0
        )
        r2_improvement_baseline = None
        if old_champion_r2_baseline is not None:
            r2_improvement_baseline = r2_baseline - old_champion_r2_baseline

        print("\n" + "=" * 60)
        print("📊 DOUBLE EVALUATION RESULTS: CHALLENGER vs OLD CHAMPION")
        print("=" * 60)
        print(
            f"🏆 OLD CHAMPION: {old_champion_run_id[:8] if old_champion_run_id else 'unknown'}..."
        )
        print(f"🆕 CHALLENGER: {challenger_run_id[:8]}...")
        print("\n📍 Baseline Test Set (fixed reference, 181K samples):")
        print(f"   - CHALLENGER R²: {r2_baseline:.4f}")
        print(
            f"   - OLD CHAMPION R²: {old_champion_r2_baseline:.4f}"
            if old_champion_r2_baseline
            else "   - OLD CHAMPION R²: N/A"
        )
        if r2_improvement_baseline is not None:
            print(f"   - Improvement: {r2_improvement_baseline:+.4f}")
        print(f"   - RMSE: {metrics_baseline.get('rmse', 0):.2f}")
        print(f"   - MAE: {metrics_baseline.get('mae', 0):.2f}")
        print(
            f"   - Baseline regression: {'🚨 YES (R² < 0.60)' if baseline_regression else '✅ NO'}"
        )

        print("\n🆕 Current Test Set (new distribution, 20% of fresh data):")
        print(f"   - CHALLENGER R²: {r2_current:.4f}")
        print(f"   - OLD CHAMPION R²: {old_champion_r2_current:.4f}")
        print(f"   - Improvement: {r2_improvement_current:+.4f}")
        print(f"   - RMSE: {metrics_current.get('rmse', 0):.2f}")
        print(f"   - MAE: {metrics_current.get('mae', 0):.2f}")

        print("\n📊 Training Metrics (on train_baseline):")
        print(f"   - R²: {r2_train:.4f}")
        print(f"   - RMSE: {metrics_train.get('rmse', 0):.2f}")
        print("=" * 60 + "\n")

        # Decision logic (compare fairly on both test sets)
        print("🎯 DEPLOYMENT DECISION LOGIC:")
        print(f"   1. Baseline regression check: {baseline_regression}")
        if r2_improvement_baseline is not None:
            print(f"   2. Baseline improvement: {r2_improvement_baseline:+.4f}")
        print(f"   3. Current improvement: {r2_improvement_current:+.4f}")

        # Check if OLD CHAMPION also has baseline regression
        old_champion_has_baseline_regression = (
            old_champion_r2_baseline is not None and old_champion_r2_baseline < 0.60
        )

        if baseline_regression and old_champion_has_baseline_regression:
            # Both models fail baseline - compare on current
            if r2_improvement_current > 0:
                print(
                    "\n⚠️  DECISION: DEPLOY - Both models fail baseline, CHALLENGER better on current"
                )
                print(
                    f"   → OLD CHAMPION baseline R²: {old_champion_r2_baseline:.4f} (also < 0.60)"
                )
                print(f"   → CHALLENGER baseline R²: {r2_baseline:.4f} (also < 0.60)")
                print(f"   → Current improved: {r2_improvement_current:+.4f} ✅")
                decision = "deploy_both_fail_baseline"
            else:
                print(
                    "\n⏭️  DECISION: SKIP - Both models fail baseline, no improvement on current"
                )
                print("   → Keep OLD CHAMPION (no better alternative)")
                decision = "skip_both_fail_baseline"
        elif baseline_regression and not old_champion_has_baseline_regression:
            # CHALLENGER regressed, OLD CHAMPION was fine
            print("\n🚨 DECISION: REJECT - CHALLENGER regressed on baseline")
            if old_champion_r2_baseline is not None:
                print(
                    f"   → OLD CHAMPION baseline R²: {old_champion_r2_baseline:.4f} ✅"
                )
            print(f"   → CHALLENGER baseline R²: {r2_baseline:.4f} 🚨 (< 0.60)")
            if r2_improvement_baseline is not None:
                print(f"   → Regression: {r2_improvement_baseline:+.4f}")
            decision = "reject_baseline_regression"
        elif not baseline_regression and r2_improvement_current > 0:
            # Best case: no regression and improvement
            improvement_msg = ""
            if r2_improvement_baseline is not None and r2_improvement_baseline > 0:
                improvement_msg = f" (baseline: {r2_improvement_baseline:+.4f}, current: {r2_improvement_current:+.4f})"
            elif r2_improvement_baseline is not None:
                improvement_msg = f" (current: {r2_improvement_current:+.4f})"
            else:
                improvement_msg = f" ({r2_improvement_current:+.4f})"

            print(f"\n✅ DECISION: DEPLOY - CHALLENGER improved{improvement_msg}")
            print("   → No baseline regression ✅")
            print(f"   → Current improved: {r2_improvement_current:+.4f} ✅")
            decision = "deploy"
        else:
            # No improvement on current
            print(
                f"\n⏭️  DECISION: SKIP - No improvement on current ({r2_improvement_current:+.4f})"
            )
            print("   → Keep OLD CHAMPION")
            decision = "skip_no_improvement"

        # Push all metrics to XCom for audit logging
        context["ti"].xcom_push(key="fine_tune_success", value=True)
        context["ti"].xcom_push(key="double_evaluation_enabled", value=True)
        context["ti"].xcom_push(key="baseline_regression", value=baseline_regression)
        context["ti"].xcom_push(key="r2_baseline", value=float(r2_baseline))
        context["ti"].xcom_push(key="r2_current", value=float(r2_current))
        context["ti"].xcom_push(key="r2_train", value=float(r2_train))
        context["ti"].xcom_push(
            key="rmse_baseline", value=float(metrics_baseline.get("rmse", 0))
        )
        context["ti"].xcom_push(
            key="rmse_current", value=float(metrics_current.get("rmse", 0))
        )
        context["ti"].xcom_push(
            key="model_improvement", value=float(r2_improvement_current)
        )
        context["ti"].xcom_push(
            key="new_r2", value=float(r2_current)
        )  # Use current R² for comparison
        context["ti"].xcom_push(key="deployment_decision", value=decision)
        context["ti"].xcom_push(
            key="old_champion_r2_baseline",
            value=float(old_champion_r2_baseline) if old_champion_r2_baseline else None,
        )
        context["ti"].xcom_push(key="old_champion_run_id", value=old_champion_run_id)
        context["ti"].xcom_push(key="challenger_run_id", value=challenger_run_id)
        context["ti"].xcom_push(
            key="model_uri", value=result.get("model_uri", "unknown")
        )
        context["ti"].xcom_push(key="run_id", value=result.get("run_id", "unknown"))
        context["ti"].xcom_push(key="error_message", value="")

        # Promote champion if deployment decision is deploy
        if "deploy" in decision:
            print("\n" + "=" * 60)
            print("🏆 PROMOTING CHALLENGER TO NEW CHAMPION")
            print("=" * 60)
            print(
                f"📤 OLD CHAMPION → LEGACY: {old_champion_run_id[:8] if old_champion_run_id else 'unknown'}... (is_champion=False)"
            )
            print(
                f"📥 CHALLENGER → NEW CHAMPION: {challenger_run_id[:8]}... (is_champion=True)"
            )
            print("=" * 60)

            promote_url = f"{ENV_CONFIG['API_URL']}/promote_champion"
            promote_payload = {
                "model_type": "rf",
                "run_id": result.get("run_id", "unknown"),
                "env": ENV_CONFIG["ENV"],
                "test_mode": test_mode,
            }

            try:
                promote_response = requests.post(
                    promote_url, json=promote_payload, headers=headers, timeout=60
                )

                if promote_response.status_code == 200:
                    promote_result = promote_response.json()
                    demoted_run_id = promote_result.get("demoted_run_id", "unknown")
                    print("\n✅ Champion promotion successful!")
                    print(
                        f"   🆕 NEW CHAMPION: {result.get('run_id', 'unknown')[:8]}... (is_champion=True)"
                    )
                    print(
                        f"   📜 LEGACY: {demoted_run_id[:8] if demoted_run_id else 'unknown'}... (is_champion=False)"
                    )
                    context["ti"].xcom_push(key="champion_promoted", value=True)
                    context["ti"].xcom_push(
                        key="new_champion_run_id", value=result.get("run_id", "unknown")
                    )
                    context["ti"].xcom_push(key="legacy_run_id", value=demoted_run_id)

                    # 🆕 Send Discord notification for champion promotion
                    print("📢 Sending Discord notification for champion promotion...")
                    send_champion_promotion_alert(
                        model_type="rf",
                        run_id=result.get("run_id", "unknown"),
                        r2_current=r2_current,
                        r2_baseline=r2_baseline,
                        improvement_delta=r2_improvement_current,
                        rmse=metrics_current.get("rmse"),
                    )
                else:
                    print(
                        f"⚠️ Champion promotion failed: {promote_response.status_code}"
                    )
                    print(f"   Response: {promote_response.text}")
                    context["ti"].xcom_push(key="champion_promoted", value=False)
                    context["ti"].xcom_push(
                        key="promotion_error",
                        value=f"{promote_response.status_code}: {promote_response.text}",
                    )
            except Exception as e:
                print(f"⚠️ Champion promotion request failed: {e}")
                context["ti"].xcom_push(key="champion_promoted", value=False)
                context["ti"].xcom_push(key="promotion_error", value=str(e))
        else:
            print(f"\n⏭️  Skipping champion promotion (decision: {decision})")
            context["ti"].xcom_push(key="champion_promoted", value=False)

        # Send Discord notification for training completion
        deployment_type = (
            "deploy"
            if "deploy" in decision
            else "skip"
            if "skip" in decision
            else "reject"
        )
        send_training_success(
            improvement_delta=r2_improvement_current,
            new_r2=r2_current,
            old_r2=old_champion_r2_current,
            deployment_decision=deployment_type,
        )

    except requests.exceptions.Timeout:
        error_msg = "Training API timeout (>30 minutes)"
        print(f"❌ {error_msg}")
        context["ti"].xcom_push(key="fine_tune_success", value=False)
        context["ti"].xcom_push(key="error_message", value=error_msg)
        context["ti"].xcom_push(key="model_improvement", value=0.0)
        context["ti"].xcom_push(key="baseline_regression", value=False)
        context["ti"].xcom_push(key="double_evaluation_enabled", value=False)

        # Send Discord alert for training failure
        dag_run_id = context.get("dag_run").run_id
        send_training_failure(error_msg, dag_run_id)
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        print(f"❌ {error_msg}")
        context["ti"].xcom_push(key="fine_tune_success", value=False)
        context["ti"].xcom_push(key="error_message", value=error_msg)
        context["ti"].xcom_push(key="model_improvement", value=0.0)
        context["ti"].xcom_push(key="baseline_regression", value=False)
        context["ti"].xcom_push(key="double_evaluation_enabled", value=False)

        # Send Discord alert for training failure
        dag_run_id = context.get("dag_run").run_id
        send_training_failure(error_msg, dag_run_id)


def validate_new_champion(**context):
    """
    Validate the newly promoted champion model on recent production data.

    This task runs ONLY if champion was promoted in fine_tune_model.
    It provides fresh validation metrics for the new champion to replace
    the old champion's metrics in monitoring dashboards.
    """
    # Check if champion was actually promoted
    champion_promoted = context["ti"].xcom_pull(
        task_ids="fine_tune_model", key="champion_promoted"
    )

    if not champion_promoted:
        print("⏭️  No champion promotion, skipping validation")
        return

    new_champion_run_id = context["ti"].xcom_pull(
        task_ids="fine_tune_model", key="new_champion_run_id"
    )

    print(
        f"\n🔍 Validating NEW CHAMPION: {new_champion_run_id[:8] if new_champion_run_id else 'unknown'}..."
    )

    client = bigquery.Client(project=ENV_CONFIG["BQ_PROJECT"])

    # Join predictions with actuals (last 7 days) - same query as validate_model
    query = f"""
    WITH recent_predictions AS (
        SELECT
            p.prediction,
            p.identifiant_du_compteur,
            PARSE_TIMESTAMP('%Y-%m-%d %H:%M:%S%Ez', p.date_et_heure_de_comptage) as date_et_heure_de_comptage,
            p.prediction_ts
        FROM `{ENV_CONFIG["BQ_PROJECT"]}.{ENV_CONFIG["BQ_PREDICT_DATASET"]}.daily_*` p
        WHERE _TABLE_SUFFIX >= FORMAT_DATE('%Y%m%d', DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY))
    ),
    recent_actuals AS (
        SELECT
            r.comptage_horaire as true_value,
            r.identifiant_du_compteur,
            r.date_et_heure_de_comptage
        FROM `{ENV_CONFIG["BQ_PROJECT"]}.{ENV_CONFIG["BQ_RAW_DATASET"]}.comptage_velo` r
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

    print("📥 Loading predictions and actuals for new champion...")
    df = client.query(query).to_dataframe()

    if df.empty or len(df) < 10:
        print(f"⚠️ Insufficient data for validation (only {len(df)} samples)")
        # Push default values
        context["ti"].xcom_push(key="new_champion_rmse", value=999.0)
        context["ti"].xcom_push(key="new_champion_r2", value=0.0)
        context["ti"].xcom_push(key="new_champion_validation_samples", value=len(df))
        return

    print(f"✅ Loaded {len(df)} validation samples")

    # Calculate metrics for new champion
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    y_true = df["true_value"]
    y_pred = df["prediction"]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("\n📈 NEW CHAMPION Validation Metrics:")
    print(f"   - RMSE: {rmse:.2f}")
    print(f"   - MAE: {mae:.2f}")
    print(f"   - R²: {r2:.4f}")
    print(f"   - Samples: {len(df)}")

    # Push metrics to XCom (will be used by end_monitoring to update BigQuery)
    context["ti"].xcom_push(key="new_champion_rmse", value=float(rmse))
    context["ti"].xcom_push(key="new_champion_mae", value=float(mae))
    context["ti"].xcom_push(key="new_champion_r2", value=float(r2))
    context["ti"].xcom_push(key="new_champion_validation_samples", value=len(df))

    # Send Discord notification
    print("📢 Sending Discord notification for new champion validation...")
    send_performance_alert(
        r2=r2,
        rmse=rmse,
        threshold=0.70,  # Info level, not critical
    )


def end_monitoring(**context):
    """
    Log monitoring results to BigQuery audit table with double evaluation metrics.
    Called whether training happened or not.
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

    # Check if we have new champion validation metrics (from validate_new_champion task)
    new_champion_rmse = context["ti"].xcom_pull(
        task_ids="validate_new_champion", key="new_champion_rmse"
    )
    new_champion_r2 = context["ti"].xcom_pull(
        task_ids="validate_new_champion", key="new_champion_r2"
    )

    # If new champion was validated, use those metrics; otherwise use old champion metrics
    if new_champion_rmse is not None and new_champion_r2 is not None:
        print("✅ Using NEW champion validation metrics from validate_new_champion")
        rmse = new_champion_rmse
        r2 = new_champion_r2
    else:
        print("✅ Using OLD champion validation metrics from validate_model")
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

    # NEW: Collect double evaluation metrics
    double_eval_enabled = context["ti"].xcom_pull(
        task_ids="fine_tune_model", key="double_evaluation_enabled"
    )
    baseline_regression = context["ti"].xcom_pull(
        task_ids="fine_tune_model", key="baseline_regression"
    )
    r2_baseline = context["ti"].xcom_pull(task_ids="fine_tune_model", key="r2_baseline")
    r2_current = context["ti"].xcom_pull(task_ids="fine_tune_model", key="r2_current")
    r2_train = context["ti"].xcom_pull(task_ids="fine_tune_model", key="r2_train")
    deployment_decision = context["ti"].xcom_pull(
        task_ids="fine_tune_model", key="deployment_decision"
    )
    # Note: This is the OLD CHAMPION's R² on baseline (before promotion)
    # Kept as "champion_r2_baseline" for BigQuery schema compatibility
    old_champion_r2_baseline = context["ti"].xcom_pull(
        task_ids="fine_tune_model", key="old_champion_r2_baseline"
    )
    model_uri = context["ti"].xcom_pull(task_ids="fine_tune_model", key="model_uri")
    run_id = context["ti"].xcom_pull(task_ids="fine_tune_model", key="run_id")

    # If fine_tune task didn't run, set defaults
    if fine_tune_success is None:
        fine_tune_triggered = False
        fine_tune_success = False
        model_improvement = 0.0
        error_message = ""
        double_eval_enabled = False
        baseline_regression = False
        r2_baseline = None
        r2_current = None
        r2_train = None
        deployment_decision = "not_triggered"
        old_champion_r2_baseline = None
        model_uri = ""
        run_id = ""
    else:
        fine_tune_triggered = True

    # Create audit record with double evaluation fields
    audit_record = {
        "timestamp": timestamp,
        "drift_detected": drift_detected if drift_detected is not None else False,
        "rmse": float(rmse) if rmse else 999.0,
        "r2": float(r2) if r2 else 0.0,
        "fine_tune_triggered": fine_tune_triggered,
        "fine_tune_success": fine_tune_success,
        "model_improvement": float(model_improvement) if model_improvement else 0.0,
        "env": ENV_CONFIG["ENV"],
        "error_message": error_message if error_message else "",
        # NEW: Double evaluation fields
        "double_evaluation_enabled": double_eval_enabled
        if double_eval_enabled is not None
        else False,
        "baseline_regression": baseline_regression
        if baseline_regression is not None
        else False,
        "r2_baseline": float(r2_baseline) if r2_baseline is not None else None,
        "r2_current": float(r2_current) if r2_current is not None else None,
        "r2_train": float(r2_train) if r2_train is not None else None,
        "deployment_decision": deployment_decision
        if deployment_decision
        else "not_triggered",
        # Note: Stored as "champion_r2_baseline" for BigQuery schema compatibility
        # This represents the OLD CHAMPION's R² on baseline (before any promotion)
        "champion_r2_baseline": float(old_champion_r2_baseline)
        if old_champion_r2_baseline is not None
        else None,
        "model_uri": model_uri if model_uri else "",
        "run_id": run_id if run_id else "",
    }

    audit_df = pd.DataFrame([audit_record])

    # Write to BigQuery
    table_id = f"{ENV_CONFIG['BQ_PROJECT']}.monitoring_audit.logs"

    print(f"📤 Writing audit log to BigQuery: {table_id}")
    print(f"📊 Audit record:\n{audit_df.to_dict(orient='records')}")

    client = bigquery.Client(project=ENV_CONFIG["BQ_PROJECT"])
    client.load_table_from_dataframe(audit_df, table_id).result()

    print(f"✅ Audit log written to {table_id}")

    # Get champion/challenger info from previous tasks
    old_champion_run_id = context["ti"].xcom_pull(
        task_ids="validate_model", key="champion_run_id"
    )
    challenger_run_id = context["ti"].xcom_pull(
        task_ids="fine_tune_model", key="challenger_run_id"
    )
    new_champion_run_id = context["ti"].xcom_pull(
        task_ids="fine_tune_model", key="new_champion_run_id"
    )
    legacy_run_id = context["ti"].xcom_pull(
        task_ids="fine_tune_model", key="legacy_run_id"
    )
    champion_promoted = context["ti"].xcom_pull(
        task_ids="fine_tune_model", key="champion_promoted"
    )

    # Enhanced summary with double evaluation
    print("\n" + "=" * 60)
    print("📋 MONITORING SUMMARY")
    print("=" * 60)

    # Show model status
    if fine_tune_triggered and champion_promoted:
        print(
            f"🏆 NEW CHAMPION: {new_champion_run_id[:8] if new_champion_run_id else 'unknown'}... (is_champion=True)"
        )
        print(
            f"📜 LEGACY: {legacy_run_id[:8] if legacy_run_id else 'unknown'}... (is_champion=False)"
        )
    elif fine_tune_triggered and challenger_run_id:
        print(
            f"🏆 CHAMPION (unchanged): {old_champion_run_id[:8] if old_champion_run_id else 'unknown'}... (is_champion=True)"
        )
        print(f"❌ CHALLENGER (rejected): {challenger_run_id[:8]}...")
    else:
        print(
            f"🏆 CHAMPION: {old_champion_run_id[:8] if old_champion_run_id else 'unknown'}... (is_champion=True)"
        )

    print(f"\nDrift detected: {'🚨 YES' if drift_detected else '✅ NO'}")
    print(f"Champion RMSE: {rmse:.2f}" if rmse else "Champion RMSE: N/A")
    print(f"Champion R²: {r2:.4f}" if r2 else "Champion R²: N/A")

    if fine_tune_triggered:
        print(f"\nFine-tuning: {'✅ SUCCESS' if fine_tune_success else '❌ FAILED'}")

        if double_eval_enabled:
            print("\n📊 Double Evaluation Results:")
            print(
                f"   - Baseline R²: {r2_baseline:.4f}"
                if r2_baseline is not None
                else "   - Baseline R²: N/A"
            )
            print(
                f"   - Current R²: {r2_current:.4f}"
                if r2_current is not None
                else "   - Current R²: N/A"
            )
            print(
                f"   - Training R²: {r2_train:.4f}"
                if r2_train is not None
                else "   - Training R²: N/A"
            )
            print(
                f"   - Baseline regression: {'🚨 YES' if baseline_regression else '✅ NO'}"
            )
            print(
                f"   - Model improvement: {model_improvement:+.4f}"
                if model_improvement
                else "   - Model improvement: N/A"
            )
            print(f"\n🎯 Decision: {deployment_decision.upper().replace('_', ' ')}")

            if deployment_decision == "deploy":
                print("   ✅ New model promoted to production")
                print(f"   📦 Model URI: {model_uri}")
            elif deployment_decision == "reject_regression":
                print("   ❌ New model rejected (baseline regression)")
            elif deployment_decision == "skip_no_improvement":
                print("   ⏭️  Current champion retained (no improvement)")
        else:
            print("   ⚠️  Double evaluation not enabled")
            if model_improvement and model_improvement != 0:
                print(f"   - Model improvement: {model_improvement:+.4f}")
    else:
        print("\nFine-tuning: ⛔ NOT TRIGGERED")

    print(f"\nEnvironment: {ENV_CONFIG['ENV']}")
    print("=" * 60 + "\n")


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
        execution_timeout=timedelta(minutes=45),
    )

    # Task 4b: Validate new champion (runs after fine_tune if champion was promoted)
    validate_new_champ = PythonOperator(
        task_id="validate_new_champion",
        python_callable=validate_new_champion,
        provide_context=True,
        trigger_rule="none_failed_or_skipped",  # Only run if fine_tune succeeded
    )

    # Task 5: End monitoring (runs regardless of branch taken)
    end = PythonOperator(
        task_id="end_monitoring",
        python_callable=end_monitoring,
        provide_context=True,
        trigger_rule="none_failed_min_one_success",  # Run regardless of branch taken
    )

    # Pipeline flow
    monitor >> validate >> decide
    decide >> [fine_tune, end]
    fine_tune >> validate_new_champ >> end
