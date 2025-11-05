"""Custom Prometheus exporter for Airflow DAG metrics and XCom values.

Exposes:
- Generic Airflow metrics (DAG/task durations)
- Business metrics from XCom (ingestion, predictions, drift, model metrics)

Run as standalone Flask app on port 9101.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from typing import Any, Optional

import requests
from flask import Flask, Response
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from requests.auth import HTTPBasicAuth

try:
    from google.cloud import bigquery

    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False
    print(
        "⚠️  google-cloud-bigquery not installed. BigQuery metrics will be unavailable."
    )

# === Prometheus Metrics ===

# Generic Airflow metrics
AIRFLOW_DAG_RUN_DURATION = Histogram(
    "airflow_dag_run_duration_seconds",
    "DAG run duration in seconds",
    labelnames=("dag_id", "state"),
)
AIRFLOW_TASK_DURATION = Histogram(
    "airflow_task_duration_seconds",
    "Task duration in seconds",
    labelnames=("dag_id", "task_id", "state"),
)
AIRFLOW_DAG_RUN_COUNT = Counter(
    "airflow_dag_runs_total",
    "Total DAG runs",
    labelnames=("dag_id", "state"),
)

# DAG 1: daily_fetch_bike_data
BIKE_API_RECORDS_FETCHED = Gauge(
    "bike_api_records_fetched_total",
    "Total records fetched from bike API",
)
BIKE_RECORDS_INGESTED = Gauge(
    "bike_records_ingested_total",
    "Records ingested after deduplication",
)
BIKE_DEDUPLICATION_RATIO = Gauge(
    "bike_deduplication_ratio",
    "Deduplication ratio (deduplicated/total)",
)
BIKE_INGESTION_DURATION = Gauge(
    "bike_ingestion_duration_seconds",
    "Last ingestion duration in seconds",
)

# DAG 2: daily_prediction
BIKE_PREDICTIONS_GENERATED = Gauge(
    "bike_predictions_generated_total",
    "Total predictions generated",
)
BIKE_PREDICTION_RMSE = Gauge(
    "bike_prediction_rmse",
    "Latest prediction RMSE",
)
BIKE_PREDICTION_MAE = Gauge(
    "bike_prediction_mae",
    "Latest prediction MAE",
)
BIKE_PREDICTION_R2 = Gauge(
    "bike_prediction_r2",
    "Latest prediction R² score",
)

# DAG 3: monitor_and_fine_tune
BIKE_DRIFT_DETECTED = Gauge(
    "bike_drift_detected",
    "Data drift detected flag (0/1)",
)
BIKE_DRIFT_SHARE = Gauge(
    "bike_drift_share",
    "Data drift share (0.0-1.0)",
)
BIKE_DRIFTED_FEATURES_COUNT = Gauge(
    "bike_drifted_features_count",
    "Number of drifted features",
)
BIKE_MODEL_R2_PRODUCTION = Gauge(
    "bike_model_r2_production",
    "Production model R² score",
)
BIKE_MODEL_RMSE_PRODUCTION = Gauge(
    "bike_model_rmse_production",
    "Production model RMSE",
)
BIKE_TRAINING_RUNS = Counter(
    "bike_training_runs_total",
    "Training runs executed",
    labelnames=("status",),
)
BIKE_MODEL_IMPROVEMENT_DELTA = Gauge(
    "bike_model_improvement_delta",
    "Model improvement delta (R² new - R² old)",
)
BIKE_MODEL_DEPLOYMENTS = Counter(
    "bike_model_deployments_total",
    "Model deployment decisions",
    labelnames=("decision",),  # deploy, skip, reject
)

# Double evaluation - Complete metrics (4 R² values)
BIKE_MODEL_R2_CHAMPION_BASELINE = Gauge(
    "bike_model_r2_champion_baseline",
    "Champion model R² on test_baseline (fixed reference, 181K samples)",
)
BIKE_MODEL_R2_CHAMPION_CURRENT = Gauge(
    "bike_model_r2_champion_current",
    "Champion model R² on test_current (new distribution, 20% split)",
)
BIKE_MODEL_R2_CHALLENGER_BASELINE = Gauge(
    "bike_model_r2_challenger_baseline",
    "Challenger model R² on test_baseline (regression check, threshold 0.60)",
)
BIKE_MODEL_R2_CHALLENGER_CURRENT = Gauge(
    "bike_model_r2_challenger_current",
    "Challenger model R² on test_current (improvement check vs champion)",
)

# === Airflow API Client ===


class AirflowAPIClient:
    """Lightweight client for Airflow REST API v1."""

    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url.rstrip("/")
        self.auth = HTTPBasicAuth(username, password)
        self.session = requests.Session()
        self.session.auth = self.auth

    def get_dag_runs(
        self, dag_id: str, limit: int = 10, start_date: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Get recent DAG runs for a specific DAG."""
        url = f"{self.base_url}/api/v1/dags/{dag_id}/dagRuns"
        params: dict[str, str | int] = {"limit": limit, "order_by": "-execution_date"}
        # Note: start_date_gte filter seems not well supported, so we fetch more and filter manually
        # if start_date:
        #     params["start_date_gte"] = start_date

        try:
            response = self.session.get(url, params=params, timeout=10)  # type: ignore[arg-type]
            response.raise_for_status()
            data = response.json()
            dag_runs = data.get("dag_runs", [])

            # Manual filtering if start_date provided
            if start_date and dag_runs:
                from dateutil import parser as date_parser  # type: ignore[import-untyped]

                start_dt = date_parser.parse(start_date)
                dag_runs = [
                    run
                    for run in dag_runs
                    if run.get("start_date")
                    and date_parser.parse(run["start_date"]) >= start_dt
                ]

            return dag_runs
        except Exception as e:
            print(f"⚠️  Error fetching DAG runs for {dag_id}: {e}")
            return []

    def get_task_instances(self, dag_id: str, dag_run_id: str) -> list[dict[str, Any]]:
        """Get task instances for a specific DAG run."""
        url = f"{self.base_url}/api/v1/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances"

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get("task_instances", [])
        except Exception as e:
            print(f"⚠️  Error fetching task instances for {dag_id}/{dag_run_id}: {e}")
            return []

    def get_xcom_value(
        self, dag_id: str, dag_run_id: str, task_id: str, key: str
    ) -> Any:
        """Get XCom value for a specific task."""
        url = f"{self.base_url}/api/v1/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/xcomEntries/{key}"

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get("value")
        except Exception as e:
            print(
                f"⚠️  Error fetching XCom {key} for {dag_id}/{dag_run_id}/{task_id}: {e}"
            )
            return None


# === Metrics Collector ===


class AirflowMetricsCollector:
    """Collect and update Prometheus metrics from Airflow API."""

    def __init__(self, client: AirflowAPIClient):
        self.client = client
        self.dag_ids = [
            "daily_fetch_bike_data",
            "daily_prediction",
            "monitor_and_fine_tune",
        ]
        self.bq_client = None
        if BIGQUERY_AVAILABLE:
            try:
                self.bq_client = bigquery.Client(
                    project=os.getenv("GOOGLE_CLOUD_PROJECT", "datascientest-460618")
                )
                print("✅ BigQuery client initialized for reading audit logs")
            except Exception as e:
                print(f"⚠️  Failed to initialize BigQuery client: {e}")

    def _get_latest_monitoring_metrics_from_bq(self) -> Optional[dict]:
        """Fetch the latest monitoring metrics from BigQuery audit table.

        Returns the most recent row from monitoring_audit.logs table.
        """
        if not self.bq_client:
            return None

        try:
            query = """
            SELECT
                timestamp,
                drift_detected,
                r2,
                rmse,
                r2_baseline,
                r2_current,
                r2_train,
                deployment_decision,
                model_improvement,
                fine_tune_success,
                baseline_regression,
                champion_r2_baseline
            FROM `datascientest-460618.monitoring_audit.logs`
            ORDER BY timestamp DESC
            LIMIT 1
            """

            query_job = self.bq_client.query(query)
            results = list(query_job.result())

            if results:
                row = results[0]
                metrics = {
                    "timestamp": row.timestamp,
                    "drift_detected": row.drift_detected,
                    "r2": row.r2,
                    "rmse": row.rmse,
                    "r2_baseline": row.r2_baseline,
                    "r2_current": row.r2_current,
                    "r2_train": row.r2_train,
                    "deployment_decision": row.deployment_decision,
                    "model_improvement": row.model_improvement,
                    "fine_tune_success": row.fine_tune_success,
                    "baseline_regression": row.baseline_regression,
                    "champion_r2_baseline": row.champion_r2_baseline,
                }
                return metrics
            return None
        except Exception as e:
            print(f"⚠️  Error querying BigQuery audit table: {e}")
            return None

    def collect_all_metrics(self) -> None:
        """Collect metrics from all DAGs."""
        print(f"🔄 Collecting Airflow metrics at {datetime.now().isoformat()}")

        # Get runs from last 7 days (with UTC timezone for comparison)
        from datetime import timezone

        start_date = (datetime.now(timezone.utc) - timedelta(days=7)).strftime(
            "%Y-%m-%dT%H:%M:%S%z"
        )
        print(f"   Looking for runs since: {start_date}")

        for dag_id in self.dag_ids:
            try:
                print(f"📊 Collecting metrics for {dag_id}...")
                self._collect_dag_metrics(dag_id, start_date)
            except Exception as e:
                import traceback

                print(f"❌ Error collecting metrics for {dag_id}: {e}")
                print(traceback.format_exc())

        print("✅ Metrics collection complete")

    def _collect_dag_metrics(self, dag_id: str, start_date: str) -> None:
        """Collect metrics for a specific DAG."""
        runs = self.client.get_dag_runs(dag_id, limit=20, start_date=start_date)

        if not runs:
            print(f"⚠️  No recent runs found for {dag_id}")
            return

        print(f"📊 Processing {len(runs)} runs for {dag_id}")

        # Get the most recent successful run for XCom values
        latest_success_run = None
        for run in runs:
            if run.get("state") == "success":
                latest_success_run = run
                break

        # Collect generic Airflow metrics (all runs)
        for run in runs:
            state = run.get("state", "unknown")
            start_date_str = run.get("start_date")
            end_date_str = run.get("end_date")

            # Count DAG runs
            AIRFLOW_DAG_RUN_COUNT.labels(dag_id=dag_id, state=state).inc(0)

            # Calculate duration if completed
            if start_date_str and end_date_str:
                start_dt = datetime.fromisoformat(start_date_str.replace("Z", "+00:00"))
                end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                duration = (end_dt - start_dt).total_seconds()
                AIRFLOW_DAG_RUN_DURATION.labels(dag_id=dag_id, state=state).observe(
                    duration
                )

            # Collect task metrics for this run
            dag_run_id = run.get("dag_run_id")
            if dag_run_id:
                self._collect_task_metrics(dag_id, dag_run_id)

        # Collect business metrics from latest successful run
        if latest_success_run:
            dag_run_id = latest_success_run.get("dag_run_id")
            if dag_run_id:
                self._collect_business_metrics(dag_id, dag_run_id)

    def _collect_task_metrics(self, dag_id: str, dag_run_id: str) -> None:
        """Collect task-level metrics for a DAG run."""
        tasks = self.client.get_task_instances(dag_id, dag_run_id)

        for task in tasks:
            task_id = task.get("task_id", "unknown")
            state = task.get("state", "unknown")
            start_date_str = task.get("start_date")
            end_date_str = task.get("end_date")

            if start_date_str and end_date_str:
                start_dt = datetime.fromisoformat(start_date_str.replace("Z", "+00:00"))
                end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                duration = (end_dt - start_dt).total_seconds()
                AIRFLOW_TASK_DURATION.labels(
                    dag_id=dag_id, task_id=task_id, state=state
                ).observe(duration)

    def _collect_business_metrics(self, dag_id: str, dag_run_id: str) -> None:
        """Collect business metrics from XCom values."""
        print(f"📦 Collecting business metrics from {dag_id}/{dag_run_id}")

        if dag_id == "daily_fetch_bike_data":
            self._collect_ingestion_metrics(dag_id, dag_run_id)
        elif dag_id == "daily_prediction":
            self._collect_prediction_metrics(dag_id, dag_run_id)
        elif dag_id == "monitor_and_fine_tune":
            self._collect_monitoring_metrics(dag_id, dag_run_id)

    def _collect_ingestion_metrics(self, dag_id: str, dag_run_id: str) -> None:
        """Collect metrics from DAG 1: daily_fetch_bike_data."""
        # Task: fetch_to_bigquery pushes: records_count, table_id, ingestion_date
        records_count = self.client.get_xcom_value(
            dag_id, dag_run_id, "fetch_to_bigquery", "records_count"
        )

        if records_count is not None:
            BIKE_RECORDS_INGESTED.set(float(records_count))
            print(f"   - Records ingested: {records_count}")

    def _collect_prediction_metrics(self, dag_id: str, dag_run_id: str) -> None:
        """Collect metrics from DAG 2: daily_prediction."""
        # Task: predict_daily_data pushes: predictions_count, rmse, mae, r2, pred_table
        predictions_count = self.client.get_xcom_value(
            dag_id, dag_run_id, "predict_daily_data", "predictions_count"
        )
        rmse = self.client.get_xcom_value(
            dag_id, dag_run_id, "predict_daily_data", "rmse"
        )
        mae = self.client.get_xcom_value(
            dag_id, dag_run_id, "predict_daily_data", "mae"
        )
        r2 = self.client.get_xcom_value(dag_id, dag_run_id, "predict_daily_data", "r2")

        if predictions_count is not None:
            BIKE_PREDICTIONS_GENERATED.set(float(predictions_count))
            print(f"   - Predictions generated: {predictions_count}")

        if rmse is not None:
            rmse_float = float(rmse)
            BIKE_PREDICTION_RMSE.set(rmse_float)
            print(f"   - RMSE: {rmse_float:.2f}")

        if mae is not None:
            mae_float = float(mae)
            BIKE_PREDICTION_MAE.set(mae_float)
            print(f"   - MAE: {mae_float:.2f}")

        if r2 is not None:
            r2_float = float(r2)
            BIKE_PREDICTION_R2.set(r2_float)
            print(f"   - R²: {r2_float:.4f}")

    def _collect_monitoring_metrics(self, dag_id: str, dag_run_id: str) -> None:
        """Collect metrics from DAG 3: monitor_and_fine_tune.

        PRIORITY: Read from BigQuery audit table (source of truth).
        FALLBACK: Read from Airflow XCom if BigQuery unavailable.
        """
        # Try BigQuery first (most reliable source)
        bq_metrics = self._get_latest_monitoring_metrics_from_bq()

        if bq_metrics:
            print("   📊 Using metrics from BigQuery audit table (source of truth)")

            # Drift metrics
            if bq_metrics.get("drift_detected") is not None:
                drift_detected = bool(bq_metrics["drift_detected"])
                BIKE_DRIFT_DETECTED.set(1.0 if drift_detected else 0.0)
                print(f"   - Drift detected: {drift_detected}")

            # === DOUBLE EVALUATION METRICS ===
            # Extract all 4 R² values from BigQuery
            champion_r2_current = bq_metrics.get("r2")  # From validate_model task
            champion_r2_baseline = bq_metrics.get(
                "champion_r2_baseline"
            )  # Champion eval on baseline
            challenger_r2_baseline = bq_metrics.get(
                "r2_baseline"
            )  # Challenger eval on baseline
            challenger_r2_current = bq_metrics.get(
                "r2_current"
            )  # Challenger eval on current
            rmse_champion = bq_metrics.get("rmse")  # Champion RMSE

            print("\n   📊 Double Evaluation Metrics:")

            # Champion metrics
            if champion_r2_current is not None:
                r2_float = float(champion_r2_current)
                BIKE_MODEL_R2_CHAMPION_CURRENT.set(r2_float)
                # Legacy metric (backward compatibility)
                BIKE_MODEL_R2_PRODUCTION.set(r2_float)
                print(f"   - Champion R² (test_current): {r2_float:.4f}")

            if champion_r2_baseline is not None:
                r2_float = float(champion_r2_baseline)
                BIKE_MODEL_R2_CHAMPION_BASELINE.set(r2_float)
                print(f"   - Champion R² (test_baseline): {r2_float:.4f}")

            # Challenger metrics
            if challenger_r2_baseline is not None:
                r2_float = float(challenger_r2_baseline)
                BIKE_MODEL_R2_CHALLENGER_BASELINE.set(r2_float)
                print(f"   - Challenger R² (test_baseline): {r2_float:.4f}")

            if challenger_r2_current is not None:
                r2_float = float(challenger_r2_current)
                BIKE_MODEL_R2_CHALLENGER_CURRENT.set(r2_float)
                # Legacy metric (backward compatibility)
                BIKE_PREDICTION_R2.set(r2_float)
                print(f"   - Challenger R² (test_current): {r2_float:.4f}")

            # RMSE
            if rmse_champion is not None:
                rmse_float = float(rmse_champion)
                BIKE_MODEL_RMSE_PRODUCTION.set(rmse_float)
                print(f"   - Champion RMSE: {rmse_float:.2f}")

            # Training metrics
            fine_tune_success = bq_metrics.get("fine_tune_success")
            if fine_tune_success is not None:
                status = "success" if fine_tune_success else "failed"
                BIKE_TRAINING_RUNS.labels(status=status).inc(0)
                print(f"   - Training status: {status}")

            model_improvement = bq_metrics.get("model_improvement")
            if model_improvement is not None:
                improvement_float = float(model_improvement)
                BIKE_MODEL_IMPROVEMENT_DELTA.set(improvement_float)
                print(f"   - Model improvement: {improvement_float:+.4f}")

            deployment_decision = bq_metrics.get("deployment_decision")
            if deployment_decision is not None:
                BIKE_MODEL_DEPLOYMENTS.labels(decision=str(deployment_decision)).inc(0)
                print(f"   - Deployment decision: {deployment_decision}")

            return  # Done with BigQuery metrics

        # FALLBACK: Use XCom if BigQuery unavailable
        print("   ⚠️  BigQuery unavailable, falling back to XCom")

        # Drift metrics - get drift_summary dict
        drift_summary = self.client.get_xcom_value(
            dag_id, dag_run_id, "monitor_drift", "drift_summary"
        )

        if drift_summary:
            # Parse drift_summary string into dict if needed
            if isinstance(drift_summary, str):
                import ast

                drift_summary = ast.literal_eval(drift_summary)

            drift_detected = drift_summary.get("drift_detected")
            drift_share = drift_summary.get("drift_share")
            drifted_features = drift_summary.get("drifted_features", [])

            if drift_detected is not None:
                BIKE_DRIFT_DETECTED.set(1.0 if drift_detected else 0.0)
                print(f"   - Drift detected: {drift_detected}")

            if drift_share is not None:
                BIKE_DRIFT_SHARE.set(float(drift_share))
                print(f"   - Drift share: {drift_share:.2%}")

            if drifted_features is not None:
                drifted_count = (
                    len(drifted_features)
                    if isinstance(drifted_features, list)
                    else int(drifted_features)
                )
                BIKE_DRIFTED_FEATURES_COUNT.set(float(drifted_count))
                print(f"   - Drifted features: {drifted_count}")

        # Model performance metrics from XCom (fallback)
        r2 = self.client.get_xcom_value(dag_id, dag_run_id, "validate_model", "r2")
        rmse = self.client.get_xcom_value(dag_id, dag_run_id, "validate_model", "rmse")

        if r2 is not None:
            r2_float = float(r2)
            BIKE_MODEL_R2_PRODUCTION.set(r2_float)
            print(f"   - Production R² (XCom fallback): {r2_float:.4f}")

        if rmse is not None:
            rmse_float = float(rmse)
            BIKE_MODEL_RMSE_PRODUCTION.set(rmse_float)
            print(f"   - Production RMSE (XCom fallback): {rmse_float:.2f}")


# === Flask App ===

app = Flask(__name__)

# Initialize Airflow client
AIRFLOW_BASE_URL = os.getenv("AIRFLOW_BASE_URL", "http://airflow-webserver:8080")
AIRFLOW_USERNAME = os.getenv("AIRFLOW_USERNAME", "airflow")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "airflow")

airflow_client = AirflowAPIClient(AIRFLOW_BASE_URL, AIRFLOW_USERNAME, AIRFLOW_PASSWORD)
metrics_collector = AirflowMetricsCollector(airflow_client)

# Track last collection time
last_collection_time = 0
COLLECTION_INTERVAL = 60  # Collect every 60 seconds


@app.route("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    global last_collection_time

    # Collect metrics if interval has passed
    current_time = time.time()
    time_since_last = current_time - last_collection_time
    print(
        f"⏱️  /metrics called - Time since last collection: {time_since_last:.1f}s (interval: {COLLECTION_INTERVAL}s)"
    )

    if time_since_last >= COLLECTION_INTERVAL:
        print("🚀 Triggering collection...")
        try:
            metrics_collector.collect_all_metrics()
            last_collection_time = current_time
        except Exception as e:
            import traceback

            print(f"❌ Error during metrics collection: {e}")
            print(traceback.format_exc())
    else:
        print(
            f"⏭️  Skipping collection (next in {COLLECTION_INTERVAL - time_since_last:.1f}s)"
        )

    return Response(generate_latest(), mimetype="text/plain")


@app.route("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    print("🚀 Starting Airflow Prometheus Exporter on port 9101")
    print(f"   - Airflow URL: {AIRFLOW_BASE_URL}")
    print(f"   - Collection interval: {COLLECTION_INTERVAL}s")
    print(f"   - Monitoring DAGs: {metrics_collector.dag_ids}")
    app.run(host="0.0.0.0", port=9101)  # nosec B104
