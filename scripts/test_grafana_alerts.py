#!/usr/bin/env python3
"""
Test Grafana alerts by injecting mock metrics into Prometheus
Requires: pip install prometheus-client
"""

import time
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
import argparse


PUSHGATEWAY_URL = "localhost:9091"


def push_metrics(registry):
    """Push metrics to Prometheus Pushgateway with airflow-metrics job name"""
    # Use the SAME job name as the real airflow-exporter to override real metrics
    job_name = "airflow-metrics"
    grouping_key = {"instance": "airflow-exporter:9101"}

    try:
        push_to_gateway(
            PUSHGATEWAY_URL, job=job_name, registry=registry, grouping_key=grouping_key
        )
        print(f"✅ Pushed metrics to {PUSHGATEWAY_URL}")
        print(f"   Job: {job_name}, Instance: {grouping_key['instance']}")
    except Exception as e:
        print(f"❌ Failed to push metrics: {e}")
        print("   Ensure Pushgateway is running: docker compose ps pushgateway")


def test_high_drift():
    """Test: High drift (>50%) alert"""
    print("\n🧪 Test 1: High Drift (>50%)")
    registry = CollectorRegistry()
    drift_share = Gauge("bike_drift_share", "Mock drift share", registry=registry)
    drift_share.set(0.65)  # 65% drift
    push_metrics(registry)
    print("   Injected: bike_drift_share=0.65")
    print("   Expected: 'High Data Drift Detected' alert in 15min")


def test_low_r2():
    """Test: Low R² (<0.65) alert"""
    print("\n🧪 Test 2: Low R² (<0.65)")
    registry = CollectorRegistry()
    r2 = Gauge("bike_model_r2_champion_current", "Mock R² score", registry=registry)
    r2.set(0.58)  # R² = 0.58
    push_metrics(registry)
    print("   Injected: bike_model_r2_champion_current=0.58")
    print("   Expected: 'Model R² Critically Low' alert in 5min")


def test_high_rmse():
    """Test: High RMSE (>70) alert"""
    print("\n🧪 Test 3: High RMSE (>70)")
    registry = CollectorRegistry()
    rmse = Gauge("bike_model_rmse_production", "Mock RMSE", registry=registry)
    rmse.set(85.0)  # RMSE = 85
    push_metrics(registry)
    print("   Injected: bike_model_rmse_production=85.0")
    print("   Expected: 'Model RMSE Above Threshold' alert in 5min")


def test_critical_drift_and_r2():
    """Test: Combined drift + low R² alert"""
    print("\n🧪 Test 4: Critical Drift + Low R² (combined)")
    registry = CollectorRegistry()
    drift_share = Gauge("bike_drift_share", "Mock drift share", registry=registry)
    r2 = Gauge("bike_model_r2_champion_current", "Mock R² score", registry=registry)
    drift_share.set(0.55)  # 55% drift
    r2.set(0.68)  # R² = 0.68
    push_metrics(registry)
    print("   Injected: bike_drift_share=0.55, bike_model_r2_champion_current=0.68")
    print("   Expected: 'Critical Drift + Declining R²' alert in 5min")


def test_api_error_rate():
    """Test: High API error rate (>5%) alert"""
    print("\n🧪 Test 5: High API Error Rate (>5%)")
    registry = CollectorRegistry()
    requests_total = Gauge(
        "fastapi_requests_total", "Mock total requests", registry=registry
    )
    errors_total = Gauge("fastapi_errors_total", "Mock total errors", registry=registry)
    requests_total.set(1000)
    errors_total.set(120)  # 12% error rate
    push_metrics(registry)
    print("   Injected: fastapi_requests_total=1000, fastapi_errors_total=120")
    print("   Expected: 'API Error Rate High' alert in 5min")


def test_no_data_ingestion():
    """Test: No data ingested alert"""
    print("\n🧪 Test 6: No Data Ingested (0 records)")
    registry = CollectorRegistry()
    ingested = Gauge(
        "bike_records_ingested_total", "Mock ingested records", registry=registry
    )
    ingested.set(0)  # No data
    push_metrics(registry)
    print("   Injected: bike_records_ingested_total=0")
    print("   Expected: 'No Data Ingested' alert in 36h (long wait)")


def test_service_down():
    """Test: Service down alert"""
    print("\n🧪 Test 7: Service Down (up=0)")
    registry = CollectorRegistry()
    up = Gauge("up", "Mock service status", registry=registry)
    up.set(0)  # Service down
    push_metrics(registry)
    print("   Injected: up=0")
    print("   Expected: 'Service Down' alert in 1min")


def test_dag_too_slow():
    """Test: DAG running too long (>3600s) alert"""
    print("\n🧪 Test 8: DAG Too Slow (>3600s)")
    registry = CollectorRegistry()
    from prometheus_client import Histogram

    # Create histogram and observe a value >3600s
    dag_duration = Histogram(
        "airflow_dag_run_duration_seconds",
        "Mock DAG duration",
        labelnames=("dag_id", "state"),
        registry=registry,
    )
    dag_duration.labels(dag_id="monitor_and_fine_tune", state="running").observe(3700)
    push_metrics(registry)
    print(
        "   Injected: airflow_dag_run_duration_seconds{dag_id='monitor_and_fine_tune'}=3700"
    )
    print("   Expected: 'DAG Running Too Long' alert in 10min")


def test_dashboards():
    """Test: Dashboard metrics (no alerts, visual verification only)"""
    print("\n🧪 Test 9: Dashboard Metrics (visual verification)")
    registry = CollectorRegistry()

    # Drift metrics
    Gauge("bike_drift_detected", "Mock drift detected", registry=registry).set(1)
    Gauge(
        "bike_drifted_features_count", "Mock drifted features", registry=registry
    ).set(8)

    # Training/deployment metrics
    Gauge("bike_model_improvement_delta", "Mock improvement", registry=registry).set(
        0.05
    )
    from prometheus_client import Counter

    Counter(
        "bike_model_deployments_total",
        "Mock deployments",
        labelnames=("decision",),
        registry=registry,
    ).labels(decision="deploy").inc()
    Counter(
        "bike_training_runs_total",
        "Mock training runs",
        labelnames=("status",),
        registry=registry,
    ).labels(status="success").inc()

    # Prediction metrics
    Gauge("bike_prediction_rmse", "Mock prediction RMSE", registry=registry).set(52.3)
    Gauge("bike_prediction_mae", "Mock prediction MAE", registry=registry).set(38.7)
    Gauge(
        "bike_predictions_generated_total", "Mock predictions", registry=registry
    ).set(450)

    # Double eval metrics
    Gauge(
        "bike_model_r2_champion_baseline",
        "Mock champion R² baseline",
        registry=registry,
    ).set(0.87)
    Gauge(
        "bike_model_r2_challenger_current",
        "Mock challenger R² current",
        registry=registry,
    ).set(0.89)
    Gauge(
        "bike_model_r2_challenger_baseline",
        "Mock challenger R² baseline",
        registry=registry,
    ).set(0.88)

    push_metrics(registry)
    print("   Injected: 11 dashboard-only metrics")
    print("   Expected: Visual updates in Grafana dashboards (no alerts)")


def restore_normal():
    """Restore all metrics to normal values"""
    print("\n🔄 Restoring normal values...")
    registry = CollectorRegistry()

    # Normal values
    Gauge("bike_drift_share", "Drift share", registry=registry).set(0.15)
    Gauge("bike_model_r2_champion_current", "R² score", registry=registry).set(0.85)
    Gauge("bike_model_rmse_production", "RMSE", registry=registry).set(45.0)
    Gauge("fastapi_requests_total", "Total requests", registry=registry).set(1000)
    Gauge("fastapi_errors_total", "Total errors", registry=registry).set(10)
    Gauge("bike_records_ingested_total", "Ingested records", registry=registry).set(500)
    Gauge("up", "Service status", registry=registry).set(1)

    push_metrics(registry)
    print("✅ All metrics restored to normal values")


def main():
    parser = argparse.ArgumentParser(
        description="Test Grafana alerts by injecting mock Prometheus metrics"
    )
    parser.add_argument(
        "--test",
        choices=[
            "all",
            "drift",
            "r2",
            "rmse",
            "combined",
            "api",
            "ingestion",
            "service",
            "dag",
            "dashboards",
            "restore",
        ],
        default="all",
        help="Which test to run (default: all)",
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=2,
        help="Seconds to wait between tests (default: 2)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("🧪 Grafana Alert Testing - Mock Metrics Injection")
    print("=" * 60)
    print(f"Pushgateway: {PUSHGATEWAY_URL}")
    print("Grafana: http://localhost:3000/alerting/list")
    print()

    tests = {
        "drift": test_high_drift,
        "r2": test_low_r2,
        "rmse": test_high_rmse,
        "combined": test_critical_drift_and_r2,
        "api": test_api_error_rate,
        "ingestion": test_no_data_ingestion,
        "service": test_service_down,
        "dag": test_dag_too_slow,
        "dashboards": test_dashboards,
        "restore": restore_normal,
    }

    if args.test == "all":
        for test_func in tests.values():
            test_func()
            time.sleep(args.wait)
    else:
        tests[args.test]()

    print("\n" + "=" * 60)
    print("✅ Mock metrics injected into Prometheus")
    print()
    print("Next steps:")
    print("1. Check Prometheus targets: http://localhost:9090/targets")
    print("2. Query metrics: http://localhost:9090/graph")
    print("3. View Grafana alerts: http://localhost:3000/alerting/list")
    print("4. Check Discord for notifications")
    print()
    print(
        "To restore normal values: python scripts/test_grafana_alerts.py --test restore"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
