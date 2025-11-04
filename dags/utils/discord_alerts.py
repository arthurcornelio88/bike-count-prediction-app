"""
Discord Alerting Module for MLOps Pipeline
Sends notifications to Discord webhook for important events
"""

import os
from datetime import datetime
from typing import Dict, Any, Optional
import requests


# Discord webhook URL from environment variable
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")


def send_discord_message(
    title: str,
    description: str,
    color: int,
    fields: Optional[list[Dict[str, Any]]] = None,
) -> bool:
    """
    Send a formatted message to Discord via webhook.

    Args:
        title: Message title
        description: Message description
        color: Embed color (decimal, e.g., 0xFF0000 for red)
        fields: Optional list of {name, value, inline} dicts

    Returns:
        bool: True if sent successfully, False otherwise
    """
    if not DISCORD_WEBHOOK_URL:
        print("⚠️  Discord webhook URL not configured. Skipping notification.")
        return False

    embed = {
        "title": title,
        "description": description,
        "color": color,
        "timestamp": datetime.utcnow().isoformat(),
        "footer": {"text": "MLOps Bike Traffic Monitoring"},
    }

    if fields:
        embed["fields"] = fields

    payload = {"embeds": [embed]}

    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        response.raise_for_status()
        print(f"✅ Discord notification sent: {title}")
        return True
    except Exception as e:
        print(f"❌ Failed to send Discord notification: {e}")
        return False


# ============================================================================
# Drift & Performance Alerts
# ============================================================================


def send_drift_alert(drift_share: float, r2: float, drifted_features: int) -> bool:
    """
    Alert when high data drift is detected (WARNING level).

    Args:
        drift_share: Drift percentage (0.0-1.0)
        r2: Current production R² score
        drifted_features: Number of features with drift

    Returns:
        bool: True if sent successfully
    """
    drift_pct = drift_share * 100

    description = (
        "⚠️  WARNING: High data drift detected in production model.\n\n"
        "**Impact**: Model predictions may degrade over time. Monitor closely and plan retraining."
    )

    fields = [
        {"name": "Drift Share", "value": f"{drift_pct:.1f}%", "inline": True},
        {"name": "Production R²", "value": f"{r2:.4f}", "inline": True},
        {"name": "Drifted Features", "value": str(drifted_features), "inline": True},
    ]

    return send_discord_message(
        title="Data Drift Detected",
        description=description,
        color=0xFFA500,  # Orange (warning)
        fields=fields,
    )


def send_performance_alert(r2: float, rmse: float, threshold: float = 0.70) -> bool:
    """
    Alert when model performance drops below threshold.

    Args:
        r2: Current R² score
        rmse: Current RMSE
        threshold: R² threshold (default 0.70)

    Returns:
        bool: True if sent successfully
    """
    if r2 < 0.65:
        severity = "🚨 CRITICAL"
        impact = "Predictions are unreliable - immediate retraining required."
        color = 0xFF0000  # Red
    else:
        severity = "⚠️  WARNING"
        impact = "Predictions may be less reliable - retraining recommended soon."
        color = 0xFFA500  # Orange

    description = f"{severity}: Production model performance below threshold.\n\n**Impact**: {impact}"

    fields = [
        {"name": "R² Score", "value": f"{r2:.4f}", "inline": True},
        {"name": "RMSE", "value": f"{rmse:.2f}", "inline": True},
        {"name": "Threshold", "value": f"{threshold:.2f}", "inline": True},
    ]

    return send_discord_message(
        title="Model Performance Degraded",
        description=description,
        color=color,
        fields=fields,
    )


# ============================================================================
# Training & Deployment Alerts
# ============================================================================


def send_training_success(
    improvement_delta: float,
    new_r2: float,
    old_r2: float,
    deployment_decision: str,
) -> bool:
    """
    Notify when model training succeeds (INFO level).

    Args:
        improvement_delta: R² improvement (new - old)
        new_r2: New model R²
        old_r2: Old model R²
        deployment_decision: "deploy", "skip", or "reject"

    Returns:
        bool: True if sent successfully
    """
    if deployment_decision == "deploy":
        emoji = "🚀"
        decision_text = "New model DEPLOYED to production"
        color = 0x00FF00  # Green
    elif deployment_decision == "skip":
        emoji = "⏭️ "
        decision_text = "Training skipped (no improvement needed)"
        color = 0x808080  # Gray
    else:  # reject
        emoji = "❌"
        decision_text = "New model REJECTED (worse than champion)"
        color = 0xFFA500  # Orange

    description = f"{emoji} Model training completed.\n\n**Decision**: {decision_text}"

    fields = [
        {"name": "New R²", "value": f"{new_r2:.4f}", "inline": True},
        {"name": "Old R²", "value": f"{old_r2:.4f}", "inline": True},
        {"name": "Improvement", "value": f"+{improvement_delta:.4f}", "inline": True},
    ]

    return send_discord_message(
        title="Training Completed",
        description=description,
        color=color,
        fields=fields,
    )


def send_training_failure(error_message: str, dag_run_id: str) -> bool:
    """
    Alert when model training fails (CRITICAL level).

    Args:
        error_message: Error description
        dag_run_id: Airflow DAG run ID for debugging

    Returns:
        bool: True if sent successfully
    """
    description = (
        "🚨 CRITICAL: Model training pipeline failed.\n\n"
        f"**Error**: {error_message[:200]}\n\n"
        "**Impact**: No new model available for deployment."
    )

    fields = [
        {"name": "DAG Run ID", "value": dag_run_id, "inline": False},
        {
            "name": "Action",
            "value": "Check Airflow logs for details",
            "inline": False,
        },
    ]

    return send_discord_message(
        title="Training Pipeline Failed",
        description=description,
        color=0xFF0000,  # Red
        fields=fields,
    )


# ============================================================================
# Infrastructure Alerts (CRITICAL)
# ============================================================================


def send_service_down_alert(service_name: str, duration_minutes: int) -> bool:
    """
    Alert when a critical service is down (CRITICAL level).

    Args:
        service_name: Name of the service
        duration_minutes: How long it's been down

    Returns:
        bool: True if sent successfully
    """
    description = (
        f"🚨 CRITICAL: Service **{service_name}** is DOWN.\n\n"
        f"**Duration**: {duration_minutes} minutes\n"
        "**Impact**: MLOps pipeline may be broken."
    )

    fields = [
        {
            "name": "Action",
            "value": f"`docker compose ps && docker compose logs {service_name}`",
            "inline": False,
        }
    ]

    return send_discord_message(
        title=f"Service Down: {service_name}",
        description=description,
        color=0xFF0000,  # Red
        fields=fields,
    )


def send_api_error_alert(error_rate: float, endpoint: str) -> bool:
    """
    Alert when API error rate is high (CRITICAL level).

    Args:
        error_rate: Error rate (0.0-1.0)
        endpoint: API endpoint with errors

    Returns:
        bool: True if sent successfully
    """
    error_pct = error_rate * 100

    description = (
        f"🚨 CRITICAL: High error rate on **{endpoint}**.\n\n"
        f"**Error Rate**: {error_pct:.1f}%\n"
        "**Impact**: Users experiencing failures."
    )

    fields = [
        {
            "name": "Action",
            "value": "`docker compose logs regmodel-backend | tail -100`",
            "inline": False,
        }
    ]

    return send_discord_message(
        title="High API Error Rate",
        description=description,
        color=0xFF0000,  # Red
        fields=fields,
    )


# ============================================================================
# Info Notifications
# ============================================================================


def send_dag_completion_notification(
    dag_id: str, duration_seconds: int, status: str
) -> bool:
    """
    Notify when an important DAG completes (INFO level).

    Args:
        dag_id: DAG identifier
        duration_seconds: Execution duration
        status: "success" or "failed"

    Returns:
        bool: True if sent successfully
    """
    emoji = "✅" if status == "success" else "❌"
    color = 0x00FF00 if status == "success" else 0xFF0000

    duration_min = duration_seconds / 60

    description = f"{emoji} DAG **{dag_id}** completed with status: **{status}**"

    fields = [
        {"name": "Duration", "value": f"{duration_min:.1f} minutes", "inline": True},
        {"name": "Status", "value": status, "inline": True},
    ]

    return send_discord_message(
        title=f"DAG {status.capitalize()}: {dag_id}",
        description=description,
        color=color,
        fields=fields,
    )
