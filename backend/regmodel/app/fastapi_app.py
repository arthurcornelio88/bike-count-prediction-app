import os
import shutil
import hashlib
import tempfile
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd

from app.model_registry_summary import get_best_model_from_summary

app = FastAPI()


def setup_credentials():
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and os.path.exists(
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    ):
        print("✅ Credentials déjà présents (local / montés)")
        return

    key_json = os.getenv("GCP_JSON_CONTENT")
    if not key_json:
        raise EnvironmentError(
            "❌ GCP credentials non trouvés dans GCP_JSON_CONTENT ni via GOOGLE_APPLICATION_CREDENTIALS"
        )

    cred_path = os.path.join(tempfile.gettempdir(), "gcp_creds.json")
    with open(cred_path, "w") as f:  # noqa: PTH123
        f.write(key_json)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path
    print("✅ Credentials injectés via GCP_JSON_CONTENT")


# === Cache global des modèles ===
model_cache = {}


# === Fonction utilitaire de cache avec nettoyage ===
def get_cache_dir(model_type: str, metric: str) -> str:
    """Génère un chemin unique et réutilisable dans temp directory"""
    key = f"{model_type}_{metric}"
    key_hash = hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()  # noqa: S324
    path = os.path.join(tempfile.gettempdir(), f"model_cache_{key_hash}")
    return path


def get_cached_model(model_type: str, metric: str):
    key = (model_type, metric)
    if key not in model_cache:
        cache_dir = get_cache_dir(model_type, metric)

        # Si le dossier existe déjà, on le supprime pour repartir propre
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)

        os.makedirs(cache_dir, exist_ok=True)

        model = get_best_model_from_summary(
            model_type=model_type,
            metric=metric,
            summary_path="gs://df_traffic_cyclist1/models/summary.json",
            env="prod",
            download_dir=cache_dir,  # ← Important pour contrôler le chemin
        )
        model_cache[key] = model
    return model_cache[key]


# === Chargement anticipé au démarrage ===
@app.on_event("startup")
def preload_models():
    setup_credentials()
    print("🚀 Préchargement des modèles...")
    for model_type, metric in [("rf", "r2"), ("nn", "r2")]:
        try:
            get_cached_model(model_type, metric)
            print(f"✅ Modèle {model_type} ({metric}) préchargé.")
        except Exception as e:
            print(f"⚠️ Erreur de chargement pour {model_type} ({metric}) : {e}")


# === Schéma de requête ===
class PredictRequest(BaseModel):
    records: List[dict]
    model_type: str
    metric: str = "r2"


# === Endpoint de prédiction ===
@app.post("/predict")
def predict(data: PredictRequest):
    df = pd.DataFrame(data.records)

    try:
        model = get_cached_model(data.model_type, data.metric)
        y_pred = model.predict_clean(df)
        return {"predictions": y_pred.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Schéma de requête pour training ===
class TrainRequest(BaseModel):
    model_type: str  # "rf", "nn", or "rf_class"
    data_source: str = "reference"  # "reference", "current", or "baseline"
    env: str = "prod"
    hyperparams: dict = {}  # Optional hyperparameters
    test_mode: bool = False  # Use small sample (1000 rows) for fast testing
    current_data: List[dict] = None  # NEW: Optional current data for double evaluation


# === Endpoint d'entraînement ===
@app.post("/train")
def train_endpoint(request: TrainRequest):
    """
    Train a model and upload to GCS + summary.json.

    NEW: Supports double test set evaluation when current_data is provided.
    - Splits current_data (80/20)
    - Evaluates on test_baseline (fixed reference) + test_current (new distribution)
    - Returns both metrics for deployment decision

    Example request (basic):
    {
        "model_type": "rf",
        "data_source": "baseline",
        "env": "prod",
        "hyperparams": {},
        "test_mode": false
    }

    Example request (with double evaluation):
    {
        "model_type": "rf",
        "data_source": "baseline",
        "env": "prod",
        "current_data": [{"col1": val1, ...}, ...],  # >= 200 samples
        "test_mode": false
    }
    """
    try:
        from app.train import train_model

        # Convert current_data to DataFrame if provided
        current_df = None
        if request.current_data:
            current_df = pd.DataFrame(request.current_data)
            print(
                f"📥 Received {len(current_df)} current data samples for double evaluation"
            )

        # Call training function
        result = train_model(
            model_type=request.model_type,
            data_source=request.data_source,
            env=request.env,
            hyperparams=request.hyperparams,
            test_mode=request.test_mode,
            current_data_df=current_df,  # NEW: Pass current data for double eval
        )

        # Build response with double evaluation results
        response = {
            "status": "success",
            "model_type": request.model_type,
            "run_id": result.get("run_id"),
            "metrics": result.get("metrics"),
            "model_uri": result.get("model_uri"),
        }

        # Add double evaluation results if available
        if result.get("metrics_baseline"):
            response.update(
                {
                    "metrics_baseline": result.get("metrics_baseline"),
                    "metrics_current": result.get("metrics_current"),
                    "baseline_regression": result.get("baseline_regression", False),
                    "double_evaluation_enabled": True,
                }
            )
        else:
            response["double_evaluation_enabled"] = False

        return response

    except Exception as e:
        import traceback

        print(f"❌ Training failed: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


# === Schéma de requête pour evaluation ===
class EvaluateRequest(BaseModel):
    model_type: str = "rf"  # "rf", "nn", or "rf_class"
    metric: str = "r2"  # Metric to select champion model
    test_baseline_path: str = "data/test_baseline.csv"  # Local path in container
    model_uri: str = (
        None  # Optional: specific model URI to evaluate (otherwise uses champion)
    )


# === Endpoint d'évaluation ===
@app.post("/evaluate")
def evaluate_endpoint(request: EvaluateRequest):
    """
    Evaluate a model (champion or specific model_uri) on test_baseline.

    This endpoint is used to compare the current champion's performance
    on the fixed test_baseline against new candidate models.

    Example request (evaluate current champion):
    {
        "model_type": "rf",
        "metric": "r2",
        "test_baseline_path": "gs://df_traffic_cyclist1/raw_data/test_baseline.csv"
    }

    Example request (evaluate specific model):
    {
        "model_type": "rf",
        "model_uri": "gs://df_traffic_cyclist1/mlflow-artifacts/1/abc123/artifacts/model",
        "test_baseline_path": "gs://df_traffic_cyclist1/raw_data/test_baseline.csv"
    }

    Returns:
    {
        "status": "success",
        "model_type": "rf",
        "model_uri": "gs://...",
        "metrics": {
            "rmse": 123.45,
            "r2": 0.75,
            "mae": 98.76
        },
        "test_size": 181202
    }
    """
    try:
        from app.train import load_and_clean_data
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        import numpy as np

        # Load test_baseline
        print(f"📥 Loading test_baseline from {request.test_baseline_path}")
        X_baseline, y_baseline = load_and_clean_data(request.test_baseline_path)
        test_size = len(y_baseline)
        print(f"✅ Test baseline loaded: {test_size} samples")

        # Load model (champion or specific URI)
        if request.model_uri:
            print(f"📦 Loading model from URI: {request.model_uri}")
            # TODO: Implement MLflow model loading by URI
            # For now, use champion as fallback
            print("⚠️  model_uri not yet implemented, using champion")
            model = get_cached_model(request.model_type, request.metric)
            model_uri = "champion"  # Placeholder
        else:
            print(
                f"🏆 Loading champion {request.model_type} model (metric={request.metric})"
            )
            model = get_cached_model(request.model_type, request.metric)
            model_uri = "champion"

        # Evaluate model
        print("🔬 Evaluating model on test_baseline...")
        y_pred = model.predict(X_baseline)

        # Handle NN predictions (flatten if needed)
        if hasattr(y_pred, "flatten"):
            y_pred = y_pred.flatten()

        # Calculate metrics
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_baseline, y_pred))),
            "r2": float(r2_score(y_baseline, y_pred)),
            "mae": float(mean_absolute_error(y_baseline, y_pred)),
        }

        print("✅ Evaluation complete:")
        print(f"   - RMSE: {metrics['rmse']:.2f}")
        print(f"   - R²: {metrics['r2']:.4f}")
        print(f"   - MAE: {metrics['mae']:.2f}")

        return {
            "status": "success",
            "model_type": request.model_type,
            "model_uri": model_uri,
            "metrics": metrics,
            "test_size": test_size,
        }

    except Exception as e:
        import traceback

        print(f"❌ Evaluation failed: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


# === Schéma de requête pour monitoring ===
class MonitorRequest(BaseModel):
    reference_path: str  # GCS path to reference sample
    current_data: List[dict]  # Current data as list of records
    output_html: str = "drift_report.html"  # Output HTML report name


# === Endpoint de monitoring avec Evidently ===
@app.post("/monitor")
def monitor_endpoint(request: MonitorRequest):
    """
    Detect data drift using Evidently.

    Compares reference data (sampled, from GCS) with current data.
    Returns drift detection results and uploads HTML report to GCS.

    Example request:
    {
        "reference_path": "gs://df_traffic_cyclist1/data/reference_data_sample.csv",
        "current_data": [{"col1": val1, "col2": val2}, ...],
        "output_html": "drift_report_20251029.html"
    }

    TODO [Phase 4 - Prometheus]: Add Prometheus metrics
          - prometheus_client.Gauge('evidently_drift_detected')
          - prometheus_client.Gauge('evidently_drift_share')
          - prometheus_client.Counter('evidently_drift_checks_total')
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        from google.cloud import storage

        # Load reference data from GCS
        print(f"📥 Loading reference data from {request.reference_path}")

        if request.reference_path.startswith("gs://"):
            path_parts = request.reference_path.replace("gs://", "").split("/", 1)
            bucket_name = path_parts[0]
            blob_path = path_parts[1]

            project_id = os.getenv("BQ_PROJECT", "datascientest-460618")
            client = storage.Client(project=project_id)
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)

            temp_ref = os.path.join(tempfile.gettempdir(), "reference_data_sample.csv")
            blob.download_to_filename(temp_ref)
            df_reference = pd.read_csv(temp_ref, sep=";")
        else:
            df_reference = pd.read_csv(request.reference_path, sep=";")

        print(f"✅ Loaded {len(df_reference)} reference records")
        print(f"Reference columns: {list(df_reference.columns)}")

        # Convert current data to DataFrame
        df_current = pd.DataFrame(request.current_data)
        print(f"✅ Loaded {len(df_current)} current records")
        print(f"Current columns: {list(df_current.columns)}")

        # Ensure both DataFrames have the same columns
        common_cols = list(set(df_reference.columns) & set(df_current.columns))
        print(f"Common columns: {common_cols}")
        if not common_cols:
            print("⚠️ No common columns found - treating as schema drift")
            # If no common columns, consider it as drift (schema change)
            return {
                "status": "success",
                "drift_summary": {
                    "drift_detected": True,
                    "drift_share": 1.0,  # Full drift due to schema change
                    "drifted_features": [],
                    "total_features": 0,
                    "reference_size": len(df_reference),
                    "current_size": len(df_current),
                    "schema_drift": True,
                },
                "report_url": None,  # No report generated
            }

        # Filter to relevant columns (exclude metadata but keep meaningful features)
        # Strategy: Keep features that actually affect model predictions
        # Exclude: technical IDs, URLs, photos, but KEEP identifiant_du_compteur (important for drift!)
        exclude_patterns = [
            "lien_",
            "url_",
            "photo",
            "ingestion_ts",
            "prediction_ts",
            "prediction",
            "id_photos",
            "id_photo",
            "identifiant_du_site",  # Site ID not used in model
            "identifiant_technique",  # Technical ID not used
            "test_lien",
        ]

        # Keep these important columns even if they match partial patterns
        always_keep = [
            "identifiant_du_compteur",  # Counter ID - critical for drift detection
            "nom_du_compteur",  # Counter name - used in model
            "comptage_horaire",  # Target variable
            "latitude",  # GPS coordinates - used in model
            "longitude",
        ]

        relevant_cols = []
        for col in common_cols:
            # Always keep if in whitelist
            if col in always_keep:
                relevant_cols.append(col)
            # Otherwise check if matches exclude patterns
            elif not any(pattern in col.lower() for pattern in exclude_patterns):
                relevant_cols.append(col)

        if not relevant_cols:
            print("⚠️ No relevant columns for drift, using all common columns")
            relevant_cols = common_cols

        df_reference = df_reference[relevant_cols]
        df_current = df_current[relevant_cols]

        print(f"📊 Analyzing {len(relevant_cols)} features for drift")

        # Run Evidently drift detection
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=df_reference, current_data=df_current)

        # Save HTML report locally first
        local_report_path = os.path.join(tempfile.gettempdir(), request.output_html)
        report.save_html(local_report_path)
        print(f"📄 HTML report generated: {request.output_html}")

        # Upload report to GCS
        gcs_report_path = f"drift_reports/{request.output_html}"
        gcs_bucket_name = os.getenv("GCS_BUCKET", "df_traffic_cyclist1")

        project_id = os.getenv("BQ_PROJECT", "datascientest-460618")
        gcs_client = storage.Client(project=project_id)
        gcs_bucket = gcs_client.bucket(gcs_bucket_name)
        report_blob = gcs_bucket.blob(gcs_report_path)
        report_blob.upload_from_filename(local_report_path)

        gcs_url = f"gs://{gcs_bucket_name}/{gcs_report_path}"
        print(f"☁️ Report uploaded to {gcs_url}")

        # Extract drift summary
        report_dict = report.as_dict()
        metrics = report_dict.get("metrics", [])

        drift_detected = False
        drift_share = 0.0
        drifted_features = []

        for metric in metrics:
            if metric.get("metric") == "DatasetDriftMetric":
                result = metric.get("result", {})
                drift_detected = result.get("dataset_drift", False)
                drift_share = result.get("drift_share", 0.0)
                drifted_features = [
                    k for k, v in result.get("drift_by_columns", {}).items() if v
                ]
                break

        print(f"{'🚨 Drift detected!' if drift_detected else '✅ No drift detected'}")
        print(f"   - Drift share: {drift_share:.2%}")
        if drifted_features:
            print(f"   - Drifted features: {', '.join(drifted_features[:10])}")

        # TODO: Push metrics to Prometheus (Phase 4)
        # prometheus_client.Gauge('evidently_drift_detected').set(1 if drift_detected else 0)
        # prometheus_client.Gauge('evidently_drift_share').set(drift_share)

        return {
            "status": "success",
            "drift_summary": {
                "drift_detected": drift_detected,
                "drift_share": drift_share,
                "drifted_features": drifted_features,
                "total_features": len(relevant_cols),
                "reference_size": len(df_reference),
                "current_size": len(df_current),
            },
            "report_url": gcs_url,
        }

    except Exception as e:
        import traceback

        print(f"❌ Drift detection failed: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Drift detection failed: {str(e)}")
