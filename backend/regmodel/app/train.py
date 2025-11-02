import pandas as pd
import argparse
import subprocess
import os

# CRITICAL: Set GCS credentials BEFORE importing mlflow
# MLflow client needs this to upload artifacts to GCS
if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") is None:
    # Priority 1: mlflow-trainer.json (has bucket write permissions)
    gcp_credentials_path = "./mlflow-trainer.json"
    if os.path.exists(gcp_credentials_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_credentials_path
        print(f"🔐 Using GCS credentials: {gcp_credentials_path}")
    else:
        # Fallback: gcp.json (may have limited permissions)
        gcp_credentials_path = "./gcp.json"
        if os.path.exists(gcp_credentials_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_credentials_path
            print(f"🔐 Using GCS credentials: {gcp_credentials_path}")
        else:
            print(
                "⚠️  WARNING: No GCS credentials found (mlflow-trainer.json or gcp.json)"
            )

import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import shutil
import numpy as np
from app.classes import RFPipeline, NNPipeline, AffluenceClassifierPipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score
from app.model_registry_summary import update_summary
from datetime import datetime


def setup_environment(env: str, model_test: bool):
    """
    Setup MLflow tracking and experiment.
    Note: GCS credentials already set at module import.
    """
    if env == "dev":
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment(
            "bike-traffic-training"
        )  # New experiment with GCS artifacts

        # Verify experiment configuration
        experiment = mlflow.get_experiment_by_name("bike-traffic-training")
        print(f"📊 Experiment: {experiment.name} (ID: {experiment.experiment_id})")
        print(f"📦 Artifact location: {experiment.artifact_location}")

        data_path = (
            "data/comptage-velo-donnees-compteurs_test.csv"
            if model_test
            else "data/comptage-velo-donnees-compteurs.csv"
        )
        artifact_path = "models/"
        os.makedirs(artifact_path, exist_ok=True)
    elif env == "prod":
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment(
            "bike-traffic-training"
        )  # New experiment with GCS artifacts
        data_path = (
            "gs://df_traffic_cyclist1/raw_data/comptage-velo-donnees-compteurs_test.csv"
            if model_test
            else "gs://df_traffic_cyclist1/raw_data/comptage-velo-donnees-compteurs.csv"
        )
        artifact_path = "models/"
        os.makedirs(artifact_path, exist_ok=True)
    else:
        raise ValueError("Environnement invalide")

    return data_path, artifact_path


def load_and_clean_data(path: str, preserve_target=False):
    df = pd.read_csv(path, sep=";")
    df[["latitude", "longitude"]] = (
        df["Coordonnées géographiques"].str.split(",", expand=True).astype(float)
    )
    df_clean = df.dropna(subset=["latitude", "longitude"])

    if preserve_target:
        return df_clean

    X = df_clean.drop(columns="Comptage horaire")
    y = df_clean["Comptage horaire"]
    return X, y


def get_test_baseline_path(env: str) -> str:
    """
    Get environment-specific path to test_baseline.csv

    Args:
        env: "dev" or "prod"

    Returns:
        Path to test_baseline.csv (local for dev, GCS for prod)
    """
    if env == "dev":
        # Local path for development (avoid downloading 300MB)
        return "data/test_baseline.csv"
    else:
        # GCS path for production (from env var)
        gcs_bucket = os.getenv("GCS_BUCKET", "df_traffic_cyclist1")
        return f"gs://{gcs_bucket}/raw_data/test_baseline.csv"


def evaluate_double(
    model, test_baseline_path: str, current_data_df: pd.DataFrame, model_type: str
):
    """
    Evaluate model on both test sets (baseline + current).

    This implements the double test set evaluation strategy:
    - test_baseline: Fixed reference test set (evaluates regression)
    - test_current: 20% of current data (evaluates performance on new distribution)

    Args:
        model: Trained model (RFPipeline, NNPipeline, or AffluenceClassifierPipeline)
        test_baseline_path: Path to test_baseline.csv (local or GCS)
        current_data_df: Current data DataFrame (will be split 80/20)
        model_type: "rf", "nn", or "rf_class"

    Returns:
        dict with:
            - metrics_baseline: {"rmse", "r2", "mae"}
            - metrics_current: {"rmse", "r2", "mae"}
            - baseline_regression: bool (True if R² < 0.60)
            - test_current_size: int
            - test_baseline_size: int
            - train_current_data: DataFrame (80% of current_data for augmented training)

    Raises:
        ValueError: If current_data_df has < 200 samples (insufficient for 80/20 split)
    """
    from sklearn.metrics import mean_absolute_error
    import tempfile

    print(f"\n{'='*60}")
    print(f"🔬 DOUBLE EVALUATION STRATEGY - {model_type.upper()}")
    print(f"{'='*60}")

    # 1. Load test_baseline from GCS or local
    print(f"📥 Loading test_baseline from {test_baseline_path}")
    X_baseline, y_baseline = load_and_clean_data(test_baseline_path)
    print(f"✅ Baseline test set loaded: {len(y_baseline)} samples")

    # 2. Split current data (80/20) - use 20% for testing
    if len(current_data_df) < 200:
        raise ValueError(
            f"Insufficient current data: {len(current_data_df)} samples < 200 minimum. "
            "Need at least 200 samples for reliable 80/20 split."
        )

    print(f"\n📊 Splitting current data (n={len(current_data_df)})...")
    test_current = current_data_df.sample(frac=0.2, random_state=42)
    train_current = current_data_df.drop(test_current.index)
    print(f"   - Train portion (80%): {len(train_current)} samples")
    print(f"   - Test portion (20%): {len(test_current)} samples")

    # Prepare test_current for evaluation
    # Save to temp file for load_and_clean_data to work
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        test_current.to_csv(f.name, index=False, sep=";")
        temp_path = f.name

    X_current, y_current = load_and_clean_data(temp_path)
    os.remove(temp_path)

    # 3. Evaluate on baseline (check for regression)
    print("\n📍 Evaluating on test_baseline...")
    y_pred_baseline = model.predict(X_baseline)

    # Handle NN predictions (flatten if needed)
    if hasattr(y_pred_baseline, "flatten"):
        y_pred_baseline = y_pred_baseline.flatten()

    metrics_baseline = {
        "rmse": float(np.sqrt(mean_squared_error(y_baseline, y_pred_baseline))),
        "r2": float(r2_score(y_baseline, y_pred_baseline)),
        "mae": float(mean_absolute_error(y_baseline, y_pred_baseline)),
    }

    # 4. Evaluate on current (check for improvement)
    print("🆕 Evaluating on test_current...")
    y_pred_current = model.predict(X_current)

    # Handle NN predictions (flatten if needed)
    if hasattr(y_pred_current, "flatten"):
        y_pred_current = y_pred_current.flatten()

    metrics_current = {
        "rmse": float(np.sqrt(mean_squared_error(y_current, y_pred_current))),
        "r2": float(r2_score(y_current, y_pred_current)),
        "mae": float(mean_absolute_error(y_current, y_pred_current)),
    }

    # 5. Check for baseline regression
    BASELINE_R2_THRESHOLD = 0.60  # Minimum acceptable R² on baseline
    baseline_regression = metrics_baseline["r2"] < BASELINE_R2_THRESHOLD

    # Print summary
    print(f"\n{'='*60}")
    print("📊 DOUBLE EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"📍 Test Baseline (fixed reference, n={len(y_baseline)}):")
    print(f"   - RMSE: {metrics_baseline['rmse']:.2f}")
    print(f"   - MAE:  {metrics_baseline['mae']:.2f}")
    print(f"   - R²:   {metrics_baseline['r2']:.4f}")
    print(f"\n🆕 Test Current (new distribution, n={len(y_current)}):")
    print(f"   - RMSE: {metrics_current['rmse']:.2f}")
    print(f"   - MAE:  {metrics_current['mae']:.2f}")
    print(f"   - R²:   {metrics_current['r2']:.4f}")
    print(
        f"\n{'🚨 REGRESSION DETECTED' if baseline_regression else '✅ No regression on baseline'}"
    )
    print(f"   Threshold: R² >= {BASELINE_R2_THRESHOLD}")
    print(f"{'='*60}\n")

    return {
        "metrics_baseline": metrics_baseline,
        "metrics_current": metrics_current,
        "baseline_regression": baseline_regression,
        "test_current_size": len(y_current),
        "test_baseline_size": len(y_baseline),
        "train_current_data": train_current,  # Return 80% for augmented training
    }


def update_model_summary(
    model_type: str,
    run_id: str,
    rmse: float = None,
    r2: float = None,
    env: str = "dev",
    test_mode: bool = False,
    model_uri: str = None,
    accuracy: float = None,
    precision: float = None,
    recall: float = None,
    f1_score: float = None,
):
    """
    Update summary.json in GCS with model metadata.
    This is the registry used by Airflow to select models for promotion.
    Updated for BOTH dev and prod environments.
    """
    # MLflow artifact URI (where the actual model files are)
    if model_uri is None:
        # Do not fabricate a gs:// link when we don't have a real artifact URI.
        # Prefer an explicit 'not found' marker so callers know the artifact wasn't available.
        mlflow_artifact_uri = "not found"
    else:
        mlflow_artifact_uri = model_uri

    # Always update summary.json (dev champion training or prod fine-tuning)
    update_summary(
        summary_path="gs://df_traffic_cyclist1/models/summary.json",
        model_type=model_type,
        run_id=run_id,
        rmse=rmse,
        r2=r2,
        model_uri=mlflow_artifact_uri,
        env=env,
        test_mode=test_mode,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
    )
    print(
        f"📝 summary.json mis à jour pour {model_type} [env={env}] (MLflow URI: {mlflow_artifact_uri})"
    )


def train_rf(X, y, env, test_mode, data_source="reference", current_data_df=None):
    """
    Train RandomForest model with optional double evaluation.

    Args:
        X: Training features
        y: Training target
        env: "dev" or "prod"
        test_mode: bool
        data_source: "reference", "current", or "baseline"
        current_data_df: Optional DataFrame for double evaluation (requires >= 200 samples)

    Returns:
        dict with metrics and optionally double evaluation results
    """
    y = y.to_numpy()
    run_name = f"RandomForest_Train_{env}" + ("_TEST" if test_mode else "")
    print("📡 Tracking URI :", mlflow.get_tracking_uri())

    with mlflow.start_run(run_name=run_name):
        run = mlflow.active_run()
        mlflow.set_tag("mode", env)
        mlflow.set_tag("test_mode", test_mode)
        mlflow.set_tag("dataset", "full" if data_source == "baseline" else "partial")
        mlflow.set_tag(
            "training_type", "champion" if data_source == "baseline" else "legacy"
        )
        mlflow.set_tag("data_source", data_source)
        mlflow.log_metric("test_mode", int(test_mode))
        mlflow.sklearn.autolog(disable=True)

        # Tag if double evaluation enabled
        if current_data_df is not None:
            mlflow.set_tag("double_evaluation", True)
            mlflow.log_metric("current_data_size", len(current_data_df))

        rf = RFPipeline()
        rf.fit(X, y)

        mlflow.log_param("rf_n_estimators", rf.model.n_estimators)
        mlflow.log_param("rf_max_depth", rf.model.max_depth)

        y_pred = rf.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        mlflow.log_metric("rf_rmse_train", rmse)
        mlflow.log_metric("rf_r2_train", r2)

        print(f"🎯 RF – RMSE : {rmse:.2f} | R² : {r2:.4f}")

        # NEW: Double evaluation if current_data provided
        double_eval_results = None
        if current_data_df is not None and len(current_data_df) >= 200:
            try:
                test_baseline_path = get_test_baseline_path(env)
                double_eval_results = evaluate_double(
                    model=rf,
                    test_baseline_path=test_baseline_path,
                    current_data_df=current_data_df,
                    model_type="rf",
                )

                # Log double metrics to MLflow
                mlflow.log_metrics(
                    {
                        "baseline_rmse": double_eval_results["metrics_baseline"][
                            "rmse"
                        ],
                        "baseline_r2": double_eval_results["metrics_baseline"]["r2"],
                        "baseline_mae": double_eval_results["metrics_baseline"]["mae"],
                        "current_rmse": double_eval_results["metrics_current"]["rmse"],
                        "current_r2": double_eval_results["metrics_current"]["r2"],
                        "current_mae": double_eval_results["metrics_current"]["mae"],
                        "baseline_regression": int(
                            double_eval_results["baseline_regression"]
                        ),
                        "test_baseline_size": double_eval_results["test_baseline_size"],
                        "test_current_size": double_eval_results["test_current_size"],
                    }
                )

                print("✅ Double evaluation metrics logged to MLflow")

            except Exception as e:
                print(f"⚠️ Double evaluation failed: {e}")
                import traceback

                print(traceback.format_exc())
                # Continue without double eval - don't fail the training
        elif current_data_df is not None and len(current_data_df) < 200:
            print(
                f"⚠️ Current data too small for double evaluation: {len(current_data_df)} < 200 samples"
            )

        # Save model to temporary directory for MLflow upload
        import tempfile
        import shutil

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = os.path.join(tmp_dir, "model")
            os.makedirs(model_dir, exist_ok=True)
            rf.save(model_dir)

            # Log complete model pipeline to MLflow (uploads to GCS artifact root)
            mlflow.log_artifacts(model_dir, artifact_path="model")

            # Get exact artifact URI from MLflow (ensures correct gs:// path)
            artifact_uri = mlflow.get_artifact_uri("model")

        # Register model if training on baseline data
        if data_source == "baseline":
            model_uri = f"runs:/{run.info.run_id}/model"
            mlflow.register_model(model_uri, "bike-traffic-rf")

        # Update summary.json in GCS
        update_model_summary(
            model_type="rf",
            run_id=run.info.run_id,
            rmse=rmse,
            r2=r2,
            env=env,
            test_mode=test_mode,
            model_uri=artifact_uri,
        )

        # Clean up temp directory in PROD
        if env == "prod":
            if os.path.exists("tmp_rf_model"):
                shutil.rmtree("tmp_rf_model")
                print("🧹 Dossier temporaire supprimé : tmp_rf_model")

        print(f"✅ Modèle RF logged to MLflow + sauvegardé dans : {model_dir}")

    # Return metrics + real run id and model uri + double eval results
    result = {
        "rmse": rmse,
        "r2": r2,
        "run_id": run.info.run_id,
        "model_uri": artifact_uri,
    }

    # Add double evaluation results if available
    if double_eval_results:
        result.update(
            {
                "metrics_baseline": double_eval_results["metrics_baseline"],
                "metrics_current": double_eval_results["metrics_current"],
                "baseline_regression": double_eval_results["baseline_regression"],
                "test_current_size": double_eval_results["test_current_size"],
                "test_baseline_size": double_eval_results["test_baseline_size"],
            }
        )

    return result


def train_nn(X, y, env, test_mode, data_source="reference", current_data_df=None):
    """
    Train Neural Network model with optional double evaluation.

    Args:
        X: Training features
        y: Training target
        env: "dev" or "prod"
        test_mode: bool
        data_source: "reference", "current", or "baseline"
        current_data_df: Optional DataFrame for double evaluation (requires >= 200 samples)

    Returns:
        dict with metrics and optionally double evaluation results
    """
    y = y.to_numpy(dtype="float32")
    run_name = f"NeuralNet_Train_{env}" + ("_TEST" if test_mode else "")

    with mlflow.start_run(run_name=run_name):
        run = mlflow.active_run()
        mlflow.set_tag("mode", env)
        mlflow.set_tag("test_mode", test_mode)
        mlflow.set_tag("data_source", data_source)
        mlflow.log_metric("test_mode", int(test_mode))
        mlflow.tensorflow.autolog(disable=True)

        # Tag if double evaluation enabled
        if current_data_df is not None:
            mlflow.set_tag("double_evaluation", True)
            mlflow.log_metric("current_data_size", len(current_data_df))

        nn = NNPipeline()
        epochs = 50
        batch_size = 128
        nn.fit(X, y, epochs=epochs, batch_size=batch_size)

        mlflow.log_param("nn_epochs", epochs)
        mlflow.log_param("nn_batch_size", batch_size)
        mlflow.log_param("nn_embedding_dim", nn.embedding_dim)

        total_params = nn.model.count_params()
        mlflow.log_metric("nn_total_params", total_params)

        y_pred = nn.predict(X).flatten()
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        mlflow.log_metric("nn_rmse_train", rmse)
        mlflow.log_metric("nn_r2_train", r2)

        print(f"🎯 NN – RMSE : {rmse:.2f} | R² : {r2:.4f} | Params: {total_params}")

        # NEW: Double evaluation if current_data provided
        double_eval_results = None
        if current_data_df is not None and len(current_data_df) >= 200:
            try:
                test_baseline_path = get_test_baseline_path(env)
                double_eval_results = evaluate_double(
                    model=nn,
                    test_baseline_path=test_baseline_path,
                    current_data_df=current_data_df,
                    model_type="nn",
                )

                # Log double metrics to MLflow
                mlflow.log_metrics(
                    {
                        "baseline_rmse": double_eval_results["metrics_baseline"][
                            "rmse"
                        ],
                        "baseline_r2": double_eval_results["metrics_baseline"]["r2"],
                        "baseline_mae": double_eval_results["metrics_baseline"]["mae"],
                        "current_rmse": double_eval_results["metrics_current"]["rmse"],
                        "current_r2": double_eval_results["metrics_current"]["r2"],
                        "current_mae": double_eval_results["metrics_current"]["mae"],
                        "baseline_regression": int(
                            double_eval_results["baseline_regression"]
                        ),
                        "test_baseline_size": double_eval_results["test_baseline_size"],
                        "test_current_size": double_eval_results["test_current_size"],
                    }
                )

                print("✅ Double evaluation metrics logged to MLflow")

            except Exception as e:
                print(f"⚠️ Double evaluation failed: {e}")
                import traceback

                print(traceback.format_exc())
                # Continue without double eval - don't fail the training
        elif current_data_df is not None and len(current_data_df) < 200:
            print(
                f"⚠️ Current data too small for double evaluation: {len(current_data_df)} < 200 samples"
            )

        # Save model to temporary directory for MLflow upload
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = os.path.join(tmp_dir, "model")
            os.makedirs(model_dir, exist_ok=True)
            nn.save(model_dir)

            # Log complete model pipeline to MLflow (uploads to GCS artifact root)
            mlflow.log_artifacts(model_dir, artifact_path="model")

            # Get exact artifact URI from MLflow
            artifact_uri = mlflow.get_artifact_uri("model")

        # Register model if not in test mode
        if not test_mode:
            model_uri = f"runs:/{run.info.run_id}/model"
            mlflow.register_model(model_uri, "bike-traffic-nn")

        # Update summary.json in GCS
        update_model_summary(
            model_type="nn",
            run_id=run.info.run_id,
            rmse=rmse,
            r2=r2,
            env=env,
            test_mode=test_mode,
            model_uri=artifact_uri,
        )

        print("✅ Modèle NN logged to MLflow (artifacts uploaded to GCS)")

    # Return metrics + real run id and model uri + double eval results
    result = {
        "rmse": rmse,
        "r2": r2,
        "run_id": run.info.run_id,
        "model_uri": artifact_uri,
    }

    # Add double evaluation results if available
    if double_eval_results:
        result.update(
            {
                "metrics_baseline": double_eval_results["metrics_baseline"],
                "metrics_current": double_eval_results["metrics_current"],
                "baseline_regression": double_eval_results["baseline_regression"],
                "test_current_size": double_eval_results["test_current_size"],
                "test_baseline_size": double_eval_results["test_baseline_size"],
            }
        )

    return result


def train_rfc(X_raw, y_unused, env, test_mode):
    run_name = f"AffluenceClassifier_Train_{env}" + ("_TEST" if test_mode else "")
    print("📡 Tracking URI :", mlflow.get_tracking_uri())

    with mlflow.start_run(run_name=run_name):
        run = mlflow.active_run()
        mlflow.set_tag("mode", env)
        mlflow.set_tag("test_mode", test_mode)
        mlflow.log_metric("test_mode", int(test_mode))
        mlflow.sklearn.autolog(disable=True)

        # Entraînement
        clf = AffluenceClassifierPipeline()
        clf.fit(X_raw)

        # Évaluation
        y_true = clf.y_test
        y_pred = clf.model.predict(clf.X_test)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        mlflow.log_metric("rfc_accuracy", acc)
        mlflow.log_metric("rfc_precision", prec)
        mlflow.log_metric("rfc_recall", rec)
        mlflow.log_metric("rfc_f1_score", f1)

        print(
            f"🎯 RFC – Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}"
        )

        # Sauvegarde locale
        model_dir = (
            os.path.join("tmp_rfc_model", "rf_class")
            if env == "prod"
            else os.path.join("models", "rf_class")
        )
        os.makedirs(model_dir, exist_ok=True)
        clf.save(model_dir)

        # Logging MLflow
        mlflow.log_artifacts(model_dir, artifact_path="rfc_model")

        if env == "prod":
            # Export vers GCS
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            gcs_model_uri = (
                f"gs://df_traffic_cyclist1/models/rf_class/{timestamp}/rf_class/"
            )
            result = subprocess.run(  # nosec B603
                ["gsutil", "-m", "cp", "-r", model_dir, gcs_model_uri],
                check=False,
                capture_output=True,
            )
            cp_success = result.returncode == 0

            if cp_success:
                update_summary(
                    summary_path="gs://df_traffic_cyclist1/models/summary.json",
                    model_type="rf_class",
                    run_id=run.info.run_id,
                    model_uri=gcs_model_uri,
                    env=env,
                    test_mode=test_mode,
                    accuracy=acc,
                    precision=prec,
                    recall=rec,
                    f1_score=f1,
                )
                print(f"📤 Modèle Affluence exporté vers {gcs_model_uri}")
            else:
                print("❌ Échec upload modèle Affluence vers GCS")

            if os.path.exists("tmp_rfc_model"):
                shutil.rmtree("tmp_rfc_model")
                print("🧹 Dossier temporaire supprimé : tmp_rfc_model")
        else:
            print(f"✅ Modèle RFC sauvegardé localement dans : {model_dir}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}


def train_model(
    model_type: str,
    data_source: str = "reference",
    env: str = "dev",  # Aligned with CLI default
    hyperparams: dict = None,
    test_mode: bool = False,
    current_data_df: pd.DataFrame = None,  # NEW: For double evaluation
):
    """
    Train a single model and return results.

    Args:
        model_type: "rf", "nn", or "rf_class"
        data_source: "reference", "current", or "baseline" (train_baseline.csv)
        env: "dev" or "prod" (default: "dev")
        hyperparams: dict of hyperparameters (optional)
        test_mode: if True, use small sample (1000 rows) for fast testing
        current_data_df: Optional current data for double evaluation (>= 200 samples)

    Returns:
        dict with run_id, metrics, model_uri, and optionally double evaluation results
    """
    # Map data_source to file path
    if test_mode:
        # Use small test sample for fast training (avoid loading 1GB file)
        data_path = "data/test_sample.csv"
        print("⚡ TEST MODE: Using test sample (1000 rows)")
    elif data_source == "baseline":
        # NEW: Use train_baseline.csv for champion training (env-based path)
        data_path = os.getenv("TRAIN_DATA_PATH", "data/train_baseline.csv")
        print(f"🏆 CHAMPION TRAINING: Using baseline from {data_path}")
    elif data_source == "reference":
        data_path = "data/reference_data.csv"
    elif data_source == "current":
        data_path = "data/current_data.csv"
    else:
        data_path = "data/comptage-velo-donnees-compteurs.csv"

    # Setup environment
    artifact_path = "models/"
    os.makedirs(artifact_path, exist_ok=True)

    # Use MLFLOW_TRACKING_URI from env, fallback to localhost for non-docker
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("bike-traffic-training")  # New experiment with GCS artifacts
    print(f"📡 MLflow tracking URI: {mlflow_uri}")

    # Note: GCS credentials already set at module import

    # Load data
    print(f"📊 Loading data from {data_path}...")
    X, y = load_and_clean_data(data_path)
    print(f"✅ Data loaded: {X.shape[0]} rows")

    if model_type == "rf":
        # Pass current_data_df for double evaluation
        metrics = train_rf(
            X, y, env, test_mode, data_source, current_data_df=current_data_df
        )
        return {
            "run_id": metrics.get("run_id"),
            "metrics": {k: metrics[k] for k in ("rmse", "r2") if k in metrics},
            "model_uri": metrics.get("model_uri"),
            # NEW: Double evaluation results
            "metrics_baseline": metrics.get("metrics_baseline"),
            "metrics_current": metrics.get("metrics_current"),
            "baseline_regression": metrics.get("baseline_regression", False),
        }

    elif model_type == "nn":
        # Pass current_data_df for double evaluation
        metrics = train_nn(
            X, y, env, test_mode, data_source, current_data_df=current_data_df
        )
        return {
            "run_id": metrics.get("run_id"),
            "metrics": {k: metrics[k] for k in ("rmse", "r2") if k in metrics},
            "model_uri": metrics.get("model_uri"),
            # NEW: Double evaluation results
            "metrics_baseline": metrics.get("metrics_baseline"),
            "metrics_current": metrics.get("metrics_current"),
            "baseline_regression": metrics.get("baseline_regression", False),
        }

    elif model_type == "rf_class":
        df_raw = load_and_clean_data(data_path, preserve_target=True)
        metrics = train_rfc(df_raw, None, env, test_mode)
        return {
            "run_id": metrics.get("run_id"),
            "metrics": metrics,
            "model_uri": metrics.get("model_uri"),
        }

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train bike count models")
    parser.add_argument(
        "--model-type",
        choices=["rf", "nn", "rf_class", "all"],
        default="all",
        help="Model type to train (default: all)",
    )
    parser.add_argument(
        "--data-source",
        choices=["reference", "current", "baseline"],
        default="baseline",
        help="Data source (default: baseline = train_baseline.csv)",
    )
    parser.add_argument(
        "--model-test",
        action="store_true",
        help="Use 1000 samples for fast training",
    )
    parser.add_argument(
        "--env", default="dev", choices=["dev", "prod"], help="Choose 'dev' or 'prod'"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("🏆 MODEL TRAINING")
    print("=" * 60)
    print(f"Model type: {args.model_type}")
    print(f"Data source: {args.data_source}")
    print(f"Environment: {args.env}")
    print(f"Test mode: {args.model_test}")
    print("=" * 60)
    print()

    # Train specific model or all models
    if args.model_type == "all":
        # Legacy behavior: train all models
        data_path, artifact_path = setup_environment(args.env, args.model_test)
        print(f"✅ Environnement {args.env} configuré. Données : {data_path}")

        X, y = load_and_clean_data(data_path)
        print(f"📊 Données chargées : {X.shape[0]} lignes")

        train_rf(X, y, args.env, args.model_test)
        train_nn(X, y, args.env, args.model_test)

        df_raw = load_and_clean_data(data_path, preserve_target=True)
        train_rfc(df_raw, None, args.env, args.model_test)
    else:
        # NEW: Train single model with data_source support
        result = train_model(
            model_type=args.model_type,
            data_source=args.data_source,
            env=args.env,
            test_mode=args.model_test,
        )
        print()
        print("=" * 60)
        print("✅ TRAINING COMPLETE")
        print("=" * 60)
        print(f"Run ID: {result['run_id']}")
        print(f"Metrics: {result['metrics']}")
        print(f"Model URI: {result['model_uri']}")
        print("=" * 60)
