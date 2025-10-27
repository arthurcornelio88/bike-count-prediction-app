"""
Helper functions for bike traffic DAGs
Provides utilities for BigQuery, GCS, and data processing
"""

import os
import time
from datetime import datetime, timedelta

import gcsfs
import pandas as pd
from google.cloud import bigquery


def get_storage_path(subdir: str, filename: str) -> str:
    """
    Returns the environment-aware storage path for a given subdir and filename.
    DEV: Local filesystem under ./data/
    PROD: Google Cloud Storage bucket path

    Args:
        subdir: Subdirectory within storage root (e.g., 'raw_data', 'models')
        filename: File name (can be empty for directory paths)

    Returns:
        Full path string (gs://... or local path)

    Examples:
        DEV:  get_storage_path("raw_data", "current.csv") → "./data/current.csv"
        PROD: get_storage_path("raw_data", "current.csv") → "gs://df_traffic_cyclist1/raw_data/current.csv"
    """
    env = os.getenv("ENV", "DEV")
    gcs_bucket = os.getenv("GCS_BUCKET", "df_traffic_cyclist1")

    if env == "PROD":
        # Use GCS path (bucket structure: raw_data/, models/, mlruns/, dvc-storage/)
        if subdir:
            return (
                f"gs://{gcs_bucket}/{subdir}/{filename}"
                if filename
                else f"gs://{gcs_bucket}/{subdir}/"
            )
        else:
            return (
                f"gs://{gcs_bucket}/{filename}" if filename else f"gs://{gcs_bucket}/"
            )
    else:
        # Use local path (project structure: data/, models/, mlruns/, etc.)
        # Special mapping: raw_data → data/ localement
        local_subdir = "data" if subdir == "raw_data" else subdir

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        if local_subdir:
            full_path = (
                os.path.join(base_dir, local_subdir, filename)
                if filename
                else os.path.join(base_dir, local_subdir) + "/"
            )
        else:
            full_path = os.path.join(base_dir, filename) if filename else base_dir + "/"

        return full_path


def get_reference_data_path() -> str:
    """
    Returns the path to reference_data.csv
    DEV: ./data/reference_data.csv
    PROD: gs://df_traffic_cyclist1/raw_data/reference_data.csv
    """
    env = os.getenv("ENV", "DEV")
    if env == "PROD":
        bucket = os.getenv("GCS_BUCKET", "df_traffic_cyclist1")
        return f"gs://{bucket}/raw_data/reference_data.csv"
    else:
        # Local: data/reference_data.csv
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../data/reference_data.csv")
        )


def get_current_data_path() -> str:
    """
    Returns the path to current_data.csv
    DEV: ./data/current_data.csv
    PROD: gs://df_traffic_cyclist1/raw_data/current_data.csv
    """
    env = os.getenv("ENV", "DEV")
    if env == "PROD":
        bucket = os.getenv("GCS_BUCKET", "df_traffic_cyclist1")
        return f"gs://{bucket}/raw_data/current_data.csv"
    else:
        # Local: data/current_data.csv
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../data/current_data.csv")
        )


def read_gcs_csv(path: str) -> pd.DataFrame:
    """
    Lit un fichier CSV, que ce soit en local ou sur GCS (gs://...).

    Args:
        path: Le chemin vers le fichier CSV

    Returns:
        DataFrame chargé

    Raises:
        FileNotFoundError: Si le fichier est introuvable
    """
    if path.startswith("gs://"):
        fs = gcsfs.GCSFileSystem(skip_instance_cache=True, cache_timeout=0)
        if not fs.exists(path):
            raise FileNotFoundError(f"⛔ Fichier introuvable sur GCS: {path}")
        with fs.open(path, "r") as f:
            return pd.read_csv(f)
    else:
        if not os.path.exists(path):
            raise FileNotFoundError(f"⛔ Fichier local introuvable: {path}")
        return pd.read_csv(path)


def write_csv(df: pd.DataFrame, path: str):
    """
    Écrit un DataFrame en CSV (local ou GCS)

    Args:
        df: DataFrame à écrire
        path: Chemin de destination (local ou gs://)
    """
    if path.startswith("gs://"):
        print(f"📝 Saving to GCS: {path}")
        fs = gcsfs.GCSFileSystem(skip_instance_cache=True, cache_timeout=0)
        with fs.open(path, "w") as f:
            df.to_csv(f, index=False)
            f.flush()
        fs.invalidate_cache(path)
        # Validation immédiate
        if not fs.exists(path):
            raise RuntimeError(f"❌ GCS file not found right after saving: {path}")
        print(f"✅ File written and verified on GCS: {path}")
    else:
        print(f"📝 Saving locally: {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)


def file_exists(path: str) -> bool:
    """Vérifie si un fichier existe (local ou GCS)"""
    if path.startswith("gs://"):
        fs = gcsfs.GCSFileSystem(skip_instance_cache=True, cache_timeout=0)
        return fs.exists(path)
    else:
        return os.path.exists(path)


def wait_for_gcs(path: str, timeout: int = 30):
    """
    Attends que le fichier GCS soit visible (avec un timeout en secondes)

    Args:
        path: Chemin GCS à vérifier
        timeout: Temps maximum d'attente en secondes

    Raises:
        FileNotFoundError: Si le fichier n'apparaît pas après timeout
    """
    if not path.startswith("gs://"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Local file not found: {path}")
        return

    fs = gcsfs.GCSFileSystem(skip_instance_cache=True, cache_timeout=0)
    for i in range(timeout):
        if fs.exists(path):
            print(f"✅ GCS file detected: {path}")
            return
        print(f"⏳ Waiting for GCS propagation ({i+1}/{timeout}): {path}")
        time.sleep(1)

    raise FileNotFoundError(f"⛔ File not found in GCS after {timeout}s: {path}")


def create_bq_dataset_if_not_exists(
    project_id: str, dataset_id: str, location: str = "europe-west1"
):
    """
    Crée un dataset BigQuery s'il n'existe pas déjà

    Args:
        project_id: ID du projet GCP
        dataset_id: ID du dataset à créer
        location: Location du dataset (défaut: europe-west1)
    """
    client = bigquery.Client(project=project_id)
    dataset_ref = f"{project_id}.{dataset_id}"

    try:
        client.get_dataset(dataset_ref)
        print(f"✅ Dataset exists: {dataset_ref}")
    except Exception as e:
        if "Not found" in str(e) or "404" in str(e):
            print(f"⚠️ Dataset not found. Creating: {dataset_ref}")
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = location
            try:
                client.create_dataset(dataset, exists_ok=True)
                print(f"✅ Dataset created: {dataset_ref}")
            except Exception as create_error:
                if "Already Exists" in str(create_error):
                    print(f"✅ Dataset already exists (race condition): {dataset_ref}")
                else:
                    raise
        else:
            raise


def create_monitoring_table_if_needed(
    project_id: str,
    dataset_id: str = "monitoring_audit",
    table_id: str = "logs",
    location: str = "europe-west1",
):
    """
    Crée la table de monitoring si elle n'existe pas

    Args:
        project_id: ID du projet GCP
        dataset_id: ID du dataset (défaut: monitoring_audit)
        table_id: ID de la table (défaut: logs)
        location: Location (défaut: europe-west1)
    """
    client = bigquery.Client(project=project_id)
    full_dataset_id = f"{project_id}.{dataset_id}"
    full_table_id = f"{full_dataset_id}.{table_id}"

    # Vérifie si le dataset existe
    try:
        client.get_dataset(full_dataset_id)
        print(f"✅ Dataset exists: {full_dataset_id}")
    except Exception as e:
        if "Not found" in str(e) or "404" in str(e):
            print(f"⚠️ Dataset not found. Creating: {full_dataset_id}")
            dataset = bigquery.Dataset(full_dataset_id)
            dataset.location = location
            try:
                client.create_dataset(dataset, exists_ok=True)
                print(f"✅ Dataset created: {full_dataset_id}")
            except Exception as create_error:
                if "Already Exists" in str(create_error):
                    print(
                        f"✅ Dataset already exists (race condition): {full_dataset_id}"
                    )
                else:
                    raise
        else:
            raise

    # Vérifie si la table existe
    try:
        client.get_table(full_table_id)
        print(f"✅ Table already exists: {full_table_id}")
    except Exception:
        print(f"⚠️ Table not found, creating: {full_table_id}")
        schema = [
            bigquery.SchemaField("timestamp", "TIMESTAMP"),
            bigquery.SchemaField("drift_detected", "BOOL"),
            bigquery.SchemaField("rmse", "FLOAT"),
            bigquery.SchemaField("r2", "FLOAT"),
            bigquery.SchemaField("fine_tune_triggered", "BOOL"),
            bigquery.SchemaField("fine_tune_success", "BOOL"),
            bigquery.SchemaField("model_improvement", "FLOAT"),
            bigquery.SchemaField("env", "STRING"),
            bigquery.SchemaField("error_message", "STRING"),
        ]
        table = bigquery.Table(full_table_id, schema=schema)
        client.create_table(table)
        print(f"✅ Table created: {full_table_id}")


def fetch_historical_data_from_bq(
    bq_client: bigquery.Client,
    bq_project: str,
    dataset: str,
    days_back: int = 7,
    limit_per_day: int = 100,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fetches historical data from the past N days from BigQuery.

    Args:
        bq_client: Initialized BigQuery client
        bq_project: Project ID
        dataset: Dataset name (e.g. 'bike_traffic_raw')
        days_back: Number of past days to search
        limit_per_day: Maximum records per day
        verbose: Whether to print progress logs

    Returns:
        A DataFrame of historical samples (may be empty)
    """
    data_frames = []

    for i in range(1, days_back + 1):
        day = (datetime.utcnow() - timedelta(days=i)).strftime("%Y%m%d")
        table_id = f"{bq_project}.{dataset}.daily_{day}"

        try:
            query = f"SELECT * FROM `{table_id}` LIMIT {limit_per_day}"  # nosec B608
            df = bq_client.query(query).to_dataframe()

            if not df.empty:
                data_frames.append(df)
                if verbose:
                    print(f"✅ Found {len(df)} records in {table_id}")

        except Exception as e:
            if verbose:
                print(f"⚠️ Skipped {table_id}: {e}")

    if data_frames:
        return pd.concat(data_frames, ignore_index=True)
    else:
        if verbose:
            print("🚫 No historical data found")
        return pd.DataFrame()


def host_to_docker_path(path: str) -> str:
    """
    Convertit un chemin absolu local (host) en chemin Docker /app/...
    Utile pour passer des paths entre Airflow (host) et API (Docker)

    Args:
        path: Chemin local absolu

    Returns:
        Chemin Docker équivalent
    """
    # Replace host base path with Docker /app path
    base_host_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    if path.startswith(base_host_path):
        relative_path = os.path.relpath(path, base_host_path)
        return f"/app/{relative_path}"
    return path


if __name__ == "__main__":
    # Test des fonctions
    print("🧪 Testing bike_helpers functions...")

    # Test get_storage_path
    print("\n📁 Testing get_storage_path:")
    print(f"  DEV raw_data: {get_storage_path('raw_data', 'test.csv')}")
    print(f"  DEV models: {get_storage_path('models', 'model.pkl')}")

    # Test avec ENV=PROD
    os.environ["ENV"] = "PROD"
    print(f"  PROD raw_data: {get_storage_path('raw_data', 'test.csv')}")
    print(f"  PROD models: {get_storage_path('models', 'model.pkl')}")

    # Test reference/current paths
    os.environ["ENV"] = "DEV"
    print(f"\n📊 Reference data path (DEV): {get_reference_data_path()}")
    print(f"📊 Current data path (DEV): {get_current_data_path()}")

    print("\n✅ Tests completed!")
