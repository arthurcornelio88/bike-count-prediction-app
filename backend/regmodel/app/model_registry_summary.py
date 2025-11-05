import os
import json
import uuid
import tempfile
from typing import Optional
import datetime
from urllib.request import urlopen
from app.classes import RFPipeline, NNPipeline, AffluenceClassifierPipeline
from google.cloud import storage


def update_summary(
    summary_path: str,
    model_type: str,
    run_id: str,
    model_uri: str,
    env: str = "prod",
    test_mode: bool = False,
    rmse: float = None,
    r2: float = None,
    accuracy: float = None,
    precision: float = None,
    recall: float = None,
    f1_score: float = None,
    is_champion: bool = False,
):
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "model_type": model_type,
        "env": env,
        "test_mode": test_mode,
        "run_id": run_id,
        "model_uri": model_uri,
        "is_champion": is_champion,
    }

    # Ajoute les métriques si elles sont fournies
    if rmse is not None:
        entry["rmse"] = rmse
    if r2 is not None:
        entry["r2"] = r2
    if accuracy is not None:
        entry["accuracy"] = accuracy
    if precision is not None:
        entry["precision"] = precision
    if recall is not None:
        entry["recall"] = recall
    if f1_score is not None:
        entry["f1_score"] = f1_score

    # Déterminer le chemin local - utiliser tempfile.gettempdir() au lieu de hardcoded /tmp
    summary_path_local = (
        os.path.join(tempfile.gettempdir(), "summary.json")
        if summary_path.startswith("gs://")
        else summary_path
    )
    summary = []

    # Télécharger ou charger si existant - utiliser Google Cloud Storage client
    if summary_path.startswith("gs://"):
        try:
            # Download existing summary from GCS
            bucket_name, blob_path = summary_path.replace("gs://", "").split("/", 1)
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)

            if blob.exists():
                summary_content = blob.download_as_text()
                try:
                    summary = json.loads(summary_content)
                except json.JSONDecodeError:
                    print("⚠️ summary.json vide ou corrompu. Réinitialisation.")
                    summary = []
            else:
                print("ℹ️ summary.json n'existe pas encore dans GCS, création...")
                summary = []
        except Exception as e:
            print(f"⚠️ Erreur lors du téléchargement de summary.json : {e}")
            summary = []
    else:
        # Local file
        if os.path.exists(summary_path_local):
            with open(summary_path_local, "r") as f:
                try:
                    summary = json.load(f)
                except json.JSONDecodeError:
                    print("⚠️ summary.json vide ou corrompu. Réinitialisation.")
                    summary = []
        else:
            summary = []

    summary.append(entry)

    if summary_path.startswith("gs://"):
        # Upload to GCS
        bucket_name, blob_path = summary_path.replace("gs://", "").split("/", 1)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_string(
            json.dumps(summary, indent=2), content_type="application/json"
        )
        print(f"✅ summary.json mis à jour et uploadé vers {summary_path}")
    else:
        # Save locally
        with open(summary_path_local, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"✅ summary.json mis à jour localement : {summary_path}")


def promote_champion(
    summary_path: str,
    model_type: str,
    run_id: str,
    env: str = "prod",
    test_mode: bool = False,
):
    """
    Promote a specific model to champion status.

    This function:
    1. Loads the existing summary.json from GCS or local
    2. Demotes any existing champion (is_champion=False) for the same model_type/env/test_mode
    3. Promotes the specified run_id to champion (is_champion=True)
    4. Saves the updated summary back to GCS or local

    Args:
        summary_path: Path to summary.json (gs:// or local)
        model_type: Type of model (rf, nn, rf_class)
        run_id: MLflow run_id to promote
        env: Environment (prod/dev)
        test_mode: Test mode flag

    Raises:
        ValueError: If run_id not found in summary
    """
    # Déterminer le chemin local - utiliser tempfile.gettempdir() au lieu de hardcoded /tmp
    summary_path_local = (
        os.path.join(tempfile.gettempdir(), "summary.json")
        if summary_path.startswith("gs://")
        else summary_path
    )
    summary = []

    # Télécharger ou charger si existant
    if summary_path.startswith("gs://"):
        try:
            bucket_name, blob_path = summary_path.replace("gs://", "").split("/", 1)
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)

            if blob.exists():
                summary_content = blob.download_as_text()
                try:
                    summary = json.loads(summary_content)
                except json.JSONDecodeError:
                    raise ValueError("⚠️ summary.json vide ou corrompu.")
            else:
                raise ValueError("⚠️ summary.json n'existe pas dans GCS")
        except Exception as e:
            raise ValueError(f"⚠️ Erreur lors du téléchargement de summary.json : {e}")
    else:
        # Local file
        if os.path.exists(summary_path_local):
            with open(summary_path_local, "r") as f:
                try:
                    summary = json.load(f)
                except json.JSONDecodeError:
                    raise ValueError("⚠️ summary.json vide ou corrompu.")
        else:
            raise ValueError(f"⚠️ summary.json n'existe pas : {summary_path_local}")

    # Find the model to promote
    found = False
    for entry in summary:
        if (
            entry["model_type"] == model_type
            and entry["env"] == env
            and entry["test_mode"] == test_mode
        ):
            if entry["run_id"] == run_id:
                entry["is_champion"] = True
                found = True
                print(f"✅ Promoted {model_type} run_id={run_id} to champion")
            else:
                # Demote other champions
                if entry.get("is_champion", False):
                    entry["is_champion"] = False
                    print(f"⬇️ Demoted previous champion run_id={entry['run_id']}")

    if not found:
        raise ValueError(
            f"❌ run_id={run_id} not found for model_type={model_type}, env={env}, test_mode={test_mode}"
        )

    # Save back
    if summary_path.startswith("gs://"):
        # Upload to GCS
        bucket_name, blob_path = summary_path.replace("gs://", "").split("/", 1)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_string(
            json.dumps(summary, indent=2), content_type="application/json"
        )
        print(f"✅ Champion promotion saved to {summary_path}")
    else:
        # Save locally
        with open(summary_path_local, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Champion promotion saved locally : {summary_path}")


# === Chargement du meilleur modèle depuis le résumé
def get_best_model_from_summary(
    model_type: str,
    summary_path: str,
    env: str = "prod",
    metric: str = "rmse",
    test_mode: Optional[bool] = False,
    download_dir: Optional[str] = None,  # 🔧 NOUVEAU
):
    if summary_path.startswith("gs://"):
        summary = _read_gcs_json(summary_path)
    elif summary_path.startswith("http"):
        with urlopen(summary_path) as f:  # nosec B310
            summary = json.load(f)
    else:
        with open(summary_path, "r") as f:
            summary = json.load(f)
    print(f"⏳ Étape 1 – Lecture du résumé depuis {summary_path}")
    print(
        f"⏳ Étape 2 – Filtrage sur model_type={model_type}, env={env}, test_mode={test_mode}"
    )

    filtered = [
        r
        for r in summary
        if r["model_type"] == model_type
        and r["env"] == env
        and r["test_mode"] == test_mode
        # and r["rmse"] > 0  # éviter les modèles fictifs/perfectibles
    ]

    if not filtered:
        raise RuntimeError(
            f"Aucun modèle trouvé pour type={model_type}, env={env}, test_mode={test_mode}"
        )

    # Step 2.1: Check for promoted champion (is_champion=True)
    champions = [r for r in filtered if r.get("is_champion", False)]

    if champions:
        print("🏆 Champion trouvé (is_champion=True), utilisation prioritaire")
        best = champions[0]  # Should be only one, but take first if multiple
    else:
        # Fallback to metric-based selection
        print(f"⚠️ Aucun champion promu, sélection par métrique {metric}")
        metric_sorting = {
            "rmse": lambda r: -r["rmse"],
            "r2": lambda r: r["r2"],
            "f1_score": lambda r: r.get("f1_score", -1),
            "accuracy": lambda r: r.get("accuracy", -1),
        }

        if metric not in metric_sorting:
            raise ValueError(f"Métrique non supportée : {metric}")

        best = max(filtered, key=metric_sorting[metric])
    print(f"🔍 Résumé sélectionné:\n{json.dumps(best, indent=2)}")
    print(f"⏳ Étape 3 – Téléchargement depuis GCS : {best['model_uri']}")

    value = best.get(metric, "N/A")
    print(
        f"✅ Modèle {model_type} sélectionné : {best.get('run_id', 'N/A')} ({metric}={value})"
    )

    local_model_path = _download_gcs_dir(
        best["model_uri"],
        prefix=model_type,
        destination_dir=download_dir,  # 🔧 on transmet à _download_gcs_dir
    )

    print(f"⏳ Étape 4 – Chargement du modèle depuis {local_model_path}")

    # 🔎 Recherche automatique du sous-dossier portant le nom du modèle (ex: rf_class)
    subfolder = os.path.join(local_model_path, model_type)
    if os.path.isdir(subfolder):
        print(f"📁 Sous-dossier détecté pour {model_type}, on l'utilise : {subfolder}")
        local_model_path = subfolder
    else:
        print(f"⚠️ Aucun sous-dossier {model_type} trouvé dans {local_model_path}")
        print(f"📂 Contenu détecté : {os.listdir(local_model_path)}")

    if model_type == "rf":
        return RFPipeline.load(local_model_path)
    elif model_type == "nn":
        return NNPipeline.load(local_model_path)
    elif model_type == "rf_class":
        return AffluenceClassifierPipeline.load(local_model_path)
    else:
        raise ValueError("Type de modèle non reconnu")


# === Téléchargement GCS vers temp directory
def _download_gcs_dir(
    gcs_uri: str, prefix="model", destination_dir: Optional[str] = None
) -> str:
    bucket_name, path = gcs_uri.replace("gs://", "").split("/", 1)
    local_tmp_dir = destination_dir or os.path.join(
        tempfile.gettempdir(), f"{prefix}_{uuid.uuid4().hex}"
    )
    os.makedirs(local_tmp_dir, exist_ok=True)

    client = storage.Client()
    blobs = list(client.list_blobs(bucket_name, prefix=path))

    print(f"📂 GCS path to download: {gcs_uri}")
    print(f"📥 Destination locale: {local_tmp_dir}")

    for blob in blobs:
        rel_path = os.path.relpath(blob.name, path)

        # ⛔ Ignore les "blobs de répertoire"
        if rel_path in (".", "") or blob.name.endswith("/"):
            print(f"🚫 Blob ignoré (répertoire ou vide) : {blob.name}")
            continue

        local_file = os.path.join(local_tmp_dir, rel_path)
        os.makedirs(os.path.dirname(local_file), exist_ok=True)

        try:
            blob.download_to_filename(local_file)
            print(f"➡️ Fichier local : {local_file}")
        except Exception as e:
            print(f"💥 Échec téléchargement : {blob.name}")
            print(f"❌ Exception : {e}")
            raise

    return local_tmp_dir


# === Lecture JSON depuis GCS
def _read_gcs_json(gs_path: str) -> dict:
    from google.cloud import storage

    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        raise EnvironmentError(
            "Variable GOOGLE_APPLICATION_CREDENTIALS non définie pour accéder à GCS"
        )

    bucket_name, blob_path = gs_path.replace("gs://", "").split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    return json.loads(blob.download_as_text())
