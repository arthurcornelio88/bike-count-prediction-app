import os
import json
import uuid
import subprocess
import tempfile
from typing import Optional
import datetime
from urllib.request import urlopen
from app.classes import AffluenceClassifierPipeline


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
):
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "model_type": model_type,
        "env": env,
        "test_mode": test_mode,
        "run_id": run_id,
        "model_uri": model_uri,
    }

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

    summary_path_local = (
        os.path.join(tempfile.gettempdir(), "summary.json")
        if summary_path.startswith("gs://")
        else summary_path
    )
    summary = []

    if summary_path.startswith("gs://"):
        subprocess.run(  # noqa: S603
            ["gsutil", "cp", summary_path, summary_path_local],
            check=False,
            capture_output=True,
        )
        if not os.path.exists(summary_path_local):
            open(summary_path_local, "a").close()  # noqa: PTH123
    if os.path.exists(summary_path_local):
        with open(summary_path_local, "r") as f:
            try:
                summary = json.load(f)
            except json.JSONDecodeError:
                print("⚠️ summary.json vide ou corrompu. Réinitialisation.")
                summary = []

    summary.append(entry)

    with open(summary_path_local, "w") as f:
        json.dump(summary, f, indent=2)

    if summary_path.startswith("gs://"):
        subprocess.run(["gsutil", "cp", summary_path_local, summary_path], check=True)  # noqa: S603
        print(f"✅ summary.json mis à jour et uploadé vers {summary_path}")
    else:
        print(f"✅ summary.json mis à jour localement : {summary_path}")


def get_best_model_from_summary(
    model_type: str,
    summary_path: str,
    env: str = "prod",
    metric: str = "rmse",
    test_mode: Optional[bool] = False,
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
    ]

    if not filtered:
        raise RuntimeError(
            f"Aucun modèle trouvé pour type={model_type}, env={env}, test_mode={test_mode}"
        )

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

    local_model_path = _download_gcs_dir(best["model_uri"], prefix=model_type)
    print(f"⏳ Étape 4 – Chargement du modèle depuis {local_model_path}")

    subfolder = os.path.join(local_model_path, model_type)
    if os.path.isdir(subfolder):
        print(f"📁 Sous-dossier détecté pour {model_type}, on l'utilise : {subfolder}")
        local_model_path = subfolder
    else:
        print(f"⚠️ Aucun sous-dossier {model_type} trouvé dans {local_model_path}")
        print(f"📂 Contenu détecté : {os.listdir(local_model_path)}")

    if model_type == "rf_class":
        return AffluenceClassifierPipeline.load(local_model_path)
    else:
        raise ValueError("Type de modèle non reconnu")


def _download_gcs_dir(gcs_uri: str, prefix="model") -> str:
    from google.cloud import storage

    bucket_name, path = gcs_uri.replace("gs://", "").split("/", 1)
    local_tmp_dir = os.path.join(tempfile.gettempdir(), f"{prefix}_{uuid.uuid4().hex}")
    os.makedirs(local_tmp_dir, exist_ok=True)

    print(
        f"🔐 Credentials utilisés : {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}"
    )
    client = storage.Client()
    blobs = list(client.list_blobs(bucket_name, prefix=path))

    print(f"📂 GCS path to download: {gcs_uri}")
    print(f"📥 Destination locale: {local_tmp_dir}")

    for blob in blobs:
        rel_path = os.path.relpath(blob.name, path)
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
