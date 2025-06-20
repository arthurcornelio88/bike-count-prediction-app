import os
import shutil
import hashlib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple
import pandas as pd

from app.model_registry_summary import get_best_model_from_summary

app = FastAPI()

# === Gestion des credentials GCP ===
GCP_CREDENTIAL_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if not GCP_CREDENTIAL_PATH or not os.path.exists(GCP_CREDENTIAL_PATH):
    raise EnvironmentError("❌ GCP credentials non trouvés ou GOOGLE_APPLICATION_CREDENTIALS non définie.")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_CREDENTIAL_PATH

# === Cache global des modèles ===
model_cache = {}

# === Fonction utilitaire de cache avec nettoyage ===
def get_cache_dir(model_type: str, metric: str) -> str:
    """Génère un chemin unique et réutilisable dans /tmp"""
    key = f"{model_type}_{metric}"
    key_hash = hashlib.md5(key.encode()).hexdigest()
    path = f"/tmp/model_cache_{key_hash}"
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
            download_dir=cache_dir  # ← Important pour contrôler le chemin
        )
        model_cache[key] = model
    return model_cache[key]

# === Chargement anticipé au démarrage ===
@app.on_event("startup")
def preload_models():
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

# === Endpoint principal ===
@app.post("/predict")
def predict(data: PredictRequest):
    df = pd.DataFrame(data.records)

    try:
        model = get_cached_model(data.model_type, data.metric)
        y_pred = model.predict_clean(df)
        return {"predictions": y_pred.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
