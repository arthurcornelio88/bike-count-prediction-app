import os
import json
import pandas as pd
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.custom_model_registry_summary import get_best_model_from_summary

# === Initialisation FastAPI ===
app = FastAPI()

# === Chargement des credentials GCP (prod ou dev) ===
def setup_credentials():
    key_json = os.getenv("GCP_JSON_CONTENT")
    if not key_json:
        raise EnvironmentError("❌ GCP_JSON_CONTENT introuvable")
    cred_path = "/tmp/gcp_creds.json"
    with open(cred_path, "w") as f:
        f.write(key_json)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path
    print("✅ Credentials chargés en PROD")


# === Cache du modèle ===
loaded_model = None

# === Chargement du modèle au démarrage ===
@app.on_event("startup")
async def load_model():
    global loaded_model
    print("🚀 Startup init")

    try:
        setup_credentials()
        summary_path = "gs://df_traffic_cyclist1/models/summary.json"
        print(f"📄 Lecture du résumé : {summary_path}")

        loaded_model = get_best_model_from_summary(
            model_type="rf_class",
            metric="f1_score",
            summary_path=summary_path,
            env="prod" # os.getenv("ENV", "dev").lower()
        )

        print("✅ Modèle 'rf_class' préchargé avec succès.")
    except Exception as e:
        print(f"⚠️ Échec du chargement du modèle au démarrage : {e}")
        loaded_model = None

# === Schéma de la requête ===
class PredictRequest(BaseModel):
    records: List[dict]

# === Endpoint prédiction ===
@app.post("/predict")
def predict(data: PredictRequest):
    print("📥 Requête reçue :", data.records)

    if loaded_model is None:
        raise HTTPException(status_code=500, detail="❌ Modèle non chargé au démarrage.")

    try:
        df = pd.DataFrame(data.records)
        y_pred = loaded_model.predict(df)
        return {
            "predictions": y_pred.tolist(),
            "label": ["faible" if p == 0 else "forte" for p in y_pred]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Erreur de prédiction : {e}")

# === Endpoint rechargement manuel ===
@app.get("/refresh_model")
def refresh_model():
    global loaded_model
    try:
        setup_credentials()
        summary_path = "gs://df_traffic_cyclist1/models/summary.json"
        loaded_model = get_best_model_from_summary(
            model_type="rf_class",
            metric="f1_score",
            summary_path=summary_path,
            env=os.getenv("ENV", "dev").lower()
        )
        return {"message": "✅ Modèle 'rf_class' rechargé avec succès."}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"❌ Échec du rechargement du modèle : {str(e)}"}
        )
