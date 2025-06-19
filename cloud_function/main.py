import os
import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict

from app.model_registry_summary import get_best_model_from_summary

# === GCP credentials ===
GCP_CREDENTIAL_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if GCP_CREDENTIAL_PATH and os.path.exists(GCP_CREDENTIAL_PATH):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_CREDENTIAL_PATH
else:
    raise EnvironmentError("❌ GCP credentials non trouvés ou GOOGLE_APPLICATION_CREDENTIALS non définie.")

# === Cache global des modèles ===
loaded_model: object = None  # On garde un seul modèle en cache

# === Préchargement du modèle rf_class lors du startup ===
@app.on_event("startup")
async def load_model():
    global loaded_model
    summary_path = "gs://df_traffic_cyclist1/models/summary.json"
    try:
        # Charger le modèle rf_class, nous utilisons ici "f1_score" comme métrique
        loaded_model = get_best_model_from_summary(
            model_type="rf_class",
            metric="f1_score",  # Adapté au cas d'un modèle de classification
            summary_path=summary_path,
            env="prod"
        )
        print("✅ Modèle 'rf_class' préchargé")
    except Exception as e:
        print(f"⚠️ Échec chargement modèle 'rf_class' : {e}")

# === FastAPI app ===
app = FastAPI()

# === Schéma de la requête ===
class PredictRequest(BaseModel):
    records: List[dict]

# === Endpoint de prédiction ===
@app.post("/predict")
def predict(data: PredictRequest):
    if loaded_model is None:
        raise HTTPException(status_code=404, detail="Modèle 'rf_class' non disponible")

    df = pd.DataFrame(data.records)

    try:
        # Prédiction avec le modèle
        y_pred = loaded_model.predict(df)
        return {"predictions": y_pred.tolist(), "label": ["faible" if p == 0 else "forte" for p in y_pred]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {e}")

# === Endpoint manuel pour recharger le modèle ===
@app.get("/refresh_model")
def refresh_model():
    global loaded_model
    summary_path = "gs://df_traffic_cyclist1/models/summary.json"
    try:
        # Recharger le modèle
        loaded_model = get_best_model_from_summary(
            model_type="rf_class",
            metric="f1_score",  # Toujours la même métrique pour 'rf_class'
            summary_path=summary_path,
            env="prod"
        )
        return {"message": "Modèle 'rf_class' rafraîchi avec succès"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Échec du rechargement du modèle 'rf_class' : {str(e)}"}
        )
