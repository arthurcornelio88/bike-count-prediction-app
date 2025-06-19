import os
import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict
from contextlib import asynccontextmanager

from app.model_registry_summary import get_best_model_from_summary

# GCP credentials
GCP_CREDENTIAL_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if GCP_CREDENTIAL_PATH and os.path.exists(GCP_CREDENTIAL_PATH):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_CREDENTIAL_PATH
else:
    raise EnvironmentError("❌ GCP credentials non trouvés ou GOOGLE_APPLICATION_CREDENTIALS non définie.")

# === Cache global des modèles
loaded_models: Dict[str, object] = {}

# === Préchargement lors du startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    summary_path = "gs://df_traffic_cyclist1/models/summary.json"
    for model_type in ["nn", "rf", "rf_class"]:
        try:
            model = get_best_model_from_summary(
                model_type=model_type,
                metric="r2" if model_type != "rf_class" else "f1_score",
                summary_path=summary_path,
                env="prod"
            )
            loaded_models[model_type] = model
            print(f"✅ Modèle préchargé pour '{model_type}'")
        except Exception as e:
            print(f"⚠️ Échec chargement modèle '{model_type}' : {e}")
    yield

# === FastAPI app avec lifespan
app = FastAPI(lifespan=lifespan)

# === Schéma de la requête
class PredictRequest(BaseModel):
    records: List[dict]
    model_type: str
    metric: str = "r2"

# === Endpoint prédiction
@app.post("/predict")
def predict(data: PredictRequest):
    if data.model_type not in loaded_models:
        raise HTTPException(status_code=404, detail=f"Modèle '{data.model_type}' non disponible")

    df = pd.DataFrame(data.records)
    model = loaded_models[data.model_type]

    try:
        if data.model_type == "rf_class":
            y_pred = model.predict(df)
            return {"predictions": y_pred.tolist(), "label": ["faible" if p == 0 else "forte" for p in y_pred]}
        else:
            y_pred = model.predict_clean(df)
            return {"predictions": y_pred.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {e}")

# === Endpoint manuel pour recharger les modèles (optionnel)
@app.get("/refresh_models")
def refresh_models():
    summary_path = "gs://df_traffic_cyclist1/models/summary.json"
    refreshed = []

    for model_type in ["nn", "rf", "rf_class"]:
        try:
            model = get_best_model_from_summary(
                model_type=model_type,
                metric="r2" if model_type != "rf_class" else "f1_score",
                summary_path=summary_path,
                env="prod"
            )
            loaded_models[model_type] = model
            refreshed.append(model_type)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"Échec pour {model_type}: {str(e)}"}
            )

    return {"refreshed_models": refreshed}
