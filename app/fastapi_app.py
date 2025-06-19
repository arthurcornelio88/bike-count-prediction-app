import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
from app.model_registry_summary import get_best_model_from_summary

# Lecture des credentials GCP via variable d’environnement
GCP_CREDENTIAL_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if GCP_CREDENTIAL_PATH and os.path.exists(GCP_CREDENTIAL_PATH):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_CREDENTIAL_PATH
else:
    raise EnvironmentError("❌ GCP credentials non trouvés ou GOOGLE_APPLICATION_CREDENTIALS non définie.")

app = FastAPI()

class PredictRequest(BaseModel):
    records: List[dict]
    model_type: str
    metric: str = "r2"

@app.post("/predict")
def predict(data: PredictRequest):
    summary_path = "gs://df_traffic_cyclist1/models/summary.json"
    model = get_best_model_from_summary(model_type=data.model_type, metric=data.metric, summary_path=summary_path, env="prod")
    df = pd.DataFrame(data.records)

    try:
        if data.model_type == "rf_class":
            y_pred = model.predict(df)
            return {"predictions": y_pred.tolist(), "label": ["faible" if p == 0 else "forte" for p in y_pred]}
        else:
            y_pred = model.predict_clean(df)
            return {"predictions": y_pred.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

