import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
import os
import json
import time
import numpy as np
import app.app_config as _  # forcer le sys.path side effect
from app.model_registry_summary import get_best_model_from_summary

def get_secret(key, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)

env = get_secret("env", "DEV")
gcp_raw = get_secret("gcp_service_account") or get_secret("GCP_SERVICE_ACCOUNT")

if gcp_raw and env == "PROD":
    gcp_dict = json.loads(gcp_raw) if isinstance(gcp_raw, str) else dict(gcp_raw)
    with open("/tmp/gcp.json", "w") as f:
        json.dump(gcp_dict, f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/gcp.json"
elif env == "DEV":
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./gcp.json"

# === Fonction de chargement dynamique des modèles via summary.json ===
def fetch_best_pipeline(model_type: str):
    print("📦 Chargement de get_best_model_from_summary depuis model_registry_summary.py")
    return get_best_model_from_summary(
        model_type=model_type,
        summary_path="gs://df_traffic_cyclist1/models/summary.json",
        metric="r2",
        env="prod",
    )

@st.cache_resource
def load_best_pipeline(model_type: str):
    return fetch_best_pipeline(model_type)

# Chargement des pipelines
rf_pipeline = load_best_pipeline("rf")
# st.write("✅ Random Forest chargé :", type(rf_pipeline)) # DEBUG
st.write("✅ Random Forest chargé !")

nn_pipeline = load_best_pipeline("nn")
# st.write("✅ Neural Net chargé !", type(nn_pipeline)) # DEBUG
st.write("✅ Neural Net chargé !")

# === UI ===
st.sidebar.title("🧭 Navigation")
page = st.sidebar.selectbox("Choisissez une page :", ["🔍 Prédiction exemple", "📂 Prédiction CSV batch"])
st.title("🚲 Prédiction du comptage horaire de vélos")
modele = st.radio("Modèle à utiliser :", ["Random Forest", "Réseau de Neurones"])

# Fonction utilitaire pour obtenir le bon pipeline
def get_pipeline(name: str):
    return rf_pipeline if name == "Random Forest" else nn_pipeline

# === Exemples manuels
raw_samples = [
    {
        'nom_du_compteur': '35 boulevard de Ménilmontant NO-SE',
        'date_et_heure_de_comptage': '2025-05-17 18:00:00+02:00',
        'coordonnées_géographiques': '48.8672, 2.3501',
        'mois_annee_comptage': 'mai 2025'
    },
    {
        'nom_du_compteur': 'Totem 73 boulevard de Sébastopol S-N',
        'date_et_heure_de_comptage': '2024-11-12 08:00:00+02:00',
        'coordonnées_géographiques': '48.8639, 2.3895',
        'mois_annee_comptage': 'novembre 2024'
    },
    {
        'nom_du_compteur': "Quai d'Orsay E-O",
        'date_et_heure_de_comptage': '2024-06-03 15:00:00+02:00',
        'coordonnées_géographiques': '48.8784, 2.3574',
        'mois_annee_comptage': 'juin 2024'
    }
]

# === Page 1 : prédiction manuelle
if page == "🔍 Prédiction exemple":
    idx = st.selectbox("Sélectionnez une observation brute à prédire :", range(len(raw_samples)), format_func=lambda i: f"Exemple {i+1}")
    sample_df = pd.DataFrame([raw_samples[idx]])
    pipeline = get_pipeline(modele)

    try:
        pred = pipeline.predict_clean(sample_df)[0]
        if isinstance(pred, (list, np.ndarray)) and isinstance(pred[0], (list, np.ndarray)):
            pred = pred[0]

        st.markdown("### 🔍 Observation sélectionnée")
        st.json(raw_samples[idx])
        st.success(f"🧾 Prédiction du comptage horaire : **{round(float(pred))} vélos**")

    except Exception as e:
        st.error("Erreur lors de la prédiction :")
        st.code(str(e))
        st.exception(e)

# === Page 2 : prédiction sur CSV
elif page == "📂 Prédiction CSV batch":
    st.header("Prédiction sur fichier CSV brut")
    uploaded_file = st.file_uploader("Chargez un fichier brut (.csv)", type="csv")

    if uploaded_file is not None:
        df_csv = pd.read_csv(uploaded_file)
        pipeline = get_pipeline(modele)
        try:
            predictions = pipeline.predict(df_csv)
            predictions = predictions.flatten() if hasattr(predictions, "flatten") else predictions
            df_csv['prediction_comptage_horaire'] = predictions.round().astype(int)

            st.markdown("✅ **Résultats :**")
            st.dataframe(df_csv.head(20))

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            file_name = f"predictions_{modele.lower().replace(' ', '_')}_{timestamp}.csv"
            csv_output = df_csv.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Télécharger les résultats", csv_output, file_name=file_name, mime="text/csv")

        except Exception as e:
            st.error("Erreur lors de la prédiction :")
            st.code(str(e))
            st.exception(e)
