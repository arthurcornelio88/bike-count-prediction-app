import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# === Configuration ===
API_URL = st.secrets["api_url"]
API_RF_CLASS_URL = st.secrets["api_rf_class_url"]
ENV = st.secrets["env"]

# === Authentification GCP ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
try:
    gcp_secret = st.secrets["gcp_service_account"]
    with open("/tmp/gcp.json", "w") as f:
        json.dump(dict(gcp_secret), f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/gcp.json"
    print("✅ GCP credentials configurés via Streamlit secrets")
except Exception as e:
    raise RuntimeError("❌ Credentials GCP manquants dans st.secrets.")

# === API Request Wrapper ===
def call_prediction_api(url: str, payload: dict, timeout: int = 60):
    try:
        with st.spinner("⏳ En attente de la réponse du modèle..."):
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()
    except requests.exceptions.Timeout:
        st.error("⏱️ L’API a mis trop de temps à répondre. Elle est peut-être en cold start.")
    except requests.exceptions.ConnectionError:
        st.error("🔌 Impossible de se connecter à l’API. Vérifier l’URL ou la disponibilité du backend.")
    except requests.exceptions.HTTPError as http_err:
        st.error(f"❌ Erreur HTTP {response.status_code} : {response.text}")
    except requests.exceptions.RequestException as req_err:
        st.error(f"⚠️ Erreur inattendue : {type(req_err).__name__} — {req_err}")
    return None

# === UI Setup ===
st.sidebar.title("🧭 Navigation")
page = st.sidebar.selectbox("Choisissez une page :", ["🔍 Prédiction exemple", "📂 Prédiction CSV batch"])
st.title("🚲 Prédiction du comptage horaire de vélos")

model_map = {
    "Random Forest": ("rf", "r2"),
    "Réseau de Neurones": ("nn", "r2"),
    "RF Classifier (Affluence)": ("rf_class", "f1_score")
}
model_choice = st.radio("Modèle à utiliser :", list(model_map.keys()))
model_type, metric = model_map[model_choice]

# === Exemples ===
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

# === Page Exemple ===
if page == "🔍 Prédiction exemple":
    idx = st.selectbox("Sélectionnez une observation :", range(len(raw_samples)), format_func=lambda i: f"Exemple {i+1}")
    selected = raw_samples[idx]
    st.markdown("### 🔍 Observation sélectionnée")
    st.json(selected)

    if st.button("🔮 Lancer la prédiction"):
        payload = {
            "records": [selected],
            "model_type": model_type,
            "metric": metric
        }
        api_url = API_RF_CLASS_URL if model_type == "rf_class" else API_URL
        st.write("🔧 Payload envoyé :", payload)
        st.write("🔗 API URL :", api_url)
        result = call_prediction_api(api_url, payload)
        if result:
            pred = result["predictions"][0]
            if model_type == "rf_class":
                st.success("📊 Affluence détectée ✅" if pred == 1 else "📉 Faible fréquentation attendue")
            else:
                st.success(f"🧾 Prédiction du comptage horaire : **{round(float(pred))} vélos**")
    
    with st.expander("🩺 Debug API"):
        if st.button("🔁 Forcer ping API"):
            try:
                ping_response = requests.get(API_URL.replace("/predict", "/docs"), timeout=10)
                if ping_response.status_code == 200:
                    st.success("✅ API en ligne (endpoint /docs accessible).")
                else:
                    st.warning(f"⚠️ API répond mais code inattendu : {ping_response.status_code}")
            except Exception as e:
                st.error(f"❌ API inaccessible : {e}")
        if st.button("🔄 Forcer /refresh_models"):
            try:
                refresh_url = API_URL.replace("/predict", "/refresh_models")
                refresh_response = requests.post(refresh_url, timeout=15)
                if refresh_response.status_code == 200:
                    st.success("✅ Modèles rechargés depuis /refresh_models.")
                    st.json(refresh_response.json())
                else:
                    st.warning(f"⚠️ Requête envoyée mais réponse inattendue : {refresh_response.status_code}")
            except Exception as e:
                st.error(f"❌ Échec du refresh : {e}")


# === Page CSV ===
elif page == "📂 Prédiction CSV batch":
    st.header("Prédiction sur fichier CSV brut")
    uploaded_file = st.file_uploader("Chargez un fichier brut (.csv)", type="csv")

    if uploaded_file is not None:
        df_csv = pd.read_csv(uploaded_file)
        payload = {
            "records": df_csv.to_dict(orient="records"),
            "model_type": model_type,
            "metric": metric
        }
        api_url = API_RF_CLASS_URL if model_type == "rf_class" else API_URL
        st.write("🔧 Payload envoyé :", payload)
        st.write("🔗 API URL :", api_url)
        result = call_prediction_api(api_url, payload)
        if result:
            predictions = result["predictions"]
            predictions = np.array(predictions).flatten()
            df_csv["prediction_comptage_horaire"] = predictions.round().astype(int) if model_type != "rf_class" else predictions.astype(int)

            st.markdown("✅ **Résultats :**")
            st.dataframe(df_csv.head(20))

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            file_name = f"predictions_{model_type}_{timestamp}.csv"
            csv_output = df_csv.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Télécharger les résultats", csv_output, file_name=file_name, mime="text/csv")
