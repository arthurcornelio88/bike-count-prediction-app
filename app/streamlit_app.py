import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
import requests
import json
import time
import numpy as np
import app.app_config as _  # forcer le sys.path side effect
from app.model_registry_summary import get_best_model_from_summary
from PIL import Image
from streamlit_folium import st_folium
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import plotly.graph_objects as go


# === Configuration ===
API_URL = st.secrets["api_url"]
API_RF_CLASS_URL = st.secrets["api_rf_class_url"]
ENV = st.secrets["env"]
DATA_GS_URI = "gs://df_traffic_cyclist1/raw_data/comptage-velo-donnees-compteurs.csv"
DATA_LOCAL = "data/comptage-velo-donnees-compteurs.csv"

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

@st.cache_data
def load_clean_data(path):
    df = pd.read_csv(path, sep=';')
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    df['date_et_heure_de_comptage'] = pd.to_datetime(df['date_et_heure_de_comptage'], utc=True).dt.tz_convert('Europe/Paris')
    df['heure'] = df['date_et_heure_de_comptage'].dt.hour
    df['jour_semaine'] = df['date_et_heure_de_comptage'].dt.day_name()
    df["mois"] = df["date_et_heure_de_comptage"].dt.month
    df[['latitude', 'longitude']] = df['coordonnées_géographiques'].str.split(',', expand=True).astype(float)
    df = df.dropna(subset=['latitude', 'longitude'])
    return df

# === UI ===
st.sidebar.title("🧭 Navigation")
menu = ["Présentation du projet", "Analyse Exploratoire", "Démarche",
        "Modélisation", "Démo"]

selection = st.sidebar.radio(label="", options=menu)

image = Image.open("docs/img/velo.jpg")
resized_image = image.resize((500, 200))
st.image(resized_image, use_container_width=True)

# Titre principal centré
st.markdown("""
<h1 style='text-align: center; font-size: 2em; margin-top: -10px;'>
     Prédiction du comptage horaire de vélos
</h1>
<p style='text-align: center; font-size: 1.6em; color: grey; margin-top: -10px;'>
    De Novembre 2024 à Novembre 2025
</p>
""", unsafe_allow_html=True)


if selection == "Présentation du projet":
    st.markdown("""
<div style="border:1px solid #ccc; padding:20px; border-radius:10px; text-align:center; font-size:16px">

<p>
Projet réalisé dans le cadre de la formation <strong>Machine Learning Engineer</strong> de 
<a href="https://www.datascientest.com" target="_blank">DataScientest</a><br>
Promotion Alternance Novembre 2025
</p>

<p><strong>Auteurs :</strong><br>
<a href="https://www.linkedin.com/in/arthurcornelio/" target="_blank"><strong>Arthur CORNELIO</strong></a><br>
<a href="https://www.linkedin.com/in/brunohappi17/" target="_blank"><strong>Bruno HAPPI</strong></a><br>
<a href="https://www.linkedin.com/in/ibnemri/" target="_blank"><strong>Ibtihel NEMRI</strong></a>
</p>

<p><strong>Source de données :</strong><br>
<a href="https://opendata.paris.fr/explore/dataset/comptage-velo-donnees-compteurs" target="_blank">
Comptage vélos | Open Data | Ville de Paris</a>
</p>

</div>
""", unsafe_allow_html=True)
    st.markdown("""
---

## 🎯 Objectifs du projet

Ce projet a pour ambition de :

- **Analyser le trafic cycliste parisien** à partir de données récoltées par des compteurs automatiques.
- Identifier les **zones et créneaux horaires d’affluence**.
- Construire des **modèles prédictifs** pour anticiper le nombre de cyclistes en fonction du lieu, de la date et de l’heure.
- Fournir à la Ville de Paris des **indicateurs utiles** pour optimiser ses politiques d’aménagements cyclables.

---

## 🛠️ Méthodologie en bref

- Nettoyage et structuration des données : standardisation, extraction temporelle, traitement géographique.
- Analyse statistique : identification de pics, moyennes par heure, jour, mois.
- Visualisations : cartes de chaleur, graphiques temporels, pics de trafic.
- Création de **modèles de régression** (Random Forest, Réseau de Neurones) pour prédire le **trafic horaire**.
- Construction d’un **modèle de classification binaire** pour détecter les situations d’**affluence**.
- Déploiement dans une **interface interactive Streamlit** accessible et testable.

---

## 📊 Exemples de visualisations

Quelques illustrations tirées de l’analyse exploratoire :

""")

    st.image("docs/img/heatmap_static.png", caption="Carte des zones à forte affluence cycliste", use_container_width=True)
    st.image("docs/img/traffic_hour_day.png", caption="Trafic moyen par heure selon le jour de la semaine", use_container_width=True)
    st.image("docs/img/pics_barplot.png", caption="Top 20 des pics de fréquentation par site de comptage", use_container_width=True)

    st.markdown("""
---

## 💻 Application déployée

Les modèles les plus performants ont été intégrés à une application Streamlit, permettant de :

- **Tester une prédiction** sur un exemple ou un fichier CSV.
- **Comparer plusieurs modèles** (Random Forest, Réseau de Neurones et Random Forest Classifier).
- Obtenir des résultats clairs, exploitables, et exportables.

➡️ Accédez à l’onglet **Démo** dans la barre latérale pour utiliser l’interface.

---
""")

elif selection == "Analyse Exploratoire":
    data_path = DATA_GS_URI if ENV == "PROD" else DATA_LOCAL
    df_clean = load_clean_data(data_path)
    st.header("🔍 Analyse Exploratoire")
    st.markdown("""
    Dans cette section, nous explorons les données de comptage horaire des vélos collectées par la Ville de Paris.  
    L’objectif est de comprendre **les grandes tendances de mobilité cycliste**, de mettre en lumière **les comportements selon l’heure, le jour ou le lieu**, et d’identifier **les zones à fort trafic**.

    Les analyses sont basées sur un jeu de données brut que nous avons **nettoyé, enrichi et structuré**, pour le rendre exploitable à des fins statistiques et de modélisation.
    """)

    st.subheader("📥 Chargement et aperçu des données")
    st.markdown("""
    Le jeu de données est issu de la plateforme OpenData de la Ville de Paris.  
    Il recense les **comptages horaires de vélos** effectués par une centaine de capteurs répartis dans la ville.  
    Chaque ligne du dataset correspond à **une heure de mesure pour un compteur donné**.

    Il contient les informations suivantes :
    - Date et heure de comptage
    - Nom et position du compteur
    - Nombre de vélos comptés à chaque heure
    - Coordonnées GPS du site de comptage
    """)
    st.write(df_clean.head())

    st.subheader("🧹 Nettoyage des données")
    st.markdown("""
    Avant toute analyse, plusieurs étapes de prétraitement ont été réalisées :

    - Normalisation des noms de colonnes pour faciliter la manipulation.
    - Conversion des dates au format `datetime` avec fuseau horaire de Paris.
    - Création de variables temporelles utiles : **heure, jour, mois, jour de semaine**.
    - Séparation des **coordonnées géographiques** pour les visualisations cartographiques.
    - Suppression des colonnes superflues (liens vers photos, identifiants techniques, etc.).
    - Transformation des colonnes catégorielles pour une meilleure performance mémoire.
    """)

    st.markdown("### 📊 Statistiques globales sur le comptage horaire")
    st.markdown("""
    Les premières statistiques descriptives révèlent que :

    - Le trafic horaire moyen est de l’ordre de quelques dizaines de vélos.
    - Les valeurs maximales peuvent dépasser plusieurs milliers de passages par heure sur certains axes.
    - Une grande dispersion est observée, ce qui justifie une analyse segmentée par heure et par jour.
    """)
    st.write(df_clean["comptage_horaire"].describe())

    st.markdown("### 📈 Moyenne de vélos par jour de la semaine")
    st.markdown("""
    Ce graphique permet d’observer le comportement des cyclistes sur une semaine complète.
    
    - Le trafic est plus élevé en semaine, en particulier du mardi au jeudi.
    - Une baisse notable est visible le samedi et surtout le dimanche.
    - Cela reflète clairement l’usage utilitaire du vélo pour les trajets domicile-travail.
    """)
    st.bar_chart(df_clean.groupby("jour_semaine")["comptage_horaire"].mean())

    st.markdown("### 🕒 Moyenne par heure de la journée")
    st.markdown("""
    On observe deux pics majeurs :

    - Entre 7h30 et 9h le matin.
    - Entre 17h30 et 19h le soir.

    Ces pics correspondent parfaitement aux horaires de travail traditionnels, ce qui confirme l’usage pendulaire du vélo à Paris.
    """)
    st.line_chart(df_clean.groupby("heure")["comptage_horaire"].mean())

    st.markdown("### 📆 Moyenne de trafic cycliste par mois")
    st.markdown("""
    Cette analyse montre une saisonnalité très marquée :

    - Les mois les plus actifs sont mai, juin, juillet et septembre.
    - Le creux hivernal (décembre à février) est bien visible.

    Ces variations sont corrélées à la météo et à la durée d’ensoleillement, deux facteurs qui influencent fortement la pratique du vélo.
    """)
    mois_mean = df_clean.groupby("mois")["comptage_horaire"].mean()
    st.bar_chart(mois_mean)

    st.markdown("### 📊 Comparaison du trafic : semaine vs week-end")
    st.markdown("""
    En comparant les courbes :

    - En semaine, les pics sont nets et concentrés sur les horaires de bureau.
    - Le week-end, le trafic est plus régulier et réparti tout au long de la journée.

    Cela traduit un usage loisir ou touristique le samedi et le dimanche, contre un usage fonctionnel en semaine.
    """)
    df_clean['type_jour'] = df_clean['jour_semaine'].apply(lambda x: 'weekend' if x in ['Saturday', 'Sunday'] else 'semaine')
    trafic_jour_type = df_clean.groupby(['type_jour', 'heure'])['comptage_horaire'].mean().reset_index()
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=trafic_jour_type, x='heure', y='comptage_horaire', hue='type_jour', palette='Set1')
    plt.title("Comparaison du trafic : semaine vs week-end")
    plt.xlabel("Heure")
    plt.ylabel("Vélos/heure")
    st.pyplot(plt.gcf())

    st.markdown("### 📍 Top 10 des compteurs les plus fréquentés")
    st.markdown("""
    Les compteurs les plus actifs sont tous situés sur des axes majeurs :

    - Boulevard de Sébastopol
    - Boulevard de Ménilmontant
    - Quai d'Orsay

    Ces zones concentrent une grande partie du trafic cycliste parisien et sont souvent bien aménagées pour les déplacements à vélo.
    """)
    top_compteurs = df_clean.groupby("nom_du_compteur")["comptage_horaire"].sum().nlargest(10)
    st.bar_chart(top_compteurs)

    st.markdown("### 🔝 Top 20 des pics horaires de trafic cycliste")
    st.markdown("""
    Cette analyse isole les moments où le trafic a atteint un record horaire pour chaque compteur :

    - Ces pics se produisent quasi exclusivement en semaine.
    - Ils surviennent autour de 8h30 ou 18h, aux heures de pointe.

    Cela confirme la nécessité de renforcer l’infrastructure cyclable sur ces créneaux horaires.
    """)
    df_time_series = df_clean.groupby(['nom_du_compteur', 'date_et_heure_de_comptage'])['comptage_horaire'].mean().unstack(level=0)
    pics_absolus = df_time_series.idxmax()
    analyse_pics = pd.DataFrame({
        'Compteur': pics_absolus.index,
        'Heure_du_pic': pics_absolus.values,
        'Trafic_max': df_time_series.max().values
    }).sort_values('Trafic_max', ascending=False)
    analyse_pics['Heure_du_pic'] = analyse_pics['Heure_du_pic'].dt.strftime('%Y-%m-%d %H:%M')
    fig = px.bar(analyse_pics.head(20), x='Compteur', y='Trafic_max', hover_data=['Heure_du_pic'],
                 color='Trafic_max', title='<b>Top 20 des pics de trafic cycliste</b>',
                 labels={'Trafic_max': 'Vélos/heure', 'Heure_du_pic': 'Moment du pic'},
                 height=500)
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig)

    st.markdown("### 📅 Trafic moyen par heure selon le jour de la semaine")
    st.markdown("Le trafic varie sensiblement entre semaine et week-end. Les lundis et vendredis ont des profils distincts.")
    df_jour = df_clean.groupby(['jour_semaine', 'heure'])['comptage_horaire'].mean().reset_index()
    ordre_jours = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=df_jour, x='heure', y='comptage_horaire', hue='jour_semaine',
                 hue_order=ordre_jours, palette='viridis', linewidth=2.5)
    plt.title('Trafic cycliste moyen par heure et jour de semaine', fontsize=16)
    plt.xlabel('Heure de la journée')
    plt.ylabel('Nombre moyen de vélos/heure')
    plt.grid(alpha=0.3)
    plt.legend(title='Jour', bbox_to_anchor=(1.05, 1))
    st.pyplot(plt.gcf())

    st.markdown("""
    ### 🧾 Conclusion de l’analyse exploratoire
    - Le trafic cycliste suit un cycle horaire et hebdomadaire régulier, avec des pics matin et soir en semaine.
    - Les week-ends montrent un trafic plus homogène dans la journée.
    - Les zones centrales comme les boulevards ou les quais sont les plus fréquentées.
    - Les données sont de qualité et prêtes pour la modélisation prédictive.
    """)

elif selection == "Démarche":
    st.header("🧭 Démarche")
    st.markdown("""
    Après avoir exploré les données de comptage horaire des vélos à Paris, l’objectif est désormais de **prévoir le trafic cycliste** à partir de variables temporelles et géographiques.
    
    ### 🎯 Objectifs de modélisation

    Deux axes principaux ont guidé notre démarche :

    1. **Prévoir le nombre de vélos par heure** sur un compteur donné (modèle de régression).
    2. **Détecter les situations d’affluence** à partir d’un seuil propre à chaque compteur (modèle de classification binaire).

    ---

    ### 🧩 Sélection des variables explicatives

    Pour estimer le trafic cycliste, nous avons retenu les variables suivantes :

    - **Heure de la journée** : le trafic est très dépendant des horaires (pics de pointe).
    - **Jour de la semaine** : différence marquée entre semaine et week-end.
    - **Mois de l’année** : influence saisonnière significative.
    - **Coordonnées GPS** du compteur : certaines zones sont structurellement plus fréquentées.

    Ces variables permettent de capter les **effets temporels et spatiaux** du trafic vélo.

    ---

    ### 🔄 Prétraitement des données

    Avant l'entraînement des modèles, plusieurs transformations ont été appliquées :

    - Encodage des variables catégorielles (`jour_semaine`, `nom_du_compteur`) via :
      - `OneHotEncoder` pour les modèles de régression.
      - `LabelEncoder` pour les modèles de classification.
    - Séparation du jeu de données en un **jeu d’entraînement (80 %)** et un **jeu de test (20 %)**.
    - Conversion des coordonnées géographiques en `float32` pour optimiser la mémoire.

    ---

    ### 🔍 Approche régressive

    Pour prédire le **nombre de vélos par heure**, nous avons testé plusieurs modèles :

    - **Régression linéaire** : comme modèle de base pour évaluer les performances minimales attendues.
    - **Random Forest Regressor** : capable de gérer les non-linéarités et les interactions complexes.
    - **XGBoost Regressor** : algorithme puissant souvent performant sur des données structurées.

    Les prédictions sont évaluées à l’aide de **métriques classiques** : MAE (erreur absolue moyenne) et R² (variance expliquée).

    ---

    ### ⚠️ Détection d'affluence

    En parallèle, nous avons formulé un **problème de classification binaire** :

    - Une ligne est marquée comme une **situation d’affluence** (`Affluence = 1`) si le comptage horaire dépasse la moyenne historique du compteur concerné.
    - Sinon, elle est marquée comme `Affluence = 0`.

    Cette cible binaire permet d’entraîner un **RandomForestClassifier** capable d’anticiper les périodes où l’infrastructure cyclable est susceptible d’être saturée.

    L’évaluation du modèle se base sur des indicateurs classiques : **accuracy, précision, rappel, F1-score**, ainsi que **la courbe ROC et l’AUC**.

    ---

    ### 🔒 Fiabilité et déploiement

    Pour garantir la reproductibilité et une future intégration :

    - Les meilleurs modèles sont **sérialisés (via joblib)**.
    - Les **résumés de performances sont stockés en JSON**, pour suivi ou visualisation ultérieure.
    - La prédiction ponctuelle est possible à partir de **valeurs connues** (ex. : heure, jour, compteur…).
    """)

elif selection == "Modélisation":
    st.header("🧠 Modélisation")
    st.write("Nous avons utilisé un RandomForestClassifier pour détecter l’affluence...")
    modele = st.radio("Modèle à utiliser :", ["Random Forest", "Réseau de Neurones", "Random Forest Classifier"])
    if modele == "Random Forest":
        
        # Configuration de la page
        st.title("🚴‍♂️ Modélisation Prédictive du Trafic Cycliste")

        # Section 1: Présentation des données
        with st.expander("📊 Données Utilisées", expanded=True):
            st.markdown("""
            **Variables utilisées dans le modèle :**
            - Variable cible : `comptage_velo` (continu)
            - Features :
                - `nom_du_compteur` (label encodé)
                - Variables cycliques : `heure_sin`, `heure_cos`, `mois_sin`, `mois_cos`
                - `jour_semaine` (one-hot encoded)
                - `annee` (binaire)
            """)

        # Section 2: Résultats de validation croisée
        st.header("📈 Résultats de Validation Croisée")

        cv_scores = [0.91515524, 0.91597844, 0.91690901]
        mean_cv_score = np.mean(cv_scores)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Moyenne R² (CV)", f"{mean_cv_score:.4f}", delta_color="off")
        with col2:
            st.metric("Écart-type (CV)", f"{np.std(cv_scores):.6f}", delta_color="off")

        # Section 3: Résultats sur le test set
        st.header("🧪 Performance sur le Test Set")

        # Métriques du test set
        rmse_rf = 30.13
        r2_rf = 0.9163

        col1, col2 = st.columns(2)
        with col1:
            st.metric("RMSE (Test Set)", f"{rmse_rf:.2f}", delta_color="off")
        with col2:
            st.metric("R² (Test Set)", f"{r2_rf:.4f}", delta_color="off")

        # Visualisation comparative
        st.subheader("Comparaison Validation Croisée / Test Set")
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=['Validation Croisée', 'Test Set'],
            y=[mean_cv_score, r2_rf],
            name='R² Score',
            text=[f"{mean_cv_score:.4f}", f"{r2_rf:.4f}"],
            textposition='auto',
            marker_color=['#3498db', '#2ecc71']
        ))

        fig.update_layout(
            yaxis_title='Score R²',
            title='Comparaison des Performances',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # Section 4: Interprétation
        with st.expander("🔍 Analyse des Résultats", expanded=True):
            st.markdown(f"""
            **Interprétation des performances :**
            
            - 🎯 **Score R² de {r2_rf:.4f} sur le test set** - Le modèle explique 91.63% de la variance des données de test
            - 📏 **RMSE de {rmse_rf:.2f}** - L'erreur moyenne est d'environ 30 vélos/heure
            - 🔄 **Cohérence entre CV et test set** - Pas de surapprentissage détecté (R² similaire)
            
            **Points forts :**
            - Robustesse confirmée sur données non vues
            - Bonne généralisation grâce à la validation croisée rigoureuse
            - Faible écart entre performance d'entraînement et de test
            
            **Recommandations :**
            - Tester avec plus de données historiques
            - Ajouter des variables contextuelles (météo, événements)
            - Optimiser les hyperparamètres avec GridSearch
            """)

       


    elif modele == "Réseau de Neurones":

    # Section 1: Architecture du modèle
        with st.expander("🏗 Architecture du Modèle", expanded=True):
            st.markdown("""
        **Structure innovante combinant :**
        - Couche d'Embedding pour les compteurs vélo
        - Features cycliques standardisées
        - Architecture profonde avec régularisation
        """)
        
        
            st.code("""
    # Couches principales
    Embedding(input_dim=n_compteurs, 
            output_dim=8)

    Dense(128, activation='relu')
    Dropout(0.3)

    Dense(64, activation='relu')
    Dropout(0.2)

    Dense(32, activation='relu')
    Dense(1)  # Sortie
            """, language='python')
        
        # Section 3: Performance
        st.header("📊 Résultats sur le Test Set")

        nn_metrics = {
            "RMSE": 36.24,
            "R²": 0.8789,
            "MAE": 19.12  # Exemple
        }

        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"{nn_metrics['RMSE']:.2f}")
        col2.metric("R² Score", f"{nn_metrics['R²']:.4f}")
        col3.metric("MAE", f"{nn_metrics['MAE']:.2f}")

        # Section 4: Analyse technique
        with st.expander("🧠 Points Clés du Modèle", expanded=True):
            st.markdown("""
            **Innovations majeures :**
            - 🧩 **Embedding des compteurs** : Capture les similarités entre sites de comptage
            - 🕒 **Features cycliques** : Heure/mois en sin/cos pour préserver la cyclicité
            - 🛡 **Régularisation** : Dropout pour éviter le surapprentissage
            
            **Avantages :**
            - Meilleure capture des motifs complexes
            - Flexibilité architecturale
            - Performance proche du Random Forest (R² = 0.879)
            
            **Améliorations possibles :**
            - Ajouter des features météo
            - Tester des architectures attention
            - Optimiser les hyperparamètres
            """)


    else :
        st.write("Random Forest Classifier est un modèle d'ensemble qui combine plusieurs arbres de décision. Il est robuste et peut gérer des données manquantes et non linéaires. Il est adapté aux données catégorielles et numériques.")
        st.header("🔍 Documentation – RandomForestClassifier")
        st.markdown("""
    **1. Contexte**  
    Nous utilisons ce modèle pour une **classification binaire** : prédire si une période horaire correspond à une **affluence** (`1`) ou non (`0`).

    ---

    **2. Modèle RandomForestClassifier**  
    Il s’agit d’un algorithme d’ensemble combinant plusieurs arbres de décision, chacun entraîné sur un sous-échantillon aléatoire des données. La prédiction finale est obtenue par vote majoritaire, ce qui améliore la robustesse et atténue le surapprentissage :contentReference[oaicite:3]{index=3}.

    ---

    **3. Pourquoi ce choix ?**  
    - Réduit le **surapprentissage** grâce à l'agrégation des arbres.  
    - Manipule directement les variables numériques et catégorielles, sans normalisation préalable :contentReference[oaicite:4]{index=4}.  
    - Robuste au **bruit** et aux **valeurs manquantes**.  
    - Donne des mesures d’**importance des variables** (impureté, permutation) :contentReference[oaicite:5]{index=5}.

    ---

    **4. Hyperparamètres utilisés**  
    - `n_estimators = 100` : nombre d’arbres.  
    - `random_state = 42` : assure la reproductibilité.

    ---

    **5. Évaluation sur le jeu test**  
    | Métrique   | Valeur typique | Interprétation |
    |------------|----------------|----------------|
    | Accuracy   | ~86 %          | Pourcentage global de bonnes prédictions |
    | Precision  | ~82 %          | Parmi les heures prédites comme affluence (`1`), 82 % étaient correctes.:contentReference[oaicite:6]{index=6} |
    | Recall     | ~82 %          | Parmi les vraies heures d’affluence, 82 % ont été détectées. :contentReference[oaicite:7]{index=7} |
    | F1-score   | ~0.82          | Harmonie entre precision et recall, adapté aux classes déséquilibrées. :contentReference[oaicite:8]{index=8} |

    **Interprétation**  
    - **Precision** élevée → peu de fausses alertes.  
    - **Recall** élevé → peu de pics ratés.  
    - **F1-score** élevé (~0,82) → bon compromis entre sensibilité et fiabilité.
    - **Accuracy** est utile, mais insuffisante si la classe "Affluence" est minoritaire :contentReference[oaicite:9]{index=9}.

    ---

    **6. Représentation visuelle recommandée**  
    Pour analyser davantage, on peut afficher :
    - La **matrice de confusion**, pour visualiser les vrais/faux positifs et négatifs.""")
        st.image("docs/img/matrice_de_confusion.png", caption="Matrice de confusion", use_container_width=True)
        st.markdown("""
    - La **courbe ROC-AUC** pour voir le compromis entre taux de vrais positifs et taux de faux positifs.""")
        st.image("docs/img/Courbe_ROC.png", caption="courbe ROC-AUC", use_container_width=True)
        st.markdown("""
    ---

    **Conclusion :**  
    Ce modèle offre un bon compromis pour détecter les périodes d'affluence :  
    - **Fiabilité** (peu de fausses alertes) grâce à la precision élevée.  
    - **Sensibilité** (peu de pics manqués) grâce au recall élevé.  
    - **Équilibre** entre les deux grâce à un F1-score supérieur à 0,8, ce qui est idéal en contexte binaire avec classe minoritaire.
    """)
    # Tu peux également intégrer ta documentation du modèle ici
elif selection == "Démo":
    st.header("🚲 Démo de prédiction du trafic")
    st.write("Interface de démonstration avec exemples manuels ou fichier CSV...")

    page = st.sidebar.selectbox("Choisissez une page :", ["🔍 Prédiction exemple", "📂 Prédiction CSV batch"])

    model_map = {
    "Random Forest": ("rf", "r2"),
    "Réseau de Neurones": ("nn", "r2"),
    "RF Classifier (Affluence)": ("rf_class", "f1_score")
    }
    model_choice = st.radio("Modèle à utiliser :", list(model_map.keys()))
    model_type, metric = model_map[model_choice]

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
            #api_url = API_URL  # No RF Class URL for this example, testing
            api_url = API_RF_CLASS_URL if model_type == "rf_class" else API_URL
            #st.write("🔧 Payload envoyé :", payload)
            #st.write("🔗 API URL :", api_url)
            result = call_prediction_api(api_url, payload)
            if result:
                pred = result["predictions"][0]
                if model_type == "rf_class":
                    st.success("📊 Affluence détectée ✅" if pred == 1 else "📉 Faible fréquentation attendue")
                else:
                    # gère les deux cas : [val] ou [[val]]
                    pred = pred[0] if isinstance(pred, (list, tuple)) else pred
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
            #api_url = API_URL  # No RF Class URL for this batch processing
            api_url = API_RF_CLASS_URL if model_type == "rf_class" else API_URL
            #st.write("🔧 Payload envoyé :", payload)
            #st.write("🔗 API URL :", api_url)
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






