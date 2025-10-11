"""
Test du DAG fetch_bike_data avec les paramètres actuels
Simule l'exécution locale pour voir les données retournées
"""

import requests
import pandas as pd

print("=" * 80)
print("🧪 TEST DAG FETCH - Paramètres actuels du DAG")
print("=" * 80)

# URL et paramètres EXACTS du DAG actuel
api_url = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/comptage-velo-donnees-compteurs/records"

params = {
    "limit": 100,  # Max autorisé par l'API
    "order_by": "date DESC",  # API v2.1 utilise 'date' pas 'date_et_heure_de_comptage'
    "timezone": "Europe/Paris",
}

print(f"\n📍 API URL: {api_url}")
print("📋 Paramètres:")
for key, value in params.items():
    print(f"   - {key}: {value}")

# Appel API
print("\n🌐 Appel API en cours...")
response = requests.get(api_url, params=params, timeout=30)

print(f"📊 Status code: {response.status_code}")

if response.status_code != 200:
    print(f"❌ Erreur API: {response.text}")
    exit(1)

data = response.json()

print("\n📦 Structure de la réponse:")
print(f"   - Clés disponibles: {list(data.keys())}")
print(f"   - Nombre de résultats: {data.get('total_count', 'N/A')}")
print(f"   - Résultats retournés: {len(data.get('results', []))}")

# Extraction des records (logique du DAG)
records = []
for record in data.get("results", []):
    if "fields" in record:
        records.append(record["fields"])
    else:
        records.append(record)

df = pd.DataFrame(records)

print("\n📊 DataFrame brut:")
print(f"   - Nombre de lignes: {len(df)}")
print(f"   - Colonnes: {df.columns.tolist()}")

# Mapping des colonnes (logique du DAG)
column_mapping = {
    "comptage_horaire": "comptage_horaire",
    "date_et_heure_de_comptage": "date_et_heure_de_comptage",
    "identifiant_du_compteur": "identifiant_du_compteur",
    "nom_du_compteur": "nom_du_compteur",
    "coordonnees_geographiques": "coordonnees_geographiques",
    "id_compteur": "identifiant_du_compteur",  # Alternative name
    "sum_counts": "comptage_horaire",  # Alternative name
}

# Rename columns if they exist
df_renamed = df.copy()
for old_name, new_name in column_mapping.items():
    if old_name in df_renamed.columns and old_name != new_name:
        df_renamed.rename(columns={old_name: new_name}, inplace=True)

print("\n📊 Après renommage:")
print(f"   - Colonnes: {df_renamed.columns.tolist()}")

# Sélection des colonnes requises
required_columns = [
    "comptage_horaire",
    "date_et_heure_de_comptage",
    "identifiant_du_compteur",
    "nom_du_compteur",
    "coordonnees_geographiques",
]

available_columns = [col for col in required_columns if col in df_renamed.columns]
missing_columns = [col for col in required_columns if col not in df_renamed.columns]

print(f"\n📋 Colonnes disponibles: {available_columns}")
print(f"⚠️  Colonnes manquantes: {missing_columns}")

if available_columns:
    df_clean = df_renamed[available_columns].copy()

    print("\n📊 Dataset final:")
    print(f"   - Nombre de lignes: {len(df_clean)}")
    print(f"   - Colonnes: {list(df_clean.columns)}")

    # Analyse des données
    print("\n🔍 ANALYSE DES DONNÉES:")

    if "comptage_horaire" in df_clean.columns:
        print("\n🚴 Statistiques comptage_horaire:")
        print(f"   - Valeurs nulles: {df_clean['comptage_horaire'].isna().sum()}")
        print(f"   - Valeurs = 0: {(df_clean['comptage_horaire'] == 0).sum()}")
        print(f"   - Valeurs > 0: {(df_clean['comptage_horaire'] > 0).sum()}")
        if (df_clean["comptage_horaire"] > 0).sum() > 0:
            valid_counts = df_clean[df_clean["comptage_horaire"] > 0][
                "comptage_horaire"
            ]
            print(f"   - Min (>0): {valid_counts.min()}")
            print(f"   - Max: {valid_counts.max()}")
            print(f"   - Moyenne: {valid_counts.mean():.2f}")

    if "date_et_heure_de_comptage" in df_clean.columns:
        print("\n📅 Période des données:")
        dates = pd.to_datetime(df_clean["date_et_heure_de_comptage"])
        print(f"   - Date min: {dates.min()}")
        print(f"   - Date max: {dates.max()}")
        print(f"   - Étendue: {(dates.max() - dates.min()).days} jours")

    # Échantillon de données
    print("\n📋 ÉCHANTILLON (5 premières lignes):")
    print(df_clean.head())

    # Échantillon avec comptage > 0
    if "comptage_horaire" in df_clean.columns:
        df_valid = df_clean[df_clean["comptage_horaire"] > 0]
        if len(df_valid) > 0:
            print("\n📋 ÉCHANTILLON avec comptage > 0 (5 premières lignes):")
            print(df_valid.head())
        else:
            print("\n⚠️  AUCUNE ligne avec comptage > 0 !")

    # Distribution des compteurs
    if "identifiant_du_compteur" in df_clean.columns:
        print(
            f"\n🚦 Compteurs uniques: {df_clean['identifiant_du_compteur'].nunique()}"
        )

else:
    print("\n❌ AUCUNE colonne disponible dans le mapping!")
    print("\n📊 Colonnes brutes disponibles:")
    print(df.columns.tolist())
    print("\n📋 Échantillon de données brutes:")
    print(df.head())

print("\n" + "=" * 80)
print("✅ Test terminé")
print("=" * 80)
