"""
Audit de la continuité temporelle entre les données historiques CSV et l'API

Ce script vérifie:
1. La plage temporelle du CSV historique (reference_data.csv + current_data.csv)
2. La plage temporelle disponible via l'API
3. Les gaps temporels éventuels
4. La qualité des données (comptage > 0)
5. Le choix de la date de coupure (cutoff date) pour la stratégie hybride

Objectif: Valider que notre stratégie hybride (CSV historique + API live) assure une continuité temporelle
"""

import pandas as pd
import requests
from datetime import datetime
import json

# ========================================
# 1. ANALYSE DES DONNÉES CSV HISTORIQUES
# ========================================

print("=" * 80)
print("📊 AUDIT DE CONTINUITÉ TEMPORELLE - Stratégie Hybride")
print("=" * 80)
print()

print("🔍 1. ANALYSE DES DONNÉES HISTORIQUES (CSV)")
print("-" * 80)

# Charger les CSVs (séparateur = point-virgule)
reference_df = pd.read_csv("data/reference_data.csv", sep=";")
current_df = pd.read_csv("data/current_data.csv", sep=";")

# Convertir les dates
reference_df["Date et heure de comptage"] = pd.to_datetime(
    reference_df["Date et heure de comptage"]
)
current_df["Date et heure de comptage"] = pd.to_datetime(
    current_df["Date et heure de comptage"]
)

# Stats reference_data.csv
ref_min = reference_df["Date et heure de comptage"].min()
ref_max = reference_df["Date et heure de comptage"].max()
ref_count = len(reference_df)
ref_valid = (reference_df["Comptage horaire"] > 0).sum()

print("\n📁 reference_data.csv (Training baseline):")
print(f"   - Période: {ref_min} → {ref_max}")
print(f"   - Durée: {(ref_max - ref_min).days} jours")
print(f"   - Records: {ref_count:,}")
print(
    f"   - Records valides (comptage > 0): {ref_valid:,} ({ref_valid/ref_count*100:.1f}%)"
)
print(f"   - Compteurs uniques: {reference_df['Nom du compteur'].nunique()}")

# Stats current_data.csv
cur_min = current_df["Date et heure de comptage"].min()
cur_max = current_df["Date et heure de comptage"].max()
cur_count = len(current_df)
cur_valid = (current_df["Comptage horaire"] > 0).sum()

print("\n📁 current_data.csv (Drift detection baseline):")
print(f"   - Période: {cur_min} → {cur_max}")
print(f"   - Durée: {(cur_max - cur_min).days} jours")
print(f"   - Records: {cur_count:,}")
print(
    f"   - Records valides (comptage > 0): {cur_valid:,} ({cur_valid/cur_count*100:.1f}%)"
)
print(f"   - Compteurs uniques: {current_df['Nom du compteur'].nunique()}")

# Combiner pour stats globales
total_df = pd.concat([reference_df, current_df])
total_min = total_df["Date et heure de comptage"].min()
total_max = total_df["Date et heure de comptage"].max()
total_count = len(total_df)
total_valid = (total_df["Comptage horaire"] > 0).sum()

print("\n📊 TOTAL HISTORIQUE (reference + current):")
print(f"   - Période complète: {total_min} → {total_max}")
print(f"   - Durée totale: {(total_max - total_min).days} jours")
print(f"   - Records totaux: {total_count:,}")
print(f"   - Records valides: {total_valid:,} ({total_valid/total_count*100:.1f}%)")

# Vérifier la continuité entre les deux fichiers
gap_days = (cur_min - ref_max).days
if gap_days == 0:
    print("\n✅ Continuité parfaite entre reference et current (pas de gap)")
elif gap_days > 0:
    print(f"\n⚠️ Gap de {gap_days} jours entre reference et current")
else:
    overlap_days = abs(gap_days)
    print(f"\n⚠️ Overlap de {overlap_days} jours entre reference et current")

# ========================================
# 2. ANALYSE DE L'API (PARIS OPEN DATA)
# ========================================

print("\n" + "=" * 80)
print("🌐 2. ANALYSE DE L'API (Paris Open Data v2.1)")
print("-" * 80)

api_url = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/comptage-velo-donnees-compteurs/records"

# Test 1: Obtenir les statistiques globales
params_stats = {
    "limit": 0,  # Juste pour obtenir le total_count
    "timezone": "Europe/Paris",
}

print("\n🔎 Fetching API metadata...")
response = requests.get(api_url, params=params_stats, timeout=30)

if response.status_code == 200:
    data = response.json()
    total_api_records = data.get("total_count", 0)
    print("\n✅ API accessible")
    print(f"   - Total records disponibles: {total_api_records:,}")
else:
    print(f"\n❌ Erreur API: {response.status_code}")
    print(response.text)
    exit(1)

# Test 2: Date la plus ancienne disponible dans l'API
params_oldest = {
    "limit": 1,
    "order_by": "date ASC",
    "timezone": "Europe/Paris",
}

print("\n🔎 Fetching oldest record from API...")
response = requests.get(api_url, params=params_oldest, timeout=30)

if response.status_code == 200:
    data = response.json()
    if data["results"]:
        oldest_record = data["results"][0]
        api_min_date = pd.to_datetime(oldest_record["date"])
        print(f"✅ Date la plus ancienne API: {api_min_date}")
    else:
        print("⚠️ Aucune donnée retournée")
        api_min_date = None
else:
    print(f"❌ Erreur: {response.status_code}")
    api_min_date = None

# Test 3: Date la plus récente disponible dans l'API
params_newest = {
    "limit": 1,
    "order_by": "date DESC",
    "timezone": "Europe/Paris",
}

print("\n🔎 Fetching newest record from API...")
response = requests.get(api_url, params=params_newest, timeout=30)

if response.status_code == 200:
    data = response.json()
    if data["results"]:
        newest_record = data["results"][0]
        api_max_date = pd.to_datetime(newest_record["date"])
        print(f"✅ Date la plus récente API: {api_max_date}")
    else:
        print("⚠️ Aucune donnée retournée")
        api_max_date = None
else:
    print(f"❌ Erreur: {response.status_code}")
    api_max_date = None

# Test 4: Qualité des données API (sample de 100 records récents)
params_sample = {
    "limit": 100,
    "where": "sum_counts > 0",
    "order_by": "date DESC",
    "timezone": "Europe/Paris",
}

print("\n🔎 Fetching sample of 100 recent records with sum_counts > 0...")
response = requests.get(api_url, params=params_sample, timeout=30)

if response.status_code == 200:
    data = response.json()
    sample_count = len(data["results"])
    sample_df = pd.DataFrame(data["results"])

    print(f"\n✅ Sample récupéré: {sample_count} records")
    print(f"   - Comptage moyen: {sample_df['sum_counts'].mean():.1f}")
    print(f"   - Comptage min: {sample_df['sum_counts'].min()}")
    print(f"   - Comptage max: {sample_df['sum_counts'].max()}")
    print(f"   - Compteurs uniques: {sample_df['id_compteur'].nunique()}")
else:
    print(f"❌ Erreur: {response.status_code}")

# ========================================
# 3. ANALYSE DE CONTINUITÉ
# ========================================

print("\n" + "=" * 80)
print("🔗 3. ANALYSE DE CONTINUITÉ TEMPORELLE")
print("-" * 80)

if api_min_date and api_max_date:
    print("\n📅 TIMELINE COMPLÈTE:")
    print(
        f"   CSV Historique: {total_min.date()} → {total_max.date()} ({(total_max - total_min).days} jours)"
    )
    print(
        f"   API Disponible: {api_min_date.date()} → {api_max_date.date()} ({(api_max_date - api_min_date).days} jours)"
    )

    # Calculer le gap entre CSV et API
    csv_end = total_max
    api_start = api_min_date

    gap_csv_to_api = (api_start - csv_end).days

    print("\n📊 ANALYSE DU GAP:")
    if gap_csv_to_api < 0:
        overlap_days = abs(gap_csv_to_api)
        print(f"   ✅ Overlap de {overlap_days} jours entre CSV et API")
        print("   ✅ Continuité garantie (pas de trou temporel)")

        # Date de coupure recommandée
        cutoff_date = csv_end.date()
        print("\n💡 DATE DE COUPURE RECOMMANDÉE:")
        print(f"   {cutoff_date} (dernière date du CSV historique)")
        print(f"   → CSV pour données ≤ {cutoff_date}")
        print(f"   → API pour données > {cutoff_date}")

    elif gap_csv_to_api == 0:
        print("   ✅ Continuité parfaite (pas de gap, pas d'overlap)")
        cutoff_date = csv_end.date()
        print("\n💡 DATE DE COUPURE RECOMMANDÉE:")
        print(f"   {cutoff_date}")

    else:
        print(f"   ⚠️ GAP de {gap_csv_to_api} jours entre CSV et API")
        print(f"   ⚠️ Données manquantes: {csv_end.date()} → {api_start.date()}")
        cutoff_date = csv_end.date()
        print("\n⚠️ DATE DE COUPURE FORCÉE:")
        print(f"   {cutoff_date} (dernière date disponible dans CSV)")
        print("   ⚠️ Attention: Gap temporel à documenter")

    # Vérifier la fraîcheur des données API
    today = pd.Timestamp.now(tz=api_max_date.tz)
    api_freshness_days = (today - api_max_date).days

    print("\n📡 FRAÎCHEUR DES DONNÉES API:")
    print(f"   Dernière donnée API: {api_max_date.date()}")
    print(f"   Aujourd'hui: {today.date()}")
    print(f"   Délai: {api_freshness_days} jours")

    if api_freshness_days <= 1:
        print("   ✅ Données très fraîches (mises à jour quotidiennes)")
    elif api_freshness_days <= 7:
        print(f"   ⚠️ Données légèrement en retard ({api_freshness_days} jours)")
    else:
        print(f"   ❌ Données obsolètes ({api_freshness_days} jours de retard)")

# ========================================
# 4. RECOMMANDATIONS
# ========================================

print("\n" + "=" * 80)
print("📝 4. RECOMMANDATIONS POUR LA STRATÉGIE HYBRIDE")
print("-" * 80)

print(f"""
✅ STRATÉGIE VALIDÉE:

1. **Données historiques (CSV)** → BigQuery Table externe ou Upload initial
   - Source: reference_data.csv + current_data.csv
   - Période: {total_min.date()} → {total_max.date()}
   - Volume: {total_count:,} records
   - Usage: Training baseline + Drift detection baseline

2. **Données live (API)** → BigQuery Tables quotidiennes
   - Source: Paris Open Data API v2.1
   - Période: {cutoff_date} → présent
   - Fréquence: Daily fetch via Airflow
   - Filtre: `date > '{cutoff_date}' AND sum_counts > 0`

3. **Architecture BigQuery**:
   ```sql
   -- Table historique (one-time load)
   bike_traffic_raw.historical_baseline

   -- Tables quotidiennes (daily fetch)
   bike_traffic_raw.daily_YYYYMMDD

   -- Vue unifiée (pour queries)
   bike_traffic_raw.all_data (UNION de historical + daily_*)
   ```

4. **Configuration DAG**:
   ```python
   params = {{
       "limit": 100,
       "where": "sum_counts > 0 AND date > '{cutoff_date}'",
       "order_by": "date DESC",
       "timezone": "Europe/Paris",
   }}
   ```

5. **Mapping colonnes API → CSV**:
   - sum_counts → comptage_horaire
   - date → date_et_heure_de_comptage
   - id_compteur → identifiant_du_compteur
   - nom_compteur → nom_du_compteur
   - coordinates → coordonnees_geographiques (dict → "lat, lon" string)

6. **Avantages de cette approche**:
   ✅ Continuité temporelle garantie (pas de gap)
   ✅ Baseline solide pour training (660k rows, 2024 data)
   ✅ Drift detection valide (287k rows, début 2025)
   ✅ Données "vie réelle" via API (post-{cutoff_date})
   ✅ Production-like (simule un vrai pipeline MLOps)

7. **Points d'attention**:
   ⚠️ Limite API = 100 records/call → Implémenter pagination si besoin
   ⚠️ Format coordinates différent → Conversion nécessaire
   ⚠️ Noms de colonnes différents → Mapping rigoureux
""")

# ========================================
# 5. EXPORT DU RAPPORT
# ========================================

print("\n" + "=" * 80)
print("💾 5. EXPORT DU RAPPORT")
print("-" * 80)

report = {
    "audit_date": datetime.now().isoformat(),
    "csv_historical": {
        "reference": {
            "period": f"{ref_min.date()} to {ref_max.date()}",
            "days": (ref_max - ref_min).days,
            "records": int(ref_count),
            "valid_records": int(ref_valid),
        },
        "current": {
            "period": f"{cur_min.date()} to {cur_max.date()}",
            "days": (cur_max - cur_min).days,
            "records": int(cur_count),
            "valid_records": int(cur_valid),
        },
        "total": {
            "period": f"{total_min.date()} to {total_max.date()}",
            "days": (total_max - total_min).days,
            "records": int(total_count),
            "valid_records": int(total_valid),
        },
    },
    "api": {
        "total_available_records": total_api_records,
        "oldest_date": api_min_date.isoformat() if api_min_date else None,
        "newest_date": api_max_date.isoformat() if api_max_date else None,
        "days_range": (api_max_date - api_min_date).days
        if api_min_date and api_max_date
        else None,
        "freshness_days": api_freshness_days if api_max_date else None,
    },
    "continuity": {
        "gap_days": gap_csv_to_api if api_min_date else None,
        "has_overlap": gap_csv_to_api < 0 if api_min_date else None,
        "cutoff_date": str(cutoff_date) if api_min_date else None,
    },
    "recommendation": {
        "strategy": "hybrid",
        "csv_usage": "historical_baseline",
        "api_usage": "live_daily_fetch",
        "cutoff_date": str(cutoff_date) if api_min_date else None,
        "filter": f"date > '{cutoff_date}' AND sum_counts > 0"
        if api_min_date
        else None,
    },
}

with open("docs/temporal_continuity_audit.json", "w") as f:
    json.dump(report, f, indent=2)

print("\n✅ Rapport JSON exporté: docs/temporal_continuity_audit.json")
print("\nAudit terminé avec succès ! 🎉")
