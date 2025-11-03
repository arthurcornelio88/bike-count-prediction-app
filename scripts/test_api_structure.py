#!/usr/bin/env python3
"""
Test script to analyze Paris Open Data API structure
Compare with historical data (2024) to ensure compatibility
"""

import requests
import pandas as pd


def test_api_structure():
    """Test the API and compare with historical data structure."""
    print("=" * 80)
    print("🔍 Testing Paris Open Data API Structure")
    print("=" * 80)

    # 1. Test API call
    api_url = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/comptage-velo-donnees-compteurs/records"

    params = {
        "limit": 100,  # Small sample
        "offset": 0,
    }

    print(f"\n🌐 Calling API: {api_url}")
    print(f"📋 Params: {params}")

    try:
        response = requests.get(api_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"❌ API Error: {e}")
        return

    print(f"\n✅ API Response: {response.status_code}")
    print(f"📊 Total available records: {data.get('total_count', 'N/A'):,}")
    print(f"📊 Results returned: {len(data.get('results', []))}")

    # 2. Extract records (API v2 structure - no nested 'fields')
    if "results" not in data or len(data["results"]) == 0:
        print("❌ No data returned from API")
        return

    # API v2 returns flat records directly
    records = data["results"]
    df_api = pd.DataFrame(records)

    print(f"\n📊 API Data Shape: {df_api.shape}")
    print(f"\n📋 API Columns ({len(df_api.columns)}):")
    for col in sorted(df_api.columns):
        print(f"   - {col}")

    # 3. Show sample data
    print("\n📄 Sample API Data (first 2 rows):")
    print(df_api.head(2).T)

    # 4. Check date range
    if "date" in df_api.columns:
        dates = pd.to_datetime(df_api["date"])
        print("\n📅 Date Range in API:")
        print(f"   - Min: {dates.min()}")
        print(f"   - Max: {dates.max()}")
        print(f"   - Unique dates: {dates.nunique()}")
        print(f"   - Year coverage: {dates.dt.year.unique()}")

    # Check what 'sum_counts' contains
    if "sum_counts" in df_api.columns:
        print("\n📊 sum_counts (comptage) stats:")
        print(f"   - Min: {df_api['sum_counts'].min()}")
        print(f"   - Max: {df_api['sum_counts'].max()}")
        print(f"   - Mean: {df_api['sum_counts'].mean():.2f}")
        print(f"   - Non-zero: {(df_api['sum_counts'] > 0).sum()} / {len(df_api)}")

    # 5. Load historical data (2024)
    print("\n" + "=" * 80)
    print("📚 Loading Historical Data (2024)")
    print("=" * 80)

    try:
        df_historical = pd.read_csv("data/reference_data.csv", nrows=1000, sep=";")
        print(f"\n📊 Historical Data Shape: {df_historical.shape}")
        print(f"\n📋 Historical Columns ({len(df_historical.columns)}):")
        for col in sorted(df_historical.columns):
            print(f"   - {col}")

        # 6. Compare columns
        print("\n" + "=" * 80)
        print("🔍 COMPARISON: API vs Historical Data")
        print("=" * 80)

        # Map API columns to historical columns
        api_to_hist_mapping = {
            "sum_counts": "comptage_horaire",
            "date": "date_et_heure_de_comptage",
            "id_compteur": "identifiant_du_compteur",
            "nom_compteur": "nom_du_compteur",
            "coordinates": "coordonnees_geographiques",
        }

        print("\n🔄 Column Mapping (API → Historical):")
        for api_col, hist_col in api_to_hist_mapping.items():
            api_exists = api_col in df_api.columns
            hist_exists = hist_col in df_historical.columns
            status = "✅" if (api_exists and hist_exists) else "❌"
            print(
                f"   {status} {api_col} → {hist_col}: API={api_exists}, Hist={hist_exists}"
            )

        api_cols = set(df_api.columns)
        hist_cols = set(df_historical.columns)

        common_cols = api_cols & hist_cols
        only_api = api_cols - hist_cols
        only_hist = hist_cols - api_cols

        print(f"\n✅ Exact common columns ({len(common_cols)}):")
        for col in sorted(common_cols):
            print(f"   - {col}")

        print(f"\n⚠️  Only in API ({len(only_api)}):")
        for col in sorted(only_api):
            print(f"   - {col}")

        print(f"\n⚠️  Only in Historical ({len(only_hist)}):")
        for col in sorted(only_hist):
            print(f"   - {col}")

        # 7. Check critical columns
        print("\n" + "=" * 80)
        print("🎯 Critical Columns Check (with mapping)")
        print("=" * 80)

        critical_mapping = {
            # API column : Historical column
            "sum_counts": "comptage_horaire",
            "date": "date_et_heure_de_comptage",
            "id_compteur": "identifiant_du_compteur",
            "nom_compteur": "nom_du_compteur",
            "coordinates": "coordonnees_geographiques",
        }

        print("\nRequired for model (API → Historical):")
        for api_col, hist_col in critical_mapping.items():
            in_api = api_col in api_cols
            in_hist = hist_col in hist_cols
            status = "✅" if (in_api and in_hist) else "❌"
            print(f"   {status} {api_col} → {hist_col}: API={in_api}, Hist={in_hist}")

        # 8. Data sample comparison
        print("\n" + "=" * 80)
        print("� Data Sample Comparison")
        print("=" * 80)

        print("\n🆕 API Sample (first 3 rows, key columns):")
        api_sample_cols = [
            c
            for c in ["id_compteur", "nom_compteur", "sum_counts", "date"]
            if c in df_api.columns
        ]
        if api_sample_cols:
            print(df_api[api_sample_cols].head(3).to_string())

        print("\n📚 Historical Sample (first 3 rows, key columns):")
        hist_sample_cols = [
            c
            for c in [
                "identifiant_du_compteur",
                "nom_du_compteur",
                "comptage_horaire",
                "date_et_heure_de_comptage",
            ]
            if c in df_historical.columns
        ]
        if hist_sample_cols:
            print(df_historical[hist_sample_cols].head(3).to_string())

    except FileNotFoundError:
        print("❌ reference_data.csv not found")
    except Exception as e:
        print(f"❌ Error loading historical data: {e}")

    # 9. Recommendations
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS")
    print("=" * 80)

    print("""
1. ✅ API is accessible and returns data
2. 🔍 Compare column structures above
3. ⚠️  If columns differ, DAG needs mapping logic
4. 💾 Storage options:

   Option A: BigQuery (current approach)
   - ✅ Queryable, SQL analytics
   - ✅ Auto-scaling
   - ⚠️  Cost: ~$0.02/GB storage + ~$5/TB queries
   - 💰 934MB = ~$0.02/month storage

   Option B: GCS CSV only
   - ✅ Cheaper storage (~$0.02/GB/month = $0.02/month)
   - ❌ No SQL queries
   - ❌ Need to download for analysis

   Option C: Hybrid
   - Keep reference_data.csv in GCS for drift detection
   - Store only daily_YYYYMMDD in BQ (small tables)
   - ✅ Best of both worlds

5. 📅 API Freshness:
   - Check if API has 2024 historical data
   - Or only recent data (2025+)
   - This affects baseline compatibility
""")


if __name__ == "__main__":
    test_api_structure()
