#!/usr/bin/env python3
"""
Advanced API test with filters to get real data (non-zero counts)
"""

import requests
import pandas as pd
from datetime import datetime, timedelta


def test_api_with_filters():
    """Test API with various filters to find real data."""
    print("=" * 80)
    print("🔍 Testing Paris Open Data API with Filters")
    print("=" * 80)

    api_url = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/comptage-velo-donnees-compteurs/records"

    # Test 1: Get recent data with non-zero counts
    print("\n📊 Test 1: Recent data with non-zero counts")
    print("-" * 80)

    params = {
        "limit": 100,
        "where": "sum_counts > 0",  # Only non-zero counts
        "order_by": "date desc",
    }

    try:
        response = requests.get(api_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        print(f"✅ Total records with sum_counts > 0: {data.get('total_count', 0):,}")

        if data.get("results"):
            df = pd.DataFrame(data["results"])
            print(f"📊 Retrieved: {len(df)} records")

            # Date range
            if "date" in df.columns:
                dates = pd.to_datetime(df["date"])
                print("\n📅 Date Range:")
                print(f"   - Min: {dates.min()}")
                print(f"   - Max: {dates.max()}")
                print(f"   - Years: {sorted(dates.dt.year.unique())}")

            # Stats on sum_counts
            if "sum_counts" in df.columns:
                print("\n📊 sum_counts stats:")
                print(f"   - Min: {df['sum_counts'].min()}")
                print(f"   - Max: {df['sum_counts'].max()}")
                print(f"   - Mean: {df['sum_counts'].mean():.2f}")
                print(f"   - Median: {df['sum_counts'].median():.2f}")

            # Sample data
            print("\n📄 Sample (first 5 rows):")
            sample_cols = ["id_compteur", "nom_compteur", "sum_counts", "date"]
            print(df[sample_cols].head(5).to_string(index=False))

    except Exception as e:
        print(f"❌ Error: {e}")

    # Test 2: Get data from a specific date range (recent months)
    print("\n" + "=" * 80)
    print("📊 Test 2: Data from last 30 days")
    print("-" * 80)

    # Calculate date 30 days ago
    date_30_days_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    params = {
        "limit": 100,
        "where": f"date >= '{date_30_days_ago}' AND sum_counts > 0",
        "order_by": "date desc",
    }

    try:
        response = requests.get(api_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        print(
            f"✅ Total records (last 30 days, sum_counts > 0): {data.get('total_count', 0):,}"
        )

        if data.get("results"):
            df = pd.DataFrame(data["results"])
            print(f"📊 Retrieved: {len(df)} records")

            # Unique counters
            if "id_compteur" in df.columns:
                print(f"\n🚲 Unique counters: {df['id_compteur'].nunique()}")
                print("   Top 5 counters by total counts:")
                top_counters = (
                    df.groupby("nom_compteur")["sum_counts"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(5)
                )
                for name, total in top_counters.items():
                    print(f"   - {name}: {total:,.0f} vélos")

    except Exception as e:
        print(f"❌ Error: {e}")

    # Test 3: Check historical data availability (2024)
    print("\n" + "=" * 80)
    print("📊 Test 3: Historical data from 2024")
    print("-" * 80)

    params = {
        "limit": 100,
        "where": "date >= '2024-01-01' AND date < '2024-02-01' AND sum_counts > 0",
        "order_by": "date asc",
    }

    try:
        response = requests.get(api_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        print(
            f"✅ Total records (Jan 2024, sum_counts > 0): {data.get('total_count', 0):,}"
        )

        if data.get("results"):
            df = pd.DataFrame(data["results"])
            print(f"📊 Retrieved: {len(df)} records")

            if "date" in df.columns:
                dates = pd.to_datetime(df["date"])
                print("\n📅 Date Range (2024 data):")
                print(f"   - First: {dates.min()}")
                print(f"   - Last: {dates.max()}")
        else:
            print("⚠️  No results for Jan 2024 with non-zero counts")

    except Exception as e:
        print(f"❌ Error: {e}")

    # Test 4: Check total dataset info
    print("\n" + "=" * 80)
    print("📊 Test 4: Total dataset statistics")
    print("-" * 80)

    # Get just one record to see total_count
    params = {"limit": 1}

    try:
        response = requests.get(api_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        total = data.get("total_count", 0)
        print(f"📊 Total records in dataset: {total:,}")

        # Estimate storage
        avg_record_size = 500  # bytes (rough estimate)
        total_size_mb = (total * avg_record_size) / (1024 * 1024)
        print(f"💾 Estimated dataset size: ~{total_size_mb:.1f} MB")

    except Exception as e:
        print(f"❌ Error: {e}")

    # Recommendations
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS BASED ON API TESTS")
    print("=" * 80)

    print("""
1. 📅 Date Filtering:
   - Use: where="date >= 'YYYY-MM-DD' AND sum_counts > 0"
   - This filters out zero-count records
   - Ensures we only get meaningful data

2. 📊 Daily Incremental Fetch (DAG strategy):
   - Fetch only TODAY's data: where="date >= '{today}'"
   - Limit to 1000-5000 records per day (should be enough)
   - Store in BigQuery: bike_traffic_raw.daily_YYYYMMDD

3. 🔄 Column Mapping Required:
   API columns → Model columns:
   - sum_counts → comptage_horaire
   - date → date_et_heure_de_comptage
   - id_compteur → identifiant_du_compteur
   - nom_compteur → nom_du_compteur
   - coordinates → coordonnees_geographiques (need to extract from dict)

4. 💾 Storage Strategy:
   Option C (Hybrid) - RECOMMENDED:
   ✅ reference_data.csv (2024 historical) → GCS only
      - Used for drift detection baseline
      - 934MB, costs ~$0.02/month
      - Already trained on this format

   ✅ New API data (2024-present) → BigQuery daily tables
      - Small daily tables (few MB each)
      - Queryable for monitoring
      - Auto-cleanup after 90 days
      - Minimal cost

   ❌ DON'T load full 934MB into BigQuery
      - Not needed for daily operations
      - Would cost more
      - Drift detection works from GCS

5. 🚨 Data Quality Issues:
   - API has many zero-count records
   - Need to filter sum_counts > 0
   - Check for data completeness before prediction

6. 🔧 DAG Updates Needed:
   - Add column mapping logic
   - Add where filter for non-zero counts
   - Add date range filter (today only)
   - Validate data before writing to BQ
""")


if __name__ == "__main__":
    test_api_with_filters()
