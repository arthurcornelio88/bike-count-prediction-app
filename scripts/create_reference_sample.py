"""
Create a sampled version of reference_data.csv for drift detection.

This script:
1. Loads the full reference_data.csv (~1GB)
2. Takes a stratified sample of 10k rows
3. Uploads to GCS: gs://df_traffic_cyclist1/data/reference_data_sample.csv

Usage:
    python scripts/create_reference_sample.py
"""

import pandas as pd
from google.cloud import storage
import os
import sys

# Configuration
REFERENCE_DATA_PATH = "data/reference_data.csv"
SAMPLE_SIZE = 10000
GCS_BUCKET = "df_traffic_cyclist1"
GCS_OUTPUT_PATH = "data/reference_data_sample.csv"
OUTPUT_LOCAL = "data/reference_data_sample.csv"


def create_stratified_sample(df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    """
    Create a stratified sample based on 'nom_du_compteur'.
    Ensures all compteurs are represented proportionally.
    """
    if "nom_du_compteur" not in df.columns:
        print("⚠️ Column 'nom_du_compteur' not found, using random sample")
        return df.sample(n=min(n_samples, len(df)), random_state=42)

    # Stratified sampling by compteur
    sample = df.groupby("nom_du_compteur", group_keys=False).apply(
        lambda x: x.sample(
            n=min(len(x), max(1, int(n_samples * len(x) / len(df)))),
            random_state=42,
        )
    )

    # If we don't have enough samples, add random ones
    if len(sample) < n_samples:
        remaining = n_samples - len(sample)
        additional = df[~df.index.isin(sample.index)].sample(
            n=min(remaining, len(df) - len(sample)), random_state=42
        )
        sample = pd.concat([sample, additional])

    return sample.head(n_samples)


def main():
    print("🔍 Creating reference data sample for drift detection...")

    # Check if reference data exists
    if not os.path.exists(REFERENCE_DATA_PATH):
        print(f"❌ Reference data not found at {REFERENCE_DATA_PATH}")
        sys.exit(1)

    # Load reference data (semicolon-separated)
    print(f"📥 Loading reference data from {REFERENCE_DATA_PATH}")
    df_full = pd.read_csv(REFERENCE_DATA_PATH, sep=";")
    print(f"✅ Loaded {len(df_full)} rows, {len(df_full.columns)} columns")
    print(f"   Size: {df_full.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    # Normalize column names (same as BigQuery)
    df_full.columns = [col.lower().replace(" ", "_") for col in df_full.columns]

    # Create stratified sample
    print(f"\n🎲 Creating stratified sample of {SAMPLE_SIZE} rows...")
    df_sample = create_stratified_sample(df_full, SAMPLE_SIZE)
    print(f"✅ Sample created: {len(df_sample)} rows")

    # Show sample distribution
    if "nom_du_compteur" in df_sample.columns:
        print("\n📊 Sample distribution by compteur:")
        compteur_counts = df_sample["nom_du_compteur"].value_counts()
        print(f"   - Unique compteurs: {len(compteur_counts)}")
        print("   - Top 5 compteurs:")
        for compteur, count in compteur_counts.head(5).items():
            print(f"     • {compteur}: {count} rows")

    # Save locally
    print(f"\n💾 Saving sample to {OUTPUT_LOCAL}...")
    df_sample.to_csv(OUTPUT_LOCAL, index=False)
    local_size = os.path.getsize(OUTPUT_LOCAL) / 1024**2
    print(f"✅ Saved locally ({local_size:.1f} MB)")

    # Upload to GCS
    print(f"\n☁️ Uploading to gs://{GCS_BUCKET}/{GCS_OUTPUT_PATH}...")
    try:
        # Get project ID from environment or use default
        project_id = os.getenv("BQ_PROJECT", "datascientest-460618")
        client = storage.Client(project=project_id)
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(GCS_OUTPUT_PATH)
        blob.upload_from_filename(OUTPUT_LOCAL)
        print("✅ Uploaded to GCS successfully")
        print(f"   📍 GCS path: gs://{GCS_BUCKET}/{GCS_OUTPUT_PATH}")
    except Exception as e:
        print(f"⚠️ Failed to upload to GCS: {e}")
        print("   Local file saved, you can upload manually later")

    # Summary
    print("\n" + "=" * 60)
    print("✅ Reference sample created successfully!")
    print(f"   - Original size: {len(df_full)} rows")
    print(
        f"   - Sample size: {len(df_sample)} rows ({len(df_sample)/len(df_full)*100:.1f}%)"
    )
    print(f"   - File size: {local_size:.1f} MB")
    print(f"   - Location: {OUTPUT_LOCAL}")
    print(f"   - GCS: gs://{GCS_BUCKET}/{GCS_OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
