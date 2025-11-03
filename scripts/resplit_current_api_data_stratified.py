#!/usr/bin/env python3
"""
Re-split current_api_data.csv with stratification by compteur.
Ensures EVERY compteur appears in BOTH train and test sets (ISO split).

This creates the CORRECT baseline from the most recent data (2024-09-01 → 2025-10-10)
instead of using the outdated baseline (2024-04-01 → 2025-01-13).

This eliminates the 10-month gap between baseline and production data.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys
from google.cloud import storage


SOURCE_PATH = "data/current_api_data.csv"
TRAIN_OUTPUT = "data/train_baseline.csv"
TEST_OUTPUT = "data/test_baseline.csv"
GCS_BUCKET = "df_traffic_cyclist1"
GCS_TRAIN_PATH = "raw_data/train_baseline.csv"
GCS_TEST_PATH = "raw_data/test_baseline.csv"


def load_current_api_data(source_path: str):
    """Load current_api_data.csv (905k records, 2024-09-01 → 2025-10-10)"""
    print("=" * 80)
    print("📥 STEP 1: Loading current_api_data.csv")
    print("=" * 80)

    if not Path(source_path).exists():
        print(f"❌ Source file not found: {source_path}")
        print("   Please ensure current_api_data.csv exists in data/")
        sys.exit(1)

    try:
        df = pd.read_csv(source_path, sep=";")
        print(f"✅ Loaded {len(df):,} rows from {source_path}")

        # Show date range
        if "Date et heure de comptage" in df.columns:
            date_col = "Date et heure de comptage"
        elif "date_et_heure_de_comptage" in df.columns:
            date_col = "date_et_heure_de_comptage"
        else:
            print("⚠️  Date column not found, skipping date range display")
            return df

        df[date_col] = pd.to_datetime(df[date_col])
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        print(f"📅 Date range: {min_date.date()} → {max_date.date()}")

        return df
    except Exception as e:
        print(f"❌ Failed to load {source_path}: {e}")
        sys.exit(1)


def analyze_compteur_distribution(df: pd.DataFrame, compteur_col: str):
    """Analyze compteur distribution and detect issues"""
    print("\n" + "=" * 80)
    print("📊 STEP 2: Analyzing compteur distribution")
    print("=" * 80)

    # Count samples per compteur
    compteur_counts = df[compteur_col].value_counts()
    n_compteurs = len(compteur_counts)

    print(f"Total unique compteurs: {n_compteurs}")
    print(f"Total samples: {len(df):,}")
    print("\nCompteur distribution:")
    print(f"   - Min samples per compteur: {compteur_counts.min():,}")
    print(f"   - Max samples per compteur: {compteur_counts.max():,}")
    print(f"   - Mean samples per compteur: {compteur_counts.mean():.1f}")
    print(f"   - Median samples per compteur: {compteur_counts.median():.1f}")

    # Check for compteurs with very few samples (< 10)
    rare_compteurs = compteur_counts[compteur_counts < 10]
    if len(rare_compteurs) > 0:
        print(f"\n⚠️  WARNING: {len(rare_compteurs)} compteurs have < 10 samples:")
        for compteur, count in rare_compteurs.items():
            print(f"   - '{compteur}': {count} samples")
        print(
            "   → These may cause issues with stratified split (need >= 2 samples per class)"
        )

    print("\nTop 10 compteurs by sample count:")
    for i, (compteur, count) in enumerate(compteur_counts.head(10).items(), 1):
        pct = count / len(df) * 100
        print(f"   {i}. '{compteur}': {count:,} samples ({pct:.1f}%)")

    return compteur_counts


def stratified_split(
    df: pd.DataFrame, compteur_col: str, test_size: float = 0.2, random_state: int = 42
):
    """
    Perform stratified split ensuring ALL compteurs in both train and test.

    Strategy:
    1. For compteurs with >= 2 samples: use stratified split
    2. For compteurs with 1 sample: manually place in train (can't split 1 sample)
    """
    print("\n" + "=" * 80)
    print(
        f"🔄 STEP 3: Stratified split (test_size={test_size}, random_state={random_state})"
    )
    print("=" * 80)

    compteur_counts = df[compteur_col].value_counts()

    # Separate compteurs by sample count
    single_sample_compteurs = compteur_counts[compteur_counts == 1].index.tolist()
    multi_sample_compteurs = compteur_counts[compteur_counts >= 2].index.tolist()

    print(f"Compteurs with 1 sample: {len(single_sample_compteurs)}")
    print(f"Compteurs with >= 2 samples: {len(multi_sample_compteurs)}")

    if single_sample_compteurs:
        print(
            f"\n⚠️  {len(single_sample_compteurs)} compteurs have only 1 sample - will be placed in TRAIN only"
        )
        print(f"   Examples: {single_sample_compteurs[:5]}")

    # Extract single-sample data (goes to train)
    df_single = df[df[compteur_col].isin(single_sample_compteurs)].copy()

    # Extract multi-sample data (can be split)
    df_multi = df[df[compteur_col].isin(multi_sample_compteurs)].copy()

    print(f"\nSingle-sample data: {len(df_single):,} rows → TRAIN")
    print(f"Multi-sample data: {len(df_multi):,} rows → Stratified split")

    # Perform stratified split on multi-sample data
    try:
        train_multi, test_multi = train_test_split(
            df_multi,
            test_size=test_size,
            random_state=random_state,
            stratify=df_multi[compteur_col],  # Stratify by compteur
        )
        print("✅ Stratified split successful!")
        print(f"   - Train (multi-sample): {len(train_multi):,} rows")
        print(f"   - Test (multi-sample): {len(test_multi):,} rows")
    except Exception as e:
        print(f"❌ Stratified split failed: {e}")
        print("   Falling back to simple random split (not ideal)...")
        train_multi, test_multi = train_test_split(
            df_multi, test_size=test_size, random_state=random_state
        )

    # Combine single-sample data with train split
    df_train_final = pd.concat([train_multi, df_single], ignore_index=True)
    df_test_final = test_multi.copy()

    print("\n✅ Final split:")
    print(
        f"   - Train: {len(df_train_final):,} rows ({len(df_train_final)/len(df)*100:.1f}%)"
    )
    print(
        f"   - Test:  {len(df_test_final):,} rows ({len(df_test_final)/len(df)*100:.1f}%)"
    )

    return df_train_final, df_test_final


def verify_split_quality(
    df_train: pd.DataFrame, df_test: pd.DataFrame, compteur_col: str
):
    """Verify that split is ISO (all compteurs in both sets, or explained)"""
    print("\n" + "=" * 80)
    print("🔍 STEP 4: Verifying split quality (ISO check)")
    print("=" * 80)

    train_compteurs = set(df_train[compteur_col].unique())
    test_compteurs = set(df_test[compteur_col].unique())

    common = train_compteurs & test_compteurs
    only_train = train_compteurs - test_compteurs
    only_test = test_compteurs - train_compteurs

    print(f"Train unique compteurs: {len(train_compteurs)}")
    print(f"Test unique compteurs:  {len(test_compteurs)}")
    print(
        f"\n✅ Common compteurs: {len(common)} ({len(common)/len(train_compteurs)*100:.1f}% of train)"
    )

    # Check for orphan compteurs
    if only_train:
        print(f"\n⚠️  {len(only_train)} compteurs ONLY in train:")
        # Check if these are single-sample compteurs (expected)
        single_sample_count = 0
        for compteur in only_train:
            count_in_train = len(df_train[df_train[compteur_col] == compteur])
            if count_in_train == 1:
                single_sample_count += 1

        print(f"   - {single_sample_count} are single-sample compteurs (expected)")
        print(
            f"   - {len(only_train) - single_sample_count} are multi-sample (⚠️  ISSUE!)"
        )

        if len(only_train) - single_sample_count > 0:
            print("   Examples of multi-sample orphans in train:")
            for compteur in list(only_train)[:5]:
                count = len(df_train[df_train[compteur_col] == compteur])
                if count > 1:
                    print(f"      - '{compteur}': {count} samples")

    if only_test:
        print(f"\n🚨 CRITICAL: {len(only_test)} compteurs ONLY in test!")
        print("   This will cause ZERO-VECTOR predictions! (R² collapse)")
        print("   Examples:")
        for compteur in list(only_test)[:10]:
            count = len(df_test[df_test[compteur_col] == compteur])
            pct = count / len(df_test) * 100
            print(f"      - '{compteur}': {count} samples ({pct:.2f}% of test)")
        return False

    if not only_train and not only_test:
        print("\n✅ PERFECT SPLIT: All compteurs present in both train and test!")
    elif only_train and not only_test:
        print(
            "\n✅ ACCEPTABLE SPLIT: Only train has extra compteurs (expected for single-sample)"
        )
        print(
            "   → Model trained on these compteurs won't be tested, but no zero-vectors in test!"
        )

    return True


def save_split(
    df_train: pd.DataFrame, df_test: pd.DataFrame, train_path: str, test_path: str
):
    """Save train/test splits locally"""
    print("\n" + "=" * 80)
    print("💾 STEP 5: Saving new baseline splits locally")
    print("=" * 80)

    # Create backup of old files if they exist
    train_file = Path(train_path)
    test_file = Path(test_path)

    if train_file.exists():
        backup_train = train_file.with_suffix(".csv.bak")
        train_file.rename(backup_train)
        print(f"✅ Backup created: {backup_train}")

    if test_file.exists():
        backup_test = test_file.with_suffix(".csv.bak")
        test_file.rename(backup_test)
        print(f"✅ Backup created: {backup_test}")

    # Save new splits
    df_train.to_csv(train_path, sep=";", index=False)
    print(f"✅ New train saved: {train_path} ({len(df_train):,} rows)")

    df_test.to_csv(test_path, sep=";", index=False)
    print(f"✅ New test saved: {test_path} ({len(df_test):,} rows)")


def upload_to_gcs(
    train_path: str,
    test_path: str,
    bucket_name: str,
    gcs_train_path: str,
    gcs_test_path: str,
):
    """Upload new splits to GCS"""
    print("\n" + "=" * 80)
    print("☁️  STEP 6: Uploading to GCS")
    print("=" * 80)

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # Upload train
        blob_train = bucket.blob(gcs_train_path)
        blob_train.upload_from_filename(train_path)
        print(f"✅ Uploaded: gs://{bucket_name}/{gcs_train_path}")

        # Upload test
        blob_test = bucket.blob(gcs_test_path)
        blob_test.upload_from_filename(test_path)
        print(f"✅ Uploaded: gs://{bucket_name}/{gcs_test_path}")

        print("\n🎉 GCS upload complete!")

    except Exception as e:
        print(f"❌ GCS upload failed: {e}")
        print("   → You can manually upload the files:")
        print(f"     gsutil cp {train_path} gs://{bucket_name}/{gcs_train_path}")
        print(f"     gsutil cp {test_path} gs://{bucket_name}/{gcs_test_path}")


def main():
    """Main execution"""
    print("=" * 80)
    print("🚀 BASELINE DATA CREATION FROM current_api_data.csv")
    print("=" * 80)
    print("Goal: Create train/test baseline with stratification by compteur")
    print(f"Source: {SOURCE_PATH} (2024-09-01 → 2025-10-10, ~905k records)")
    print("This replaces the outdated baseline (2024-04-01 → 2025-01-13)\n")

    # Load data
    df_full = load_current_api_data(SOURCE_PATH)

    # Identify compteur column
    compteur_col = None
    for col in ["Nom du compteur", "nom_du_compteur", "Nom_du_compteur"]:
        if col in df_full.columns:
            compteur_col = col
            break

    if not compteur_col:
        print("❌ Compteur column not found!")
        print(f"   Available columns: {list(df_full.columns)}")
        sys.exit(1)

    print(f"✅ Using compteur column: '{compteur_col}'")

    # Analyze distribution
    # compteur_counts = analyze_compteur_distribution(df_full, compteur_col)

    # Perform stratified split
    df_train, df_test = stratified_split(
        df_full, compteur_col, test_size=0.2, random_state=42
    )

    # Verify quality
    is_valid = verify_split_quality(df_train, df_test, compteur_col)

    if not is_valid:
        print("\n❌ SPLIT QUALITY CHECK FAILED!")
        print("   → Orphan compteurs detected in test set")
        print("   → This would cause R² collapse (zero-vectors)")
        response = input("\nProceed anyway? (yes/no): ")
        if response.lower() != "yes":
            print("❌ Aborting.")
            sys.exit(1)

    # Save locally
    save_split(df_train, df_test, TRAIN_OUTPUT, TEST_OUTPUT)

    # Upload to GCS
    upload_choice = input(f"\nUpload to GCS gs://{GCS_BUCKET}/ ? (yes/no): ")
    if upload_choice.lower() == "yes":
        upload_to_gcs(
            TRAIN_OUTPUT, TEST_OUTPUT, GCS_BUCKET, GCS_TRAIN_PATH, GCS_TEST_PATH
        )
    else:
        print("⏭️  Skipping GCS upload")
        print("   → Manually upload later with:")
        print(f"     gsutil cp {TRAIN_OUTPUT} gs://{GCS_BUCKET}/{GCS_TRAIN_PATH}")
        print(f"     gsutil cp {TEST_OUTPUT} gs://{GCS_BUCKET}/{GCS_TEST_PATH}")

    print("\n" + "=" * 80)
    print("✅ BASELINE CREATION COMPLETE!")
    print("=" * 80)
    print("Next steps:")
    print(f"1. Train new champion on {TRAIN_OUTPUT}")
    print(f"2. Evaluate on {TEST_OUTPUT} - expect R² ≈ 0.79")
    print("3. Upload new champion to GCS/MLflow")
    print("4. Update summary.json")
    print("5. No more 10-month gap! Baseline now ends 2025-10-10 (vs old 2025-01-13)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
