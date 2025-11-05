#!/usr/bin/env python3
"""
Analyze compteur distribution in train_baseline vs test_baseline.
Quick diagnosis to understand why R² = 0.08 on baseline data.
"""

import pandas as pd
import sys
from pathlib import Path

# Paths
TRAIN_PATH = "data/train_baseline.csv"
TEST_PATH = "data/test_baseline.csv"
CURRENT_PATH = "data/current_data.csv"


def analyze_compteurs(train_path: str, test_path: str, current_path: str = None):
    """Compare compteur distribution across datasets"""

    print("=" * 80)
    print("🔍 BASELINE DATA ANALYSIS - COMPTEUR DISTRIBUTION")
    print("=" * 80)

    # Load data
    print("\n📥 Loading data...")
    try:
        df_train = pd.read_csv(train_path, sep=";")
        print(f"✅ Train baseline: {len(df_train):,} rows")
    except Exception as e:
        print(f"❌ Failed to load train_baseline: {e}")
        return

    try:
        df_test = pd.read_csv(test_path, sep=";")
        print(f"✅ Test baseline: {len(df_test):,} rows")
    except Exception as e:
        print(f"❌ Failed to load test_baseline: {e}")
        return

    df_current = None
    if current_path and Path(current_path).exists():
        try:
            df_current = pd.read_csv(current_path, sep=";")
            print(f"✅ Current data: {len(df_current):,} rows")
        except Exception as e:
            print(f"⚠️  Failed to load current_data: {e}")

    # Identify compteur column
    compteur_col = None
    for col in ["Nom du compteur", "nom_du_compteur", "Nom_du_compteur"]:
        if col in df_train.columns:
            compteur_col = col
            break

    if not compteur_col:
        print("❌ Compteur column not found!")
        print(f"Available columns: {list(df_train.columns)}")
        return

    print(f"📊 Using compteur column: '{compteur_col}'")

    # Get unique compteurs
    train_compteurs = set(df_train[compteur_col].dropna().unique())
    test_compteurs = set(df_test[compteur_col].dropna().unique())

    print(f"\n{'=' * 80}")
    print("📊 COMPTEUR STATISTICS")
    print(f"{'=' * 80}")
    print(f"Train unique compteurs: {len(train_compteurs)}")
    print(f"Test unique compteurs:  {len(test_compteurs)}")

    # Overlap analysis
    common = train_compteurs & test_compteurs
    only_train = train_compteurs - test_compteurs
    only_test = test_compteurs - train_compteurs

    print("\n🔍 Overlap Analysis:")
    print(
        f"   - Common compteurs:     {len(common)} ({len(common)/len(train_compteurs)*100:.1f}% of train)"
    )
    print(f"   - Only in train:        {len(only_train)}")
    print(f"   - Only in test:         {len(only_test)} 🚨 UNKNOWN TO MODEL!")

    if only_test:
        print("\n🚨 UNKNOWN COMPTEURS IN TEST SET:")
        print(f"   Model will fail on these {len(only_test)} compteurs!")

        # Show examples
        only_test_list = sorted(only_test)
        print("\n   Examples (showing first 10):")
        for i, compteur in enumerate(only_test_list[:10], 1):
            count = len(df_test[df_test[compteur_col] == compteur])
            pct = count / len(df_test) * 100
            print(f"      {i}. '{compteur}' ({count:,} rows, {pct:.1f}% of test)")

        if len(only_test_list) > 10:
            print(f"      ... and {len(only_test_list) - 10} more")

        # Calculate impact
        unknown_rows = df_test[df_test[compteur_col].isin(only_test)]
        unknown_pct = len(unknown_rows) / len(df_test) * 100
        print(
            f"\n   📊 Impact: {len(unknown_rows):,} rows ({unknown_pct:.1f}% of test set)"
        )
        print("   💡 This explains R² = 0.08! Model has no idea about these compteurs.")

    # Check if current data adds new compteurs
    if df_current is not None and compteur_col in df_current.columns:
        current_compteurs = set(df_current[compteur_col].dropna().unique())
        new_in_current = current_compteurs - train_compteurs

        print("\n📈 Current Data Analysis:")
        print(f"   - Current unique compteurs: {len(current_compteurs)}")
        print(f"   - New compteurs (not in train): {len(new_in_current)}")

        if new_in_current:
            print("\n   🆕 NEW COMPTEURS IN CURRENT DATA:")
            new_list = sorted(new_in_current)
            for i, compteur in enumerate(new_list[:10], 1):
                count = len(df_current[df_current[compteur_col] == compteur])
                pct = count / len(df_current) * 100
                print(
                    f"      {i}. '{compteur}' ({count:,} rows, {pct:.1f}% of current)"
                )

            if len(new_list) > 10:
                print(f"      ... and {len(new_list) - 10} more")

            # Calculate impact
            new_rows = df_current[df_current[compteur_col].isin(new_in_current)]
            new_pct = len(new_rows) / len(df_current) * 100
            print(
                f"\n   📊 Impact: {len(new_rows):,} rows ({new_pct:.1f}% of current data)"
            )

    # Distribution analysis
    print(f"\n{'=' * 80}")
    print("📊 COMPTEUR FREQUENCY DISTRIBUTION")
    print(f"{'=' * 80}")

    print("\nTop 10 compteurs in TRAIN:")
    train_counts = df_train[compteur_col].value_counts().head(10)
    for i, (compteur, count) in enumerate(train_counts.items(), 1):
        pct = count / len(df_train) * 100
        in_test = "✅" if compteur in test_compteurs else "❌"
        print(f"   {i}. {in_test} '{compteur}': {count:,} ({pct:.1f}%)")

    print("\nTop 10 compteurs in TEST:")
    test_counts = df_test[compteur_col].value_counts().head(10)
    for i, (compteur, count) in enumerate(test_counts.items(), 1):
        pct = count / len(df_test) * 100
        in_train = "✅" if compteur in train_compteurs else "🚨 UNKNOWN"
        print(f"   {i}. {in_train} '{compteur}': {count:,} ({pct:.1f}%)")

    # Date range analysis
    print(f"\n{'=' * 80}")
    print("📅 DATE RANGE ANALYSIS")
    print(f"{'=' * 80}")

    date_col = None
    for col in ["Date et heure de comptage", "date_et_heure_de_comptage"]:
        if col in df_train.columns:
            date_col = col
            break

    if date_col:
        df_train[date_col] = pd.to_datetime(df_train[date_col], utc=True)
        df_test[date_col] = pd.to_datetime(df_test[date_col], utc=True)

        print(f"\nTRAIN dates: {df_train[date_col].min()} → {df_train[date_col].max()}")
        print(f"TEST dates:  {df_test[date_col].min()} → {df_test[date_col].max()}")

        if df_current is not None and date_col in df_current.columns:
            df_current[date_col] = pd.to_datetime(df_current[date_col], utc=True)
            print(
                f"CURRENT dates: {df_current[date_col].min()} → {df_current[date_col].max()}"
            )

    # Recommendations
    print(f"\n{'=' * 80}")
    print("💡 RECOMMENDATIONS")
    print(f"{'=' * 80}")

    if only_test:
        print("\n🚨 CRITICAL ISSUE FOUND:")
        print(
            f"   Test set contains {len(only_test)} unknown compteurs ({len(unknown_rows):,} rows)"
        )
        print("   This causes catastrophic R² = 0.08 on baseline!")
        print("\n   📋 Solutions:")
        print("      1. Re-split data ensuring ALL compteurs in train")
        print("      2. Implement geographic fallback (nearest compteur)")
        print("      3. Train model without compteur feature (temporal only)")
        print("      4. Collect more data for unknown compteurs before splitting")

    if new_in_current and df_current is not None:
        print("\n🆕 CURRENT DATA CONTAINS NEW COMPTEURS:")
        print(f"   {len(new_in_current)} new compteurs found ({len(new_rows):,} rows)")
        print("\n   📋 Fine-tuning Strategy:")
        print("      1. Add current_data to training (sliding window)")
        print("      2. Re-evaluate on test_baseline after retraining")
        print(
            "      3. Expected: R² should IMPROVE on baseline after including new compteurs"
        )

    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    # Check if files exist
    if not Path(TRAIN_PATH).exists():
        print(f"❌ Train file not found: {TRAIN_PATH}")
        sys.exit(1)

    if not Path(TEST_PATH).exists():
        print(f"❌ Test file not found: {TEST_PATH}")
        sys.exit(1)

    analyze_compteurs(TRAIN_PATH, TEST_PATH, CURRENT_PATH)
