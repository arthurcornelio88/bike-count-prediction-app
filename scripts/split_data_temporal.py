#!/usr/bin/env python3
"""Temporal data split for baseline creation.

This script splits the bike traffic dataset into train/test subsets based on temporal split.

NEW STRATEGY (2025-10-11):
After data quality validation, we use current_api_data.csv (905k records, 2024-09-01 → 2025-10-10)
as the unified baseline with 80/20 train/test split.

Rationale:
- Perfect correlation (r=1.0) between all data sources
- Zero temporal gap with live API (starts 2025-10-11)
- Test set represents current traffic patterns

Usage:
    # Default 80/20 split
    python scripts/split_data_temporal.py

    # Custom split ratio
    python scripts/split_data_temporal.py --train-ratio 0.75

    # Specify input file
    python scripts/split_data_temporal.py --input data/current_api_data.csv
"""

import argparse
import pandas as pd


def split_temporal_data(
    input_file: str = "data/current_api_data.csv",
    train_ratio: float = 0.8,
    output_train: str = "data/train_baseline.csv",
    output_test: str = "data/test_baseline.csv",
) -> tuple[str, str]:
    """Split dataset into train/test based on temporal cutoff.

    Args:
        input_file: Path to input CSV
        train_ratio: Proportion for training (default 0.8 = 80%)
        output_train: Path to output training CSV
        output_test: Path to output test CSV

    Returns:
        Tuple of (train_path, test_path)
    """
    print(f"📂 Loading data from {input_file}...")
    df = pd.read_csv(input_file, sep=";")

    # Parse datetime column
    df["date"] = pd.to_datetime(
        df["Date et heure de comptage"], utc=True
    ).dt.tz_convert("Europe/Paris")

    # Sort by date to ensure temporal order
    df = df.sort_values("date")

    # Calculate cutoff index based on train_ratio
    cutoff_idx = int(len(df) * train_ratio)
    cutoff_date = df.iloc[cutoff_idx]["date"]

    # Temporal split
    df_train = df[df["date"] < cutoff_date].copy()
    df_test = df[df["date"] >= cutoff_date].copy()

    # Display statistics
    train_pct = len(df_train) / len(df) * 100
    test_pct = len(df_test) / len(df) * 100

    print("\n📊 Split Statistics:")
    print("=" * 60)
    print(f"Input file:        {input_file}")
    print(f"Total records:     {len(df):,}")
    print(f"Cutoff date:       {cutoff_date.date()}")
    print()
    print("TRAIN SET:")
    print(
        f"  - Period:        {df_train['date'].min().date()} → {df_train['date'].max().date()}"
    )
    print(f"  - Records:       {len(df_train):,} ({train_pct:.1f}%)")
    print(f"  - Valid (>0):    {(df_train['Comptage horaire'] > 0).sum():,}")
    print(f"  - Counters:      {df_train['Nom du compteur'].nunique()}")
    print()
    print("TEST SET:")
    print(
        f"  - Period:        {df_test['date'].min().date()} → {df_test['date'].max().date()}"
    )
    print(f"  - Records:       {len(df_test):,} ({test_pct:.1f}%)")
    print(f"  - Valid (>0):    {(df_test['Comptage horaire'] > 0).sum():,}")
    print(f"  - Counters:      {df_test['Nom du compteur'].nunique()}")
    print("=" * 60)

    # Save without temporary 'date' column, force semicolon separator
    df_train.drop(columns=["date"]).to_csv(output_train, sep=";", index=False)
    df_test.drop(columns=["date"]).to_csv(output_test, sep=";", index=False)

    print("\n✅ Files created:")
    print(f"  - {output_train}")
    print(f"  - {output_test}")

    return output_train, output_test


def main():
    """Main entry point with CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Split bike traffic data into train/test sets"
    )
    parser.add_argument(
        "--input",
        default="data/current_api_data.csv",
        help="Input CSV file (default: data/current_api_data.csv)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio (default: 0.8 for 80%%)",
    )
    parser.add_argument(
        "--output-train",
        default="data/train_baseline.csv",
        help="Output training CSV (default: data/train_baseline.csv)",
    )
    parser.add_argument(
        "--output-test",
        default="data/test_baseline.csv",
        help="Output test CSV (default: data/test_baseline.csv)",
    )

    args = parser.parse_args()

    # Run split
    train_path, test_path = split_temporal_data(
        input_file=args.input,
        train_ratio=args.train_ratio,
        output_train=args.output_train,
        output_test=args.output_test,
    )

    # Next steps
    print("\n" + "=" * 60)
    print("📋 NEXT STEPS (DVC + GCS + Training)")
    print("=" * 60)

    print("\n1️⃣ Track datasets with DVC:")
    print(f"   dvc add {train_path}")
    print(f"   dvc add {test_path}")

    print("\n2️⃣ Configure GCS credentials (if not done):")
    print("   export GOOGLE_APPLICATION_CREDENTIALS=./mlflow-trainer.json")
    print("   dvc remote modify gcs_storage credentialpath ./mlflow-trainer.json")

    print("\n3️⃣ Push data to GCS:")
    print("   dvc push")

    print("\n4️⃣ Commit DVC metadata to Git:")
    print("   git add data/*.dvc .dvc/config .gitignore")
    print('   git commit -m "chore: add train/test baseline from current_api_data"')

    print("\n5️⃣ Train champion model:")
    print("   python scripts/train_legacy_model.py \\")
    print(f"       --train {train_path} \\")
    print(f"       --test {test_path} \\")
    print("       --output models/champion_v1")

    print("\n6️⃣ Upload baseline to BigQuery:")
    print("   # See docs/bigquery_setup.md for SQL commands")

    print("\n" + "=" * 60)
    print("📚 Documentation:")
    print("   - docs/fetch_data_strategy.md (full strategy)")
    print("   - docs/dvc.md (DVC workflow)")
    print("   - MLOPS_ROADMAP.md (Phase 3 implementation)")
    print("=" * 60)


if __name__ == "__main__":
    main()
