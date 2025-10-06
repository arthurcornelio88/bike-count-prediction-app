#!/usr/bin/env python3
"""Temporal data split: reference (train/test) vs current (production).

This script splits the bike traffic dataset into two temporal subsets:
- reference_data.csv: First 70% of data (for training/testing)
- current_data.csv: Last 30% of data (for drift detection)

Dataset range: 2024-04-01 to 2025-05-17
Cutoff date: 2025-01-14 (70/30 split)

Usage:
    python scripts/split_data_temporal.py
"""

import pandas as pd
from datetime import datetime
import pytz

# Configuration
INPUT_FILE = "data/comptage-velo-donnees-compteurs.csv"
OUTPUT_REF = "data/reference_data.csv"
OUTPUT_CURRENT = "data/current_data.csv"

# Cutoff date with timezone (70% split point)
PARIS_TZ = pytz.timezone('Europe/Paris')
CUTOFF_DATE = PARIS_TZ.localize(datetime(2025, 1, 14))


def split_temporal_data():
    """Split dataset into reference and current based on temporal cutoff."""

    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE, sep=";")

    # Parse datetime column
    df['date'] = pd.to_datetime(
        df['Date et heure de comptage'],
        utc=True
    ).dt.tz_convert('Europe/Paris')

    # Temporal split
    df_reference = df[df['date'] < CUTOFF_DATE].copy()
    df_current = df[df['date'] >= CUTOFF_DATE].copy()

    # Display statistics
    ref_pct = len(df_reference) / len(df) * 100
    curr_pct = len(df_current) / len(df) * 100

    print(f"\nSplit statistics:")
    print(f"  - Reference (< {CUTOFF_DATE.date()}): {len(df_reference):,} rows ({ref_pct:.1f}%)")
    print(f"  - Current (>= {CUTOFF_DATE.date()}):  {len(df_current):,} rows ({curr_pct:.1f}%)")
    print(f"  - Total:                     {len(df):,} rows")

    # Save without temporary 'date' column, force semicolon separator
    df_reference.drop(columns=['date']).to_csv(OUTPUT_REF, sep=";", index=False)
    df_current.drop(columns=['date']).to_csv(OUTPUT_CURRENT, sep=";", index=False)

    print(f"\nFiles created:")
    print(f"  - {OUTPUT_REF}")
    print(f"  - {OUTPUT_CURRENT}")

    return OUTPUT_REF, OUTPUT_CURRENT


if __name__ == "__main__":
    split_temporal_data()

    print("\n" + "="*60)
    print("Next steps (DVC workflow):")
    print("="*60)

    print("\n1. Track datasets with DVC:")
    print("   dvc add data/reference_data.csv")
    print("   dvc add data/current_data.csv")

    print("\n2. Configure GCS credentials (if not done):")
    print("   export GOOGLE_APPLICATION_CREDENTIALS=./mlflow-trainer.json")
    print("   dvc remote modify gcs_storage credentialpath ./mlflow-trainer.json")

    print("\n3. Push data to GCS:")
    print("   dvc push")

    print("\n4. Commit DVC metadata to Git:")
    print("   git add data/*.dvc .dvc/config .gitignore")
    print("   git commit -m 'feat: add DVC temporal data split'")

    print("\n" + "="*60)
    print("📚 Full documentation: docs/dvc.md")
    print("="*60)
