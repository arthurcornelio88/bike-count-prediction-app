"""
Create a small test sample from reference_data.csv for fast testing.
Uses 1000 random rows and saves to data/test_sample.csv
"""

import pandas as pd


def create_test_sample(
    input_file: str = "data/reference_data.csv",
    output_file: str = "data/test_sample.csv",
    n_samples: int = 1000,
    random_state: int = 42,
):
    """
    Create a small test sample from reference data.

    Args:
        input_file: Path to reference_data.csv
        output_file: Path to save test_sample.csv
        n_samples: Number of samples to extract
        random_state: Random seed for reproducibility
    """
    print(f"📊 Reading {input_file}...")

    # Read with low_memory=False to avoid dtype warnings

    df = pd.read_csv(input_file, sep=";", low_memory=False)

    print(f"✅ Loaded {len(df):,} rows")
    print(f"⚡ Sampling {n_samples:,} rows...")

    # Sample random rows
    df_sample = df.sample(n=min(n_samples, len(df)), random_state=random_state)

    print(f"💾 Saving to {output_file}...")
    df_sample.to_csv(output_file, sep=";", index=False)

    # Get file size
    import os

    size_mb = os.path.getsize(output_file) / (1024 * 1024)

    print("✅ Test sample created:")
    print(f"   - Rows: {len(df_sample):,}")
    print(f"   - Size: {size_mb:.2f} MB")
    print(f"   - Location: {output_file}")
    print("\n📦 Next steps:")
    print(f"   1. dvc add {output_file}")
    print(f"   2. git add {output_file}.dvc .gitignore")
    print("   3. git commit -m 'Add test sample for fast training'")
    print("   4. dvc push")


if __name__ == "__main__":
    create_test_sample()
