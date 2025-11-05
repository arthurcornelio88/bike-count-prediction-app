#!/usr/bin/env python3
"""
Analyze feature importance to decide if compteur name can be dropped.
Compare models with/without compteur feature.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend" / "regmodel"))

from app.classes import RawCleanerTransformer, TimeFeatureTransformer

TRAIN_PATH = "data/train_baseline.csv"


def load_and_prepare_data(
    path: str, keep_compteur: bool = True, sample_size: int = 50000
):
    """Load and prepare data with optional compteur feature"""

    print(f"\n{'='*80}")
    print(
        f"📊 Loading data (sample_size={sample_size:,}, keep_compteur={keep_compteur})"
    )
    print(f"{'='*80}")

    # Load sample (to speed up analysis)
    df = pd.read_csv(path, sep=";", nrows=sample_size)
    print(f"✅ Loaded {len(df):,} rows")

    # Separate target
    y = df["Comptage horaire"].copy()
    df = df.drop(columns=["Comptage horaire"])

    # Clean data
    cleaner = RawCleanerTransformer(keep_compteur=keep_compteur)
    df_clean = cleaner.fit_transform(df)

    # Time features
    time_transformer = TimeFeatureTransformer()
    df_features = time_transformer.fit_transform(df_clean)

    # Handle compteur if kept
    if keep_compteur and "nom_du_compteur" in df_features.columns:
        # One-hot encode compteur
        df_features = pd.get_dummies(
            df_features, columns=["nom_du_compteur"], drop_first=True
        )

    print(f"✅ Features prepared: {df_features.shape[1]} columns")
    print(
        f"   Columns: {list(df_features.columns)[:10]}{'...' if len(df_features.columns) > 10 else ''}"
    )

    return df_features, y


def train_and_evaluate(X_train, X_test, y_train, y_test, model_name: str):
    """Train RF and return metrics"""

    print(f"\n🎯 Training {model_name}...")

    rf = RandomForestRegressor(
        n_estimators=50,  # Reduced for speed
        max_depth=20,
        random_state=42,
        n_jobs=-1,
    )

    rf.fit(X_train, y_train)

    # Train metrics
    y_pred_train = rf.predict(X_train)
    r2_train = r2_score(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))

    # Test metrics
    y_pred_test = rf.predict(X_test)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print(f"   Train - R²: {r2_train:.4f}, RMSE: {rmse_train:.2f}")
    print(f"   Test  - R²: {r2_test:.4f}, RMSE: {rmse_test:.2f}")

    # Feature importance (top 10)
    if hasattr(rf, "feature_importances_"):
        importances = pd.DataFrame(
            {"feature": X_train.columns, "importance": rf.feature_importances_}
        ).sort_values("importance", ascending=False)

        print("\n   📊 Top 10 Features:")
        for i, row in importances.head(10).iterrows():
            print(f"      {row['feature'][:50]:50s} {row['importance']:.4f}")

    return {
        "model": rf,
        "r2_train": r2_train,
        "r2_test": r2_test,
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
        "importances": importances if "importances" in locals() else None,
    }


def analyze_compteur_vs_latlon():
    """Compare model performance with/without compteur feature"""

    print("=" * 80)
    print("🔬 FEATURE IMPORTANCE ANALYSIS: COMPTEUR vs LAT/LON")
    print("=" * 80)

    # Check if file exists
    if not Path(TRAIN_PATH).exists():
        print(f"❌ Train file not found: {TRAIN_PATH}")
        return

    sample_size = 50000  # Use 50K samples for speed

    # Scenario 1: WITH compteur
    print("\n" + "=" * 80)
    print("📊 SCENARIO 1: Model WITH Compteur Feature")
    print("=" * 80)

    X_with, y_with = load_and_prepare_data(
        TRAIN_PATH, keep_compteur=True, sample_size=sample_size
    )
    X_train_with, X_test_with, y_train_with, y_test_with = train_test_split(
        X_with, y_with, test_size=0.2, random_state=42
    )

    results_with = train_and_evaluate(
        X_train_with,
        X_test_with,
        y_train_with,
        y_test_with,
        "RandomForest WITH Compteur",
    )

    # Scenario 2: WITHOUT compteur (only lat/lon + temporal)
    print("\n" + "=" * 80)
    print("📊 SCENARIO 2: Model WITHOUT Compteur (Lat/Lon + Temporal only)")
    print("=" * 80)

    X_without, y_without = load_and_prepare_data(
        TRAIN_PATH, keep_compteur=False, sample_size=sample_size
    )
    X_train_without, X_test_without, y_train_without, y_test_without = train_test_split(
        X_without, y_without, test_size=0.2, random_state=42
    )

    results_without = train_and_evaluate(
        X_train_without,
        X_test_without,
        y_train_without,
        y_test_without,
        "RandomForest WITHOUT Compteur",
    )

    # Comparison
    print("\n" + "=" * 80)
    print("📊 COMPARISON & RECOMMENDATIONS")
    print("=" * 80)

    r2_diff = results_with["r2_test"] - results_without["r2_test"]
    rmse_diff = results_without["rmse_test"] - results_with["rmse_test"]

    print("\nTest Set Performance:")
    print(
        f"   WITH compteur:    R² = {results_with['r2_test']:.4f}, RMSE = {results_with['rmse_test']:.2f}"
    )
    print(
        f"   WITHOUT compteur: R² = {results_without['r2_test']:.4f}, RMSE = {results_without['rmse_test']:.2f}"
    )
    print(
        f"\n   Δ R²:   {r2_diff:+.4f} ({'WITH' if r2_diff > 0 else 'WITHOUT'} is better)"
    )
    print(
        f"   Δ RMSE: {rmse_diff:+.2f} ({'WITH' if rmse_diff > 0 else 'WITHOUT'} is better)"
    )

    # Decision criteria
    print(f"\n{'='*80}")
    print("💡 DECISION CRITERIA")
    print(f"{'='*80}")

    if abs(r2_diff) < 0.05:  # Less than 5% difference
        print("\n✅ RECOMMENDATION: DROP COMPTEUR FEATURE")
        print(f"   Reason: Minimal impact on performance (ΔR² = {r2_diff:+.4f})")
        print("   Benefits:")
        print("      1. ✅ No unknown compteur problem in test set")
        print("      2. ✅ Model generalizes to ANY new location (lat/lon)")
        print("      3. ✅ Simpler model, fewer features")
        print("      4. ✅ Easier deployment (no compteur mapping needed)")
        print(
            f"\n   Trade-off: Slightly lower R² ({results_without['r2_test']:.4f} vs {results_with['r2_test']:.4f})"
        )

    elif r2_diff > 0.05:  # WITH compteur is significantly better
        print("\n⚠️  RECOMMENDATION: KEEP COMPTEUR (BUT FIX UNKNOWN HANDLING)")
        print(f"   Reason: Significant performance gain (ΔR² = {r2_diff:+.4f})")
        print("   Required fixes:")
        print("      1. Implement geographic fallback (nearest compteur by GPS)")
        print("      2. OR: Create 'unknown' embedding for unseen compteurs")
        print("      3. OR: Re-split data ensuring all compteurs in train")
        print(f"\n   Benefit: Better predictions ({results_with['r2_test']:.4f} R²)")

    else:  # WITHOUT is better (unlikely)
        print("\n🎉 RECOMMENDATION: DROP COMPTEUR FEATURE")
        print(f"   Reason: Model performs BETTER without it (ΔR² = {r2_diff:+.4f})")
        print("   This suggests compteur name adds noise or overfitting.")

    # Analyze lat/lon importance
    if results_without["importances"] is not None:
        lat_lon_features = ["latitude", "longitude"]
        lat_lon_importance = results_without["importances"][
            results_without["importances"]["feature"].isin(lat_lon_features)
        ]["importance"].sum()

        print("\n📍 Geographic Information (Lat/Lon):")
        print(f"   Combined importance: {lat_lon_importance:.4f}")
        print("   Individual:")
        for feat in lat_lon_features:
            imp = results_without["importances"][
                results_without["importances"]["feature"] == feat
            ]["importance"].values
            if len(imp) > 0:
                print(f"      {feat}: {imp[0]:.4f}")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    analyze_compteur_vs_latlon()
