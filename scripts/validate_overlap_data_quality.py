"""
Data Quality Validation: CSV vs API on Overlap Period

This script compares data quality between:
- Historical CSV: 2024-09-01 → 2025-05-17 (overlap period)
- API data: 2024-09-01 → 2025-05-17 (same period)

Validates:
1. Record counts and coverage
2. Counter overlap (same counters in both sources?)
3. Statistical distributions (mean, std, percentiles)
4. Temporal coverage (missing dates?)
5. Outliers and anomalies
6. Correlation between sources (for same counter/date pairs)

Goal: Decide if API data is trustworthy enough to use for training
"""

import pandas as pd
from datetime import datetime
import json
from scipy import stats

# ========================================
# CONFIGURATION
# ========================================

OVERLAP_START = "2024-09-01"
OVERLAP_END = "2025-05-17"

API_URL = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/comptage-velo-donnees-compteurs/records"

print("=" * 80)
print("🔍 DATA QUALITY VALIDATION - CSV vs API (Overlap Period)")
print("=" * 80)
print(f"\nOverlap period: {OVERLAP_START} → {OVERLAP_END}")
print()

# ========================================
# 1. LOAD HISTORICAL CSV DATA
# ========================================

print("=" * 80)
print("📁 1. LOADING HISTORICAL CSV DATA")
print("-" * 80)

# Load both CSV files
reference_df = pd.read_csv("data/reference_data.csv", sep=";")
current_df = pd.read_csv("data/current_data.csv", sep=";")

# Combine
csv_df = pd.concat([reference_df, current_df], ignore_index=True)

# Parse dates
csv_df["Date et heure de comptage"] = pd.to_datetime(
    csv_df["Date et heure de comptage"], utc=True
).dt.tz_convert("Europe/Paris")

# Filter to overlap period
csv_overlap = csv_df[
    (csv_df["Date et heure de comptage"] >= OVERLAP_START)
    & (csv_df["Date et heure de comptage"] <= OVERLAP_END)
].copy()

# Filter valid records (comptage > 0)
csv_overlap_valid = csv_overlap[csv_overlap["Comptage horaire"] > 0].copy()

print("\n✅ CSV loaded:")
print(f"   - Total records in overlap period: {len(csv_overlap):,}")
print(
    f"   - Valid records (comptage > 0): {len(csv_overlap_valid):,} ({len(csv_overlap_valid)/len(csv_overlap)*100:.1f}%)"
)
print(
    f"   - Date range: {csv_overlap['Date et heure de comptage'].min()} → {csv_overlap['Date et heure de comptage'].max()}"
)
print(f"   - Unique counters: {csv_overlap['Nom du compteur'].nunique()}")

# ========================================
# 2. LOAD API DATA FROM CSV
# ========================================

print("\n" + "=" * 80)
print("🌐 2. LOADING API DATA (from current_api_data.csv)")
print("-" * 80)

# Load API CSV (downloaded from Paris Open Data)
# Detect separator automatically or use common ones
try:
    api_df = pd.read_csv("data/current_api_data.csv", sep=";")
    print("✅ Loaded with sep=';'")
except Exception:
    try:
        api_df = pd.read_csv("data/current_api_data.csv", sep=",")
        print("✅ Loaded with sep=','")
    except Exception as e:
        print(f"❌ Error loading current_api_data.csv: {e}")
        print("Please ensure the file exists at data/current_api_data.csv")
        exit(1)

print("\n📊 API CSV loaded:")
print(f"   - Total records: {len(api_df):,}")
print(f"   - Columns: {api_df.columns.tolist()[:5]}... ({len(api_df.columns)} total)")

# Detect date column (could be 'date', 'Date et heure de comptage', etc.)
date_col = None
for col in ["date", "Date et heure de comptage", "date_et_heure_de_comptage"]:
    if col in api_df.columns:
        date_col = col
        break

if date_col is None:
    print(
        f"❌ Could not find date column. Available columns: {api_df.columns.tolist()}"
    )
    exit(1)

# Detect count column (could be 'sum_counts', 'Comptage horaire', etc.)
count_col = None
for col in ["sum_counts", "Comptage horaire", "comptage_horaire"]:
    if col in api_df.columns:
        count_col = col
        break

if count_col is None:
    print(
        f"❌ Could not find count column. Available columns: {api_df.columns.tolist()}"
    )
    exit(1)

# Detect counter name column
counter_col = None
for col in ["nom_compteur", "Nom du compteur", "nom_du_compteur", "id_compteur"]:
    if col in api_df.columns:
        counter_col = col
        break

print(f"   - Date column: {date_col}")
print(f"   - Count column: {count_col}")
print(f"   - Counter column: {counter_col}")

# Parse dates
api_df[date_col] = pd.to_datetime(api_df[date_col], utc=True).dt.tz_convert(
    "Europe/Paris"
)

# Filter to overlap period
api_overlap = api_df[
    (api_df[date_col] >= OVERLAP_START) & (api_df[date_col] <= OVERLAP_END)
].copy()

# Filter valid records
api_overlap_valid = api_overlap[api_overlap[count_col] > 0].copy()

print("\n✅ API data filtered to overlap period:")
print(f"   - Total records in overlap: {len(api_overlap):,}")
print(
    f"   - Valid records (count > 0): {len(api_overlap_valid):,} ({len(api_overlap_valid)/len(api_overlap)*100:.1f}%)"
)
print(f"   - Date range: {api_overlap[date_col].min()} → {api_overlap[date_col].max()}")
if counter_col:
    print(f"   - Unique counters: {api_overlap[counter_col].nunique()}")
print(f"   - Mean count: {api_overlap_valid[count_col].mean():.1f}")

# ========================================
# 3. DATA QUALITY COMPARISON
# ========================================

print("\n" + "=" * 80)
print("📊 3. DATA QUALITY COMPARISON")
print("-" * 80)

# 3.1 Statistical Distributions
print("\n3.1 Statistical Distributions (Count Values)")
print("-" * 40)

csv_stats = csv_overlap_valid["Comptage horaire"].describe()
api_stats = api_overlap_valid[count_col].describe()

comparison = pd.DataFrame(
    {
        "CSV (Full Overlap)": csv_stats,
        "API (Full Overlap)": api_stats,
        "Difference (%)": ((api_stats - csv_stats) / csv_stats * 100).round(1),
    }
)

print(comparison)

# 3.2 Temporal Coverage
print("\n3.2 Temporal Coverage")
print("-" * 40)

csv_days = csv_overlap_valid["Date et heure de comptage"].dt.date.nunique()
api_days = api_overlap[date_col].dt.date.nunique()

overlap_start_date = pd.to_datetime(OVERLAP_START).date()
overlap_end_date = pd.to_datetime(OVERLAP_END).date()
expected_days = (overlap_end_date - overlap_start_date).days + 1

print(
    f"CSV unique days: {csv_days}/{expected_days} ({csv_days/expected_days*100:.1f}%)"
)
print(
    f"API unique days: {api_days}/{expected_days} ({api_days/expected_days*100:.1f}%)"
)

# 3.3 Counter Coverage
print("\n3.3 Counter Coverage")
print("-" * 40)

csv_counters = set(csv_overlap_valid["Nom du compteur"].unique())
api_counters = set(api_overlap[counter_col].unique()) if counter_col else set()

# Normalize counter names for comparison (remove spaces, lowercase)
csv_counters_norm = {c.strip().lower() for c in csv_counters}
api_counters_norm = {c.strip().lower() for c in api_counters}

common_counters = csv_counters_norm & api_counters_norm
csv_only = csv_counters_norm - api_counters_norm
api_only = api_counters_norm - csv_counters_norm

print(f"CSV counters: {len(csv_counters)}")
print(f"API counters: {len(api_counters)} [sampled]")
print(f"Common counters: {len(common_counters)}")
print(f"CSV-only counters: {len(csv_only)}")
print(f"API-only counters: {len(api_only)}")

if len(common_counters) > 0:
    overlap_pct = len(common_counters) / len(csv_counters) * 100
    print(
        f"\n✅ Counter overlap: {overlap_pct:.1f}% of CSV counters found in API sample"
    )
else:
    print("\n⚠️ No common counters found (possibly naming differences)")

# 3.4 Distribution Comparison (Histogram)
print("\n3.4 Distribution Comparison")
print("-" * 40)

# KS test (Kolmogorov-Smirnov)
ks_statistic, ks_pvalue = stats.ks_2samp(
    csv_overlap_valid["Comptage horaire"], api_overlap_valid[count_col]
)

print("Kolmogorov-Smirnov test:")
print(f"   - Statistic: {ks_statistic:.4f}")
print(f"   - P-value: {ks_pvalue:.4f}")

if ks_pvalue > 0.05:
    print("   ✅ Distributions are similar (p > 0.05)")
else:
    print("   ⚠️ Distributions differ significantly (p < 0.05)")

# Mann-Whitney U test (non-parametric)
u_statistic, u_pvalue = stats.mannwhitneyu(
    csv_overlap_valid["Comptage horaire"], api_overlap_valid[count_col]
)

print("\nMann-Whitney U test:")
print(f"   - Statistic: {u_statistic:.0f}")
print(f"   - P-value: {u_pvalue:.4f}")

if u_pvalue > 0.05:
    print("   ✅ Medians are similar (p > 0.05)")
else:
    print("   ⚠️ Medians differ significantly (p < 0.05)")

# 3.5 Outlier Detection
print("\n3.5 Outlier Detection (IQR method)")
print("-" * 40)


def count_outliers(series, name):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = ((series < lower_bound) | (series > upper_bound)).sum()
    pct = outliers / len(series) * 100

    print(f"{name}:")
    print(f"   - Q1: {Q1:.1f}, Q3: {Q3:.1f}, IQR: {IQR:.1f}")
    print(f"   - Bounds: [{lower_bound:.1f}, {upper_bound:.1f}]")
    print(f"   - Outliers: {outliers:,} ({pct:.1f}%)")

    return outliers, pct


csv_outliers, csv_outlier_pct = count_outliers(
    csv_overlap_valid["Comptage horaire"], "CSV"
)
api_outliers, api_outlier_pct = count_outliers(api_overlap_valid[count_col], "API")

# ========================================
# 4. MATCH-BASED COMPARISON (SAME COUNTER + DATE)
# ========================================

print("\n" + "=" * 80)
print("🔗 4. MATCH-BASED COMPARISON (Same Counter + Hour)")
print("-" * 80)

# Normalize for matching
csv_match = csv_overlap_valid.copy()
csv_match["counter_norm"] = csv_match["Nom du compteur"].str.strip().str.lower()
# Convert to UTC to avoid DST ambiguity issues
csv_match["date_hour"] = (
    csv_match["Date et heure de comptage"].dt.tz_convert("UTC").dt.floor("h")
)

api_match = api_overlap_valid.copy()
api_match["counter_norm"] = api_match[counter_col].str.strip().str.lower()
api_match["date_hour"] = api_match[date_col].dt.tz_convert("UTC").dt.floor("h")

# Merge on counter + date_hour
# Rename columns to avoid suffix issues
csv_match_subset = csv_match[["counter_norm", "date_hour", "Comptage horaire"]].rename(
    columns={"Comptage horaire": "count_csv"}
)
api_match_subset = api_match[["counter_norm", "date_hour", count_col]].rename(
    columns={count_col: "count_api"}
)

merged = pd.merge(
    csv_match_subset, api_match_subset, on=["counter_norm", "date_hour"], how="inner"
)

print(f"\n📊 Matched records: {len(merged):,}")

if len(merged) > 0:
    print(f"   - CSV total: {len(csv_overlap_valid):,}")
    print(f"   - API total: {len(api_overlap_valid):,}")
    print(
        f"   - Match rate: {len(merged)/min(len(csv_overlap_valid), len(api_overlap_valid))*100:.1f}%"
    )

    # Correlation
    correlation = merged["count_csv"].corr(merged["count_api"])
    print("\n📈 Correlation (CSV vs API for matched records):")
    print(f"   - Pearson r: {correlation:.4f}")

    if correlation > 0.9:
        print("   ✅ Very strong correlation (r > 0.9)")
    elif correlation > 0.7:
        print("   ✅ Strong correlation (r > 0.7)")
    elif correlation > 0.5:
        print("   ⚠️ Moderate correlation (r > 0.5)")
    else:
        print("   ❌ Weak correlation (r < 0.5)")

    # Mean Absolute Error
    mae = (merged["count_csv"] - merged["count_api"]).abs().mean()
    mape = (
        (merged["count_csv"] - merged["count_api"]).abs() / merged["count_csv"] * 100
    ).mean()

    print("\n📊 Error Metrics (CSV as reference):")
    print(f"   - MAE: {mae:.2f}")
    print(f"   - MAPE: {mape:.2f}%")

    # Are they significantly different?
    t_stat, t_pvalue = stats.ttest_rel(merged["count_csv"], merged["count_api"])

    print("\n📊 Paired t-test (matched records):")
    print(f"   - t-statistic: {t_stat:.4f}")
    print(f"   - P-value: {t_pvalue:.4f}")

    if t_pvalue > 0.05:
        print("   ✅ No significant difference (p > 0.05)")
    else:
        print("   ⚠️ Significant difference detected (p < 0.05)")
        mean_diff = (merged["count_csv"] - merged["count_api"]).mean()
        print(f"   - Mean difference: {mean_diff:.2f} (CSV - API)")

else:
    print(
        "\n⚠️ No matched records found (different sampling periods or counter naming issues)"
    )

# ========================================
# 5. SUMMARY AND RECOMMENDATIONS
# ========================================

print("\n" + "=" * 80)
print("📝 5. SUMMARY AND RECOMMENDATIONS")
print("-" * 80)

# Quality score
quality_scores = []

# 1. Distribution similarity (KS test)
if ks_pvalue > 0.05:
    quality_scores.append(("Distribution similarity", "✅ Pass", ks_pvalue))
else:
    quality_scores.append(("Distribution similarity", "⚠️ Fail", ks_pvalue))

# 2. Median similarity (Mann-Whitney)
if u_pvalue > 0.05:
    quality_scores.append(("Median similarity", "✅ Pass", u_pvalue))
else:
    quality_scores.append(("Median similarity", "⚠️ Fail", u_pvalue))

# 3. Counter overlap
if overlap_pct > 80:
    quality_scores.append(
        ("Counter overlap", f"✅ Pass ({overlap_pct:.1f}%)", overlap_pct / 100)
    )
elif overlap_pct > 50:
    quality_scores.append(
        ("Counter overlap", f"⚠️ Moderate ({overlap_pct:.1f}%)", overlap_pct / 100)
    )
else:
    quality_scores.append(
        ("Counter overlap", f"❌ Fail ({overlap_pct:.1f}%)", overlap_pct / 100)
    )

# 4. Correlation (if matched records exist)
if len(merged) > 0:
    if correlation > 0.7:
        quality_scores.append(
            ("Correlation (matched)", f"✅ Pass (r={correlation:.2f})", correlation)
        )
    elif correlation > 0.5:
        quality_scores.append(
            ("Correlation (matched)", f"⚠️ Moderate (r={correlation:.2f})", correlation)
        )
    else:
        quality_scores.append(
            ("Correlation (matched)", f"❌ Fail (r={correlation:.2f})", correlation)
        )

print("\n📊 Quality Assessment:")
for test, result, score in quality_scores:
    print(f"   {test:30s} {result}")

# Overall recommendation
pass_count = sum(1 for _, result, _ in quality_scores if "✅" in result)
total_tests = len(quality_scores)

print(f"\n📊 Overall Score: {pass_count}/{total_tests} tests passed")

print("\n💡 RECOMMENDATION:")
if pass_count >= total_tests * 0.75:
    print("   ✅ API data is TRUSTWORTHY for training")
    print("   ✅ Safe to use API data for model fine-tuning")
    print("   ✅ Proceed with hybrid strategy (CSV baseline + API fine-tuning)")
elif pass_count >= total_tests * 0.5:
    print("   ⚠️ API data has SOME DIFFERENCES from CSV")
    print("   ⚠️ Investigate specific differences before using for training")
    print("   ⚠️ Consider using CSV-only for baseline, API for drift detection only")
else:
    print("   ❌ API data has SIGNIFICANT DIFFERENCES from CSV")
    print("   ❌ DO NOT use API data for training without investigation")
    print("   ❌ Stick to CSV-only approach for now")

# ========================================
# 6. EXPORT RESULTS
# ========================================

print("\n" + "=" * 80)
print("💾 6. EXPORT RESULTS")
print("-" * 80)

results = {
    "validation_date": datetime.now().isoformat(),
    "overlap_period": {
        "start": OVERLAP_START,
        "end": OVERLAP_END,
        "days": expected_days,
    },
    "csv_data": {
        "total_records": int(len(csv_overlap)),
        "valid_records": int(len(csv_overlap_valid)),
        "unique_days": int(csv_days),
        "unique_counters": int(len(csv_counters)),
        "mean_comptage": float(csv_overlap_valid["Comptage horaire"].mean()),
        "outliers_count": int(csv_outliers),
        "outliers_pct": float(csv_outlier_pct),
    },
    "api_data": {
        "total_records": int(len(api_overlap)),
        "valid_records": int(len(api_overlap_valid)),
        "unique_days": int(api_days),
        "unique_counters": int(len(api_counters)),
        "mean_count": float(api_overlap_valid[count_col].mean()),
        "outliers_count": int(api_outliers),
        "outliers_pct": float(api_outlier_pct),
    },
    "comparison": {
        "counter_overlap_pct": float(overlap_pct),
        "common_counters": int(len(common_counters)),
        "ks_test": {
            "statistic": float(ks_statistic),
            "pvalue": float(ks_pvalue),
            "passed": bool(ks_pvalue > 0.05),
        },
        "mannwhitney_test": {
            "statistic": float(u_statistic),
            "pvalue": float(u_pvalue),
            "passed": bool(u_pvalue > 0.05),
        },
    },
    "matched_records": {
        "count": int(len(merged)) if len(merged) > 0 else 0,
        "correlation": float(correlation) if len(merged) > 0 else None,
        "mae": float(mae) if len(merged) > 0 else None,
        "mape": float(mape) if len(merged) > 0 else None,
        "ttest_pvalue": float(t_pvalue) if len(merged) > 0 else None,
    },
    "quality_assessment": {
        "tests_passed": pass_count,
        "total_tests": total_tests,
        "score_pct": float(pass_count / total_tests * 100),
        "recommendation": "trustworthy"
        if pass_count >= total_tests * 0.75
        else "investigate"
        if pass_count >= total_tests * 0.5
        else "not_recommended",
    },
}

with open("docs/overlap_data_quality_validation.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ Results exported: docs/overlap_data_quality_validation.json")
print("\nValidation complete! 🎉")
