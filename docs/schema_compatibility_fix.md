# Schema Compatibility Fix - CSV vs BigQuery Formats

**Date**: 2025-11-02
**Issue**: `KeyError: 'Coordonnées géographiques'` during double evaluation
**Status**: ✅ RESOLVED

---

## Problem

The double evaluation feature failed when processing data from BigQuery because:

- **CSV files** (train_baseline.csv, test_baseline.csv): Have combined coordinates in `"Coordonnées géographiques"` column
  (format: `"48.837006, 2.274824"`)
- **BigQuery data** (from DAG current_data): Has separate `latitude` and `longitude` columns

When `evaluate_double()` saved BigQuery data to a temporary CSV
and reloaded it via `load_and_clean_data()`, the function expected the combined format and crashed.

### Error Traceback

```python
KeyError: 'Coordonnées géographiques'
    df["Coordonnées géographiques"].str.split(",", expand=True).astype(float)
```

---

## Root Cause Analysis

The coordinate transformation logic was duplicated in two places:

1. **[train.py:80-82](../backend/regmodel/app/train.py#L80-L82)** - `load_and_clean_data()` function
2. **[classes.py:73-75](../backend/regmodel/app/classes.py#L73-L75)** - `RawCleanerTransformer` class

Both assumed the CSV format with combined coordinates, making the system brittle when data came from different sources (BigQuery).

---

## Solution (Robust Architecture)

### Principle: Single Source of Truth

Move ALL data transformations to the sklearn pipeline transformers, leaving data loaders simple and format-agnostic.

### Changes Applied

#### 1. Simplified `load_and_clean_data()` - [train.py:78-104](../backend/regmodel/app/train.py#L78-L104)

**Before**:

```python
def load_and_clean_data(path: str, preserve_target=False):
    df = pd.read_csv(path, sep=";")
    df[["latitude", "longitude"]] = (
        df["Coordonnées géographiques"].str.split(",", expand=True).astype(float)
    )
    df_clean = df.dropna(subset=["latitude", "longitude"])
    # ...
```

**After**:

```python
def load_and_clean_data(path: str, preserve_target=False):
    """
    Load raw data from CSV without any transformation.
    All transformations are handled by RawCleanerTransformer.
    """
    df = pd.read_csv(path, sep=";")

    # Drop rows with missing target only
    if "Comptage horaire" in df.columns:
        df = df.dropna(subset=["Comptage horaire"])

    # Return raw data - transformer will handle coordinates
    if preserve_target:
        return df

    X = df.drop(columns="Comptage horaire")
    y = df["Comptage horaire"]
    return X, y
```

**Rationale**:

- Single Responsibility: Loader loads, doesn't transform
- Format-agnostic: Works with any CSV structure
- Delegates complexity to transformers

#### 2. Enhanced `RawCleanerTransformer` - [classes.py:72-83](../backend/regmodel/app/classes.py#L72-L83)

**Before**:

```python
# Coordonnées
X[["latitude", "longitude"]] = (
    X["coordonnées_géographiques"].str.split(",", expand=True).astype(float)
)
```

**After**:

```python
# Coordonnées - handle both CSV format (combined) and BigQuery format (separate)
if "coordonnées_géographiques" in X.columns:
    # CSV format: split combined coordinates
    X[["latitude", "longitude"]] = (
        X["coordonnées_géographiques"].str.split(",", expand=True).astype(float)
    )
elif "latitude" not in X.columns or "longitude" not in X.columns:
    # If neither format is present, raise error
    raise ValueError(
        f"Missing coordinate columns. Available columns: {list(X.columns)}"
    )
# If latitude/longitude already exist (BigQuery format), keep them as-is
```

**Rationale**:

- Handles both formats automatically
- Clear error messages if format is unexpected
- After column normalization (lines 49-54), column names are lowercase/underscored

---

## Data Flow

### CSV Training (Baseline)

```text
train_baseline.csv
  ├─ "Coordonnées géographiques": "48.837006, 2.274824"
  └─> load_and_clean_data() → raw DataFrame
       └─> RawCleanerTransformer
            ├─ Normalizes columns: "Coordonnées géographiques" → "coordonnées_géographiques"
            ├─ Splits: "48.837006, 2.274824" → latitude=48.837006, longitude=2.274824
            └─> Features ready for model
```

### BigQuery Fine-Tuning (Production)

```text
BigQuery current_data
  ├─ latitude: 48.837006
  ├─ longitude: 2.274824
  └─> API receives dict → pd.DataFrame
       └─> evaluate_double()
            ├─ Saves to temp CSV (preserves separate columns)
            ├─> load_and_clean_data() → raw DataFrame
            └─> RawCleanerTransformer
                 ├─ Normalizes columns: latitude → latitude, longitude → longitude
                 ├─ Detects separate columns already exist → keeps them
                 └─> Features ready for model
```

### Expected Output

---

## Testing

### Manual Test Script

```bash
# Trigger DAG with force flag to bypass normal logic
docker exec airflow-webserver airflow dags trigger monitor_and_fine_tune \
  --conf '{"force_fine_tune": true}'

# Monitor logs
docker logs regmodel-api --tail 100 -f | grep -E "DOUBLE|📍|🆕|KeyError"
```

### Expected Output

```text
🔬 DOUBLE EVALUATION STRATEGY - RF
📥 Loading test_baseline from data/test_baseline.csv
✅ Baseline test set loaded: 181202 samples
📊 Splitting current data (n=2000)...
   - Train portion (80%): 1600 samples
   - Test portion (20%): 400 samples
📍 Evaluating on test_baseline...
✅ RawCleanerTransformer called  # <-- CSV format handled
🆕 Evaluating on test_current...
✅ RawCleanerTransformer called  # <-- BigQuery format handled
✅ No regression on baseline
```

---

## Benefits

| Aspect | Before | After |
|--------|--------|-------|
| **Maintainability** | Transformation logic in 2 places | Single source of truth (transformer) |
| **Testability** | Hard to test different formats | Easy to test transformer with mock data |
| **Extensibility** | Adding new format = change 2 places | Adding new format = change 1 place |
| **Robustness** | Silent failures on format mismatch | Clear error messages with column listing |
| **Debugging** | Unclear where transformation happens | Pipeline logs show transformer calls |

---

## Related Files

| File | Purpose | Changes |
|------|---------|---------|
| [train.py](../backend/regmodel/app/train.py) | Training orchestration | Simplified `load_and_clean_data()` (lines 78-104) |
| [classes.py](../backend/regmodel/app/classes.py) | Model pipelines | Enhanced `RawCleanerTransformer` (lines 72-83) |
| [dag_monitor_and_train.py](../dags/dag_monitor_and_train.py) | Airflow orchestration | Uses BigQuery separate columns (line 262-277) |
| [training_strategy.md](./training_strategy.md) | Training documentation | Documents double evaluation strategy |

---

## Lessons Learned

1. **Avoid duplication**: Transformation logic should live in transformers, not loaders
2. **Design for variance**: Real-world data comes in multiple formats
3. **Fail explicitly**: Clear error messages save debugging time
4. **Test integration**: Unit tests on transformers + integration tests with real DAGs
5. **Document schema**: Keep a canonical schema doc for all data sources

---

## Future Improvements

- [ ] Add unit tests for `RawCleanerTransformer` with both formats
- [ ] Create schema validation utility to check data before transformation
- [ ] Add Prometheus metrics for schema mismatches (data drift detection)
- [ ] Document expected schemas for all data sources (CSV, BigQuery, API)

---

**References**:

- [Double Evaluation Strategy](./training_strategy.md#double-test-set-evaluation)
- [DAG Documentation](./dags.md#dag-3-monitor-and-fine-tune)
- [MLOps Roadmap](../MLOPS_ROADMAP.md)
