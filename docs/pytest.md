# 🧪 Pytest Documentation

## Overview

This project uses **pytest** for automated testing of ML pipelines, preprocessing transformers, and API endpoints.

**Test coverage goal**: >80% of core modules (`src/`, `app/`)

---

## 📁 Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_pipelines.py        # ML pipeline tests (RF, NN) - 13 tests ✅
├── test_preprocessing.py    # Transformer tests - 17 tests ✅
└── test_api_regmodel.py     # FastAPI endpoint tests (TODO)

pytest.ini                   # Pytest configuration
```

---

## 🚀 Running Tests

### Run all tests
```bash
pytest tests/ -v
```

### Run specific test file
```bash
pytest tests/test_pipelines.py -v
```

### Run specific test class
```bash
pytest tests/test_pipelines.py::TestRFPipeline -v
```

### Run specific test function
```bash
pytest tests/test_pipelines.py::TestRFPipeline::test_rf_pipeline_fit -v
```

### Run with coverage
```bash
pytest tests/ --cov=src --cov=app --cov-report=html
```

Coverage report will be generated in `htmlcov/index.html`

---

## 📊 Test Results Summary

### ✅ `test_pipelines.py` — ML Pipeline Tests

**Status**: 13/13 tests passing ✅
**Execution time**: ~20 seconds
**Coverage**: RFPipeline, NNPipeline

![Pytest Pipelines Results](img/pytest_pipelines_results.png)

---

### ✅ `test_preprocessing.py` — Preprocessing Transformers Tests

**Status**: 17/17 tests passing ✅
**Execution time**: ~5 seconds
**Coverage**: RawCleanerTransformer, TimeFeatureTransformer

![Pytest Preprocessing Results](img/pytest_preprocessing_results.png)

---

## 📋 Test Cases Breakdown

### 1️⃣ **TestRFPipeline** (Random Forest)

| Test | Description | Assertions |
|------|-------------|------------|
| `test_rf_pipeline_instantiation` | Verify RF pipeline can be created | Pipeline object exists, has model & cleaner attributes |
| `test_rf_pipeline_fit` | Test training on sample data | Model fitted, n_estimators = 50 |
| `test_rf_pipeline_predict` | Test prediction on new data | Returns 10 predictions, all ≥ 0 (valid bike counts) |
| `test_rf_pipeline_reproducibility` | Ensure same data → same predictions | Predictions match across runs (random_state=42) |
| `test_rf_pipeline_save_load` | Verify model persistence | Save/load cycle preserves predictions |

**Key validations**:
- ✅ Pipeline accepts raw bike traffic data
- ✅ Preprocessing (cleaner, feature engineering) works
- ✅ Predictions are non-negative (bike counts)
- ✅ Model serialization/deserialization

---

### 2️⃣ **TestNNPipeline** (Neural Network)

| Test | Description | Assertions |
|------|-------------|------------|
| `test_nn_pipeline_instantiation` | Verify NN pipeline can be created | Pipeline exists, embedding_dim = 8 |
| `test_nn_pipeline_fit` | Test training with Keras model | Model built, n_features set |
| `test_nn_pipeline_predict` | Test NN predictions | Returns 10 predictions, all ≥ 0 |
| `test_nn_pipeline_embedding_dim` | Test custom embedding dimension | Embedding_dim = 16 applied correctly |
| `test_nn_pipeline_save_load` | Verify model persistence | Predictions match after save/load (tolerance 10%) |
| `test_nn_pipeline_model_architecture` | Validate model structure | 2 inputs (compteur_id + features), params > 0 |

**Key validations**:
- ✅ NN pipeline with embedding layer works
- ✅ Custom hyperparameters (embedding_dim) configurable
- ✅ Keras model saved as `.keras` format
- ✅ Architecture: dual input (categorical + numerical)

---

### 3️⃣ **TestPipelineEdgeCases** (Edge Cases)

| Test | Description | Assertions |
|------|-------------|------------|
| `test_rf_pipeline_with_minimal_data` | Test with 10 samples only | Pipeline handles small datasets |
| `test_nn_pipeline_with_different_batch_sizes` | Test NN with batch_size=[16,32,64] | All batch sizes work correctly |

**Key validations**:
- ✅ Pipelines robust to small datasets
- ✅ NN flexible with batch size variations

---

### 4️⃣ **TestRawCleanerTransformer** (Preprocessing)

| Test | Description | Assertions |
|------|-------------|------------|
| `test_transformer_instantiation` | Verify transformer can be created | Transformer exists, keep_compteur configurable |
| `test_column_name_standardization` | Test column names standardized | Lowercase, underscores, no leading/trailing _ |
| `test_datetime_parsing` | Test datetime parsing + timezone | Temporal features created (heure, mois, jour_semaine) |
| `test_geographic_coordinates_parsing` | Test lat/lon extraction | Latitude/longitude columns created, Paris coordinates |
| `test_unwanted_columns_removed` | Test column cleanup | Removes mois_annee_comptage, identifiant_*, etc. |
| `test_compteur_column_handling` | Test keep_compteur parameter | Keeps or drops nom_du_compteur based on flag |
| `test_column_sorting` | Test column order | Output columns sorted alphabetically |
| `test_transformer_fit_transform_idempotent` | Test reproducibility | Same result on re-run |

**Key validations**:
- ✅ Column standardization (lowercase, underscores)
- ✅ Datetime parsing with timezone (Europe/Paris)
- ✅ Geographic coordinates extraction (lat/lon)
- ✅ Unwanted columns removed
- ✅ Alphabetical column sorting

---

### 5️⃣ **TestTimeFeatureTransformer** (Cyclical Encoding)

| Test | Description | Assertions |
|------|-------------|------------|
| `test_transformer_instantiation` | Verify transformer can be created | Transformer exists |
| `test_cyclical_encoding_heure` | Test hour cyclical encoding | heure_sin/cos created, range [-1,1], midnight/noon validated |
| `test_cyclical_encoding_mois` | Test month cyclical encoding | mois_sin/cos created, range [-1,1] |
| `test_annee_mapping` | Test year mapping | 2024→0, 2025→1 |
| `test_output_columns_preserved` | Test non-transformed columns kept | jour_mois, jour_semaine preserved |
| `test_transformer_with_pipeline` | Test in sklearn Pipeline | Works after RawCleaner |

**Key validations**:
- ✅ Cyclical encoding for hour (sin/cos 24h cycle)
- ✅ Cyclical encoding for month (sin/cos 12-month cycle)
- ✅ Year mapping (2024→0, 2025→1)
- ✅ Original temporal columns removed after transformation

---

### 6️⃣ **TestPreprocessingEdgeCases** (Edge Cases)

| Test | Description | Assertions |
|------|-------------|------------|
| `test_raw_cleaner_with_missing_coordinates` | Test handling of missing coordinates | Transformer doesn't crash, valid rows processed |
| `test_time_transformer_with_edge_hours` | Test boundary hours (0, 6, 12, 18, 23) | All transformations succeed |
| `test_raw_cleaner_column_order_consistency` | Test column order consistency | Same column order regardless of input order |

**Key validations**:
- ✅ Handles missing/null coordinates gracefully
- ✅ Boundary hour values processed correctly
- ✅ Consistent column ordering

---

## 🔧 Fixtures (`conftest.py`)

### Shared Fixtures

| Fixture | Scope | Description |
|---------|-------|-------------|
| `suppress_tf_warnings` | session | Suppress TensorFlow warnings during tests |
| `sample_bike_data` | function | Generate 200 rows of realistic bike traffic data |
| `sample_predictions` | function | Sample payload for API prediction requests |
| `mock_gcs_credentials` | function | Mock GCS credentials for testing |

### Sample Data Structure

```python
{
    'nom_du_compteur': ['Totem 73 boulevard de Sébastopol S-N', ...],
    'Date et heure de comptage': ['2024-04-01 00:00:00+02:00', ...],
    'Coordonnées géographiques': ['48.8672, 2.3501', ...]
}
```

**Note**: `mois_annee_comptage` is **excluded** (removed by `RawCleanerTransformer`)

---

## 📝 Pytest Configuration (`pytest.ini`)

```ini
[pytest]
testpaths = tests
python_files = test_*.py

addopts =
    -v                    # Verbose output
    --strict-markers      # Enforce marker registration
    --tb=short            # Short traceback format
    --disable-warnings    # Suppress deprecation warnings
    -ra                   # Show all test summary info

markers =
    unit: Unit tests
    integration: Integration tests
    slow: Tests > 1 second
    api: API endpoint tests
    pipeline: ML pipeline tests
```

---

## 🔗 Integration with CI/CD

### GitHub Actions workflow (`.github/workflows/ci.yml`)

```yaml
name: MLOps CI

on:
  push:
    branches: [feat/*, master]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Run tests
        run: pytest tests/ -v --cov=src --cov=app --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

---

## 📈 Test Metrics

### Current Status

| Metric | Value |
|--------|-------|
| **Total tests** | 30 |
| **Passing** | 30 ✅ |
| **Failing** | 0 |
| **Execution time** | ~25s |
| **Test files** | 2 (test_pipelines.py, test_preprocessing.py) |
| **Coverage** (estimated) | ~80-85% |

**Breakdown by file**:
- `test_pipelines.py`: 13 tests (~20s)
- `test_preprocessing.py`: 17 tests (~5s)

### Next Steps

- [ ] Add `test_api_regmodel.py` (FastAPI endpoints)
- [ ] Add `test_model_registry.py` (summary.json logic)
- [ ] Run coverage report: `pytest tests/ --cov=src --cov=app --cov-report=html`
- [ ] Add tests after grafana, prometheus, new apis, etc (TODO in the end of the great work)
- [ ] Reach >80% overall coverage

---

## 📚 Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Fixtures Guide](https://docs.pytest.org/en/stable/fixture.html)
- [Pytest Coverage](https://pytest-cov.readthedocs.io/)

---

## ✅ Checklist

- [x] `test_pipelines.py` created (13 tests)
- [x] `test_preprocessing.py` created (17 tests)
- [x] `conftest.py` with shared fixtures
- [x] `pytest.ini` configuration
- [x] All 30 tests passing locally
- [ ] GitHub Actions CI configured
- [ ] Coverage report generated (>80%)
- [ ] API tests added
