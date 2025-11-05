# MLflow & Model Registry Architecture

**Last Updated**: 2025-11-05

---

## Overview

This project predicts **hourly bike counts in Paris** from raw sensor data. It includes:

* **3 ML models**: Random Forest (regression), Neural Net (regression), RF Classifier (binary classification)
* A processing + training + artifact storage pipeline
* Model tracking via **MLflow** (local/dev or GCS/prod)
* A **Streamlit** application connected to a model registry (`summary.json`) in GCS

---

## Processing Pipelines

### Common Cleaning

* Class `RawCleanerTransformer`
  * Standardizes column names
  * Extracts temporal features (`day`, `hour`, etc.)
  * Parses geographic coordinates
  * Encodes days of the week
  * Cleans `nom_du_compteur`

### Model-Specific Pipelines

| Pipeline                      | Type                   | Architecture                                                        |
| ----------------------------- | ---------------------- | ------------------------------------------------------------------- |
| `RFPipeline`                  | Regression             | sklearn `RandomForestRegressor` + preprocessing `ColumnTransformer` |
| `NNPipeline`                  | Regression             | Keras NN with embedding + scaled features                           |
| `AffluenceClassifierPipeline` | Binary Classification  | sklearn `RandomForestClassifier` + stratified split                 |

---

## Training (`train.py`)

### `dev` vs `prod` Mode

| Mode   | Data                         | MLflow Tracking                               | Artifacts                                                      |
| ------ | ---------------------------- | --------------------------------------------- | -------------------------------------------------------------- |
| `dev`  | Local CSV (`./data/`)        | `http://127.0.0.1:5000` local + `mlruns_dev/` | Local storage in `./models/`                                   |
| `prod` | Data on GCS (`gs://...`)     | Same MLflow, but artifacts = GCS              | Export model + summary to `gs://df_traffic_cyclist1/models/`   |

### Full Training

```bash
python src/train.py --env prod
```

* Records each run in MLflow
* Saves models in `tmp_*`
* Uploads to `gs://df_traffic_cyclist1/models/{model_type}/{timestamp}/`
* Updates `summary.json` registry

### MLflow Tracking: `mlruns` Local vs GCP

The `mlruns/` directory is the **backbone of MLflow tracking**. It contains:

* Training **metadata** (hyperparameters, metrics, tags…)
* Recorded **artifacts** (`.joblib`, `.keras` models, images, logs…)

The project distinguishes two well-isolated environments:

#### 🧪 `dev` Environment

* **Backend Store (local)**:
  * All runs saved in local directory:
    ```text
    ./mlruns_dev/
    └── <experiment_id>/
        └── <run_id>/
            └── meta.yaml, params/, metrics/
    ```

* **Artifact Store (local too)**:
  * Generated artifacts (models, logs) stored in:
    ```text
    ./mlruns_dev/<experiment_id>/<run_id>/artifacts/
    ```

💡 This mode allows local work without cloud dependency.

#### ☁️ `prod` Environment

* **Backend Store (local)**:
  * Metadata still stored locally:
    ```text
    ./mlruns_prod/
    ```

* **Artifact Store (cloud - GCS)**:
  * Artifact files stored in:
    ```text
    gs://df_traffic_cyclist1/mlruns/<experiment_id>/<run_id>/artifacts/
    ```

🎯 Advantage: models and logs are accessible in the cloud, but we keep a local history of all trainings.

### Visualization in MLflow

Whether in `dev` or `prod`, experiments appear in **the same MLflow UI interface**, for example:

```text
http://127.0.0.1:5000/#/experiments/0
```

The difference is in the **artifact access path** displayed:

* `file:///.../mlruns_dev/...` for `dev`
* `gs://.../mlruns/...` for `prod`

---

## Model Registry (`summary.json`)

### Format

```json
{
  "timestamp": "2025-06-18T22:59:26.111358",
  "model_type": "nn",
  "env": "prod",
  "test_mode": true,
  "run_id": "abcd...",
  "r2": 0.71,
  "rmse": 54.9,
  "model_uri": "gs://df_traffic_cyclist1/models/nn/20250619_005924/",
  "is_champion": false
}
```

🧠 This is an **append-only** history that stores all models trained in `prod`.

### Automatically Managed

```python
update_summary(...)
```

---

## Dynamic Model Selection

The Streamlit application (and any consumer) can load the **best model** based on:

* `model_type` (`rf`, `nn`, `rf_class`)
* `metric` (`r2`, `f1_score`, etc.)
* `env` and `test_mode`
* Champion flag (`is_champion=True`)

### Loading Example

```python
from app.model_registry_summary import get_best_model_from_summary

pipeline = get_best_model_from_summary(
    model_type="nn",
    summary_path="gs://df_traffic_cyclist1/models/summary.json",
    metric="r2",
    env="prod",
    test_mode=True
)
```

💡 It downloads artifacts from GCS to `/tmp/`, automatically detects the right
subdirectories (`rf/`, `nn/`, etc.), and reloads the right model via `.load()`.

---

## Streamlit Application

### Features

* Choice between `Random Forest`, `Neural Net`, `RF Classifier (Affluence)`
* Manual prediction mode or batch CSV
* Download prediction file
* Cached model loading from `summary.json`

### Security

* GCP credentials are automatically injected from `st.secrets` or environment variable

---

## Key Design Patterns

* **`summary.json` as static and decentralized registry**
* **Append-only logging** → historical traceability
* **Dynamic "best model" loading** based on metrics
* **Clear separation of `dev` and `prod` modes**
* **Upload to GCS + MLflow tracking = complete audit**

---

## Champion Model System

**Status**: Implemented (Phase 5)

The system now supports explicit champion model designation:

* **Champion promotion**: Models can be explicitly promoted to champion status via `/promote_champion` endpoint
* **Priority loading**: Champion models (`is_champion=True`) are loaded first, regardless of metric scores
* **Metadata caching**: Champion status and model metadata are cached in FastAPI for quick access
* **Audit trail**: All promotions logged to BigQuery and Discord notifications

See [training_strategy.md](./training_strategy.md) for complete training and deployment workflow.

---

**Related Documentation:**

- [training_strategy.md](./training_strategy.md) — Complete training workflow
- [mlflow_cloudsql.md](./mlflow_cloudsql.md) — MLflow setup and troubleshooting
- [dags.md](./dags.md) — Airflow DAG documentation
- [MLOPS_ROADMAP.md](../MLOPS_ROADMAP.md) — Overall MLOps architecture
