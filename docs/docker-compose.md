# 🐳 Docker Compose Architecture

## Overview

The docker-compose stack provides a complete local development environment with:
- **RegModel API** (FastAPI) - port 8000
- **MLflow Tracking Server** - port 5000
- **ClassModel API** (Legacy) - port 8080 (optional)

---

## 🏗️ Architecture Diagram

```
┌──────────────────────────────────────────────────────────┐
│                    Docker Compose Stack                   │
├──────────────────────────────────────────────────────────┤
│                                                            │
│  ┌─────────────────┐         ┌─────────────────────┐     │
│  │   MLflow Server │         │   RegModel API      │     │
│  │   Port: 5000    │◄────────│   Port: 8000        │     │
│  │                 │ depends │                     │     │
│  │ - Tracking UI   │         │ - /predict          │     │
│  │ - Experiments   │         │ - /train            │     │
│  │ - Artifacts     │         │ - /health           │     │
│  └─────────────────┘         └─────────────────────┘     │
│         │                              │                  │
│         │ volumes                      │ volumes          │
│         ▼                              ▼                  │
│  ./mlruns_dev/                  ./backend/regmodel/      │
│  ./mlflow_artifacts/            ./data/                  │
│                                  ./gcp.json               │
└──────────────────────────────────────────────────────────┘
```

---

## 🚀 Usage

### Start the stack

```bash
# Start RegModel API + MLflow
docker compose up

# Build and start (if code changed)
docker compose up --build

# Background mode
docker compose up -d
```

### Stop the stack

```bash
# Stop containers
docker compose down

# Stop and remove volumes
docker compose down -v
```

### View logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f regmodel-backend
docker compose logs -f mlflow
```

### Restart a service

```bash
docker compose restart regmodel-backend
```

---

## 📦 Services

### 1. MLflow Tracking Server

**Container**: `mlflow-server`
**Image**: `ghcr.io/mlflow/mlflow:v2.22.0`
**Port**: `5000`

**Features**:
- File-based backend store (`./mlruns_dev`)
- Local artifact storage (`./mlflow_artifacts`)
- GCS support via mounted credentials
- No healthcheck (lightweight startup)

**Access**:
- UI: http://localhost:5000
- API: http://mlflow:5000 (from containers)

**Volumes**:
```yaml
- ./mlruns_dev:/mlflow/mlruns
- ./mlflow_artifacts:/mlflow/artifacts
- ./gcp.json:/mlflow/gcp.json:ro
```

---

### 2. RegModel API (Main Service)

**Container**: `regmodel-api`
**Port**: `8000`

**Features**:
- FastAPI application with hot reload
- Integrated with MLflow tracking
- GCS model loading/uploading
- DVC data support

**Endpoints**:
- `POST /predict` - Run inference
- `POST /train` - Train and upload model
- `GET /health` - Health check

**Access**:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Redoc: http://localhost:8000/redoc

**Environment variables**:
```env
ENV=DEV
GOOGLE_APPLICATION_CREDENTIALS=/app/gcp.json
MLFLOW_TRACKING_URI=http://mlflow:5000
```

**Volumes**:
```yaml
- ./backend/regmodel/app:/app/app   # Hot reload (app code only)
- ./gcp.json:/app/gcp.json:ro       # GCS credentials
- ./data:/app/data:ro               # DVC datasets (includes test_sample.csv)
```

**Dependencies**: Starts after MLflow (no healthcheck dependency)

---

### 3. ClassModel API (Legacy)

**Container**: `classmodel-api`
**Port**: `8080`
**Profile**: `legacy`

This service is now legacy and only starts when explicitly requested:

```bash
docker compose --profile legacy up
```

---

## 🛠️ Development Workflow

### 1. First time setup

```bash
# Ensure gcp.json exists
ls -la gcp.json

# Ensure data directory exists
mkdir -p data

# Pull DVC data (REQUIRED for training)
dvc pull data/test_sample.csv.dvc         # For test_mode=true (1000 rows, ~1MB)
dvc pull data/reference_data.csv.dvc      # For full training (optional, 980MB)
dvc pull data/current_data.csv.dvc        # For drift monitoring (optional, 430MB)
```

### 2. Start development environment

```bash
docker compose up --build
```

### 3. Test the API

```bash
# Health check
curl http://localhost:8000/health

# MLflow UI
open http://localhost:5000

# Train a model (FAST - test mode recommended for dev)
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "rf",
    "data_source": "reference",
    "env": "dev",
    "test_mode": true
  }'
```

### 4. Make code changes

The regmodel service has hot reload enabled, so code changes are reflected immediately.

### 5. View MLflow experiments

Visit http://localhost:5000 to see tracked experiments, metrics, and artifacts.

---

## 🐛 Troubleshooting

### MLflow healthcheck failing

```bash
# Check MLflow logs
docker compose logs mlflow

# Manually check health endpoint
curl http://localhost:5000/health
```

### RegModel API not starting

```bash
# Check if MLflow is healthy first
docker compose ps

# Check RegModel logs
docker compose logs regmodel-backend

# Rebuild the image
docker compose build regmodel-backend
```

### Permission issues with volumes

```bash
# Fix ownership of mounted directories
sudo chown -R $USER:$USER mlruns_dev mlflow_artifacts
```

### GCS credentials not found

Ensure `gcp.json` exists at the root of the project:

```bash
ls -la gcp.json
# Should output: -rw-r--r-- ... gcp.json
```

---

## 📊 Monitored Directories

| Directory | Purpose | Mounted In |
|-----------|---------|------------|
| `./mlruns_dev/` | MLflow tracking data | mlflow |
| `./mlflow_artifacts/` | Model artifacts (local) | mlflow |
| `./backend/regmodel/` | API source code | regmodel-backend |
| `./data/` | DVC datasets | regmodel-backend |
| `./gcp.json` | GCS credentials | both |

---

## 🔄 Integration with CI/CD

The docker-compose setup is designed for **local development only**. For production:

- RegModel API → Deployed to **Cloud Run**
- MLflow → Can be deployed separately or use managed service
- Data → Stored in **GCS** (`gs://df_traffic_cyclist1/`)

See [backend.md](backend.md) for Cloud Run deployment instructions.

---

## 🎯 Next Steps

After validating the local stack:

1. ✅ Verify MLflow tracking works
2. ✅ Test `/train` endpoint with all model types (rf, nn, rf_class)
3. 🚧 Add Airflow orchestration (Phase 3)
4. 🚧 Add Prometheus + Grafana monitoring (Phase 4)

---

## 📝 Notes

- **Hot reload**: Enabled for regmodel-backend for fast iteration
- **Data isolation**: Each service has its own working directory
- **Network**: All services communicate via Docker's internal network
- **Profiles**: Use `--profile legacy` to start optional services
- **UV optimization**: RegModel uses UV for fast dependency installation (build time ~40s vs ~2min with pip)

---

## 🔗 Related Documentation

- [Backend API](backend.md) - API endpoints and deployment
- [MLOps Roadmap](../MLOPS_ROADMAP.md) - Full MLOps implementation plan
- [Architecture](architecture.md) - System architecture overview
