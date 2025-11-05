# Demo Script - MLOps Bike Traffic Prediction (2 minutes)

**Presentation Date**: November 7, 2025
**Video Duration**: 2 minutes (fast-forward editing in DaVinci Resolve)

---

## SCENE 1: Infrastructure Startup (0:00 - 0:15)

### Actions

```bash
# Terminal command
./scripts/start-all.sh --with-monitoring
```

### What to Show

- Terminal output scrolling (fast-forward this)
- Docker containers starting up (15 services)

### Voice-over / Text Overlay

> "Production MLOps infrastructure with 15 Docker services: MLflow, Airflow, FastAPI, Prometheus, Grafana"

### Fast-forward to:

- All services healthy (green checkmarks)
- Ready message: `✅ All services running`

---

## SCENE 2: Airflow - Data Pipeline (0:15 - 0:45)

### Actions

**Open Airflow UI**: `http://localhost:8081`

**Trigger DAG 1** (`dag_daily_fetch_data`):
- Click on DAG
- Click "Trigger DAG" button
- Wait for completion (fast-forward this)

**Trigger DAG 2** (`dag_daily_prediction`):
- Click on DAG
- Click "Trigger DAG" button
- Wait for completion (fast-forward this)

### What to Show

1. **Airflow DAG 1** (Data Ingestion):
   - Status: Success (green)
   - Duration: ~30 seconds

2. **Airflow DAG 2** (Predictions):
   - Status: Success (green)
   - Duration: ~45 seconds

3. **Discord Webhook** (switch to Discord):
   - Show notification: "DAG Success: dag_daily_fetch_data"
   - Show notification: "DAG Success: dag_daily_prediction"

4. **BigQuery Console** (quick switch):
   - Dataset `bike_traffic_raw` → table `comptage_velo` (show row count)
   - Dataset `bike_traffic_predictions` (show row count)

### Voice-over / Text Overlay

> "DAG 1: Daily data ingestion from Paris Open Data API → BigQuery"
>
> "DAG 2: ML predictions on last 7 days → BigQuery"
>
> "Discord alerts: Pipeline success ✅"

---

## SCENE 3: Airflow - MLOps Pipeline (0:45 - 1:30)

### Actions

**Trigger DAG 3** (`dag_monitor_and_train`):
- Click on DAG
- Click "Trigger DAG with config"
- JSON config:
  ```json
  {
    "force_fine_tune": true,
    "test_mode": false
  }
  ```
- Click "Trigger"
- **Fast-forward the training** (show progress bar, task completion)

### What to Show

1. **Airflow DAG 3** (Monitor & Train):
   - Task 1: `validate_model` (green)
   - Task 2: `detect_drift` (green)
   - Task 3: `make_decision` → `fine_tune_model` (green)
   - Task 4: `fine_tune_model` (green, FAST-FORWARD)
   - Task 5: `log_audit` (green)

2. **Discord Webhook** (switch to Discord):
   - Show notification: "🏆 Champion Model Promoted"
   - Fields:
     - Model Type: rf
     - R² (current): 0.78
     - R² (baseline): 0.65
     - Improvement: +0.05

3. **BigQuery Console**:
   - Dataset `monitoring_audit` → table `logs`
   - Show latest row with:
     - drift_detected: true
     - fine_tune_triggered: true
     - deployment_decision: "deploy"
     - r2_current: 0.78

### Voice-over / Text Overlay

> "DAG 3: Weekly monitoring & training pipeline"
>
> "1. Validate champion model (R² check)"
>
> "2. Detect drift with Evidently"
>
> "3. Sliding window training (660K baseline + 1.6K fresh data)"
>
> "4. Double evaluation (test_baseline + test_current)"
>
> "5. Deploy new champion ✅"
>
> "Discord alert: Champion promoted with +5% improvement 🏆"

---

## SCENE 4: Grafana Dashboards (1:30 - 2:00)

### Actions

**Open Grafana**: `http://localhost:3000`

**Show 4 dashboards** (5-7 seconds each):

1. **MLOps - Overview**
   - Drift Status: YES (50%)
   - Champion R²: 0.78
   - API Request Rate: graph
   - Services Health: all green

2. **MLOps - Model Performance**
   - R² Trend (champion vs challenger): line graph
   - RMSE: 32.5
   - API Latency P95: 150ms

3. **MLOps - Drift Monitoring**
   - Data Drift Over Time: line graph (50% drift)
   - Drifted Features Count: 15 features

4. **MLOps - Training & Deployment**
   - Training Success Rate: 100%
   - Deployment Decisions: pie chart (deploy/skip/reject)
   - Champion vs Challenger R²: bar chart

### What to Show

- Quick pan through each dashboard (don't linger)
- Focus on KEY METRICS ONLY:
  - Drift: 50%
  - Champion R²: 0.78
  - Training success: 100%

### Voice-over / Text Overlay

> "Real-time monitoring with Grafana:"
>
> "📊 Overview: Drift 50%, Champion R² 0.78, All services healthy"
>
> "📈 Performance: RMSE 32.5, API latency <200ms"
>
> "🔍 Drift: 15 features drifted, model adapted"
>
> "✅ Training: 100% success rate, automated deployment"

---

## FINAL SCENE: Summary (2:00)

### Text Overlay (no voice, just text on black screen)

```
✅ Production MLOps Pipeline

• 3 Airflow DAGs (ingestion, predictions, training)
• Champion/Challenger system with automated deployment
• Sliding window training (660K + 1.6K samples)
• Double evaluation (test_baseline + test_current)
• Real-time monitoring (Prometheus + Grafana)
• Discord alerting for critical events

📊 All metrics tracked • 🏆 Zero-downtime deployments
```

---

## Technical Notes for Video Editing

### Fast-Forward Segments (DaVinci Resolve)

1. **Docker startup** (0:00-0:15):
   - Speed: 4x
   - Show only first 3-5 seconds at normal speed
   - Then jump cut to "all services ready"

2. **DAG 1 & DAG 2 execution** (0:15-0:45):
   - Speed: 8x for task execution
   - Normal speed for: trigger button click, Discord alerts, BigQuery views

3. **DAG 3 training** (0:45-1:30):
   - Speed: 16x for training task (it takes ~20 minutes in reality)
   - Normal speed for: decision tasks, Discord champion alert, BigQuery audit

4. **Grafana dashboards** (1:30-2:00):
   - Speed: 1.5x (slight speed-up for smooth flow)
   - Smooth transitions between dashboards (crossfade 0.5s)

### Camera Focus / Screen Capture

- **Full screen captures** for:
  - Terminal (startup)
  - Airflow UI (DAGs)
  - Discord (alerts)
  - BigQuery (tables)
  - Grafana (dashboards)

- **Picture-in-picture** (optional):
  - Show terminal logs in corner while showing Airflow UI

### Audio

- **Background music**: Low-volume tech/data science track
- **Voice-over**: Clear, concise technical narration (OR text overlays if no voice)
- **Sound effects** (optional):
  - Success chime for Discord alerts
  - Soft "whoosh" for dashboard transitions

---

## Checklist Before Recording

### Pre-requisites

- [ ] All Docker services stopped: `docker compose down -v`
- [ ] Clean slate BigQuery tables (or recent data)
- [ ] Discord webhook configured and tested
- [ ] Grafana dashboards loaded (provisioned)
- [ ] Champion model exists in `summary.json`

### Screen Setup

- [ ] Terminal ready with command: `./scripts/start-all.sh --with-monitoring`
- [ ] Browser tabs pre-opened (but not loaded):
  - Tab 1: `http://localhost:8081` (Airflow)
  - Tab 2: Discord channel with webhook
  - Tab 3: BigQuery console (logged in)
  - Tab 4: `http://localhost:3000` (Grafana, logged in)
- [ ] Screen resolution: 1920x1080 (for clean recording)
- [ ] Browser zoom: 100%
- [ ] Hide bookmarks bar, browser extensions

### Test Run

- [ ] Do a full dry run (without recording) to check timing
- [ ] Verify Discord alerts are sent
- [ ] Verify BigQuery tables populate
- [ ] Verify Grafana dashboards render correctly

---

## Backup Plan (If Training Fails)

### If DAG 3 fails during recording:

1. **Pre-record DAG 3 training** separately (with `test_mode=false`)
2. Use that recording for the demo
3. Splice it into the main video during editing

### If Discord alerts don't show:

1. Use screenshots of previous alerts
2. Overlay them with smooth transitions

### If Grafana dashboards are empty:

1. Use `pushgateway` to inject mock metrics beforehand:
   ```bash
   # Run this before demo
   curl -X POST http://localhost:9091/metrics/job/demo/instance/test \
     --data-binary @mock_metrics.txt
   ```

---

## Post-Production Checklist

- [ ] Color grading (consistent brightness, contrast)
- [ ] Text overlays (clean, readable font like "Roboto Mono")
- [ ] Transitions (smooth, not distracting)
- [ ] Audio levels (balanced, no clipping)
- [ ] Export settings: 1080p, 60fps, H.264
- [ ] Total duration: 2:00 ± 5 seconds

---

**Good luck with the recording! 🎬**
