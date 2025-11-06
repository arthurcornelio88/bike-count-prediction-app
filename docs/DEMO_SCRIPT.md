# Demo Script - MLOps Bike Traffic Prediction (2 minutes)

**Presentation Date**: November 7, 2025
**Video Duration**: 2 minutes 20 seconds (fast-forward editing in DaVinci Resolve)
**Key Feature**: Demonstrates complete champion/challenger lifecycle with validate_new_champion task

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

## SCENE 3: Airflow - MLOps Pipeline (0:45 - 1:45)

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

1. **Airflow DAG 3** (Monitor & Train) - Task Graph View:
   - Task 1: `monitor_drift` (green)
   - Task 2: `validate_model` (green) - validates OLD CHAMPION
   - Task 3: `decide_fine_tune` → branches to `fine_tune_model` (green)
   - Task 4: `fine_tune_model` (green, FAST-FORWARD) - trains CHALLENGER + promotes to NEW CHAMPION
   - Task 5: `validate_new_champion` (green) - validates NEW CHAMPION metrics
   - Task 6: `end_monitoring` (green) - audit logs with NEW CHAMPION metrics

2. **Airflow Task Logs** (quick glimpse):
   - Show `fine_tune_model` logs with:
     - "OLD CHAMPION: abc12345..."
     - "CHALLENGER: xyz98765..."
     - "DECISION: DEPLOY - CHALLENGER improved"
     - "PROMOTING CHALLENGER TO NEW CHAMPION"
   - Show `validate_new_champion` logs with:
     - "Validating NEW CHAMPION: xyz98765..."
     - "NEW CHAMPION Validation Metrics: RMSE: 32.5, R²: 0.78"

3. **Discord Webhook** (switch to Discord):
   - Show notification: "🏆 Champion Model Promoted"
   - Fields:
     - Model Type: rf
     - New Champion Run ID: xyz98765...
     - R² (current): 0.78
     - R² (baseline): 0.65
     - Improvement: +0.05
     - RMSE: 32.5

4. **BigQuery Console**:
   - Dataset `monitoring_audit` → table `logs`
   - Show latest row with:
     - drift_detected: true
     - fine_tune_triggered: true
     - deployment_decision: "deploy"
     - r2_current: 0.78 (NEW CHAMPION metric)
     - r2_baseline: 0.65
     - champion_promoted: true

### Voice-over / Text Overlay

> "DAG 3: Weekly monitoring & training pipeline with automated champion promotion"
>
> "1. Monitor drift with Evidently"
>
> "2. Validate OLD CHAMPION on production data (last 7 days)"
>
> "3. Decision: Hybrid strategy (reactive + proactive)"
>
> "4. Train CHALLENGER with sliding window (660K baseline + 1.6K fresh data)"
>
> "5. Double evaluation: test_baseline (regression check) + test_current (improvement check)"
>
> "6. Automatic promotion: CHALLENGER → NEW CHAMPION if metrics improved"
>
> "7. Validate NEW CHAMPION on production data (fresh metrics for monitoring)"
>
> "8. Audit log: Track OLD → NEW champion transition"
>
> "Discord alert: Champion promoted with +5% improvement 🏆"

---

## SCENE 4: Verify Champion Replacement (1:45 - 1:55)

### Actions

**Return to Airflow UI** and **trigger DAG 2** (`dag_daily_prediction`) again:
- Click on DAG
- Click "Trigger DAG" button
- Wait for completion (fast-forward)

### What to Show

1. **Airflow DAG 2 Logs** (quick glimpse):
   - Show task `predict_last_7_days` logs with:
     - "Loading champion model from summary.json..."
     - "Champion model: rf (run_id: xyz98765..., is_champion: True)"
     - "✅ NEW CHAMPION automatically loaded for predictions"

2. **BigQuery Console**:
   - Dataset `bike_traffic_predictions` → table `daily_YYYYMMDD`
   - Show latest predictions with NEW CHAMPION run_id

3. **FastAPI Docs** (optional):
   - Open `http://localhost:8000/docs`
   - Call `/model_summary` endpoint
   - Show response with NEW CHAMPION marked as `is_champion: true`

### Voice-over / Text Overlay

> "Zero-downtime deployment: New predictions automatically use the promoted champion"
>
> "DAG 2 re-run: Loads NEW CHAMPION from summary.json (xyz98765...)"
>
> "No API restart needed - model registry updated seamlessly"

---

## SCENE 5: Grafana Dashboards (1:55 - 2:15)

### Actions

**Open Grafana**: `http://localhost:3000`

**Show 4 dashboards** (5 seconds each):

1. **MLOps - Overview**
   - Drift Status: YES (50%)
   - Champion R²: 0.78 (NEW CHAMPION metric from validate_new_champion)
   - API Request Rate: graph
   - Services Health: all green

2. **MLOps - Model Performance**
   - R² Trend (champion): line graph showing improvement after promotion
   - RMSE: 32.5 (improved after NEW CHAMPION)
   - API Latency P95: <200ms

3. **MLOps - Drift Monitoring**
   - Data Drift Over Time: line graph (50% drift triggered retraining)
   - Drifted Features Count: 15 features

4. **MLOps - Training & Deployment**
   - Training Success Rate: 100%
   - Deployment Decisions: pie chart (last decision: "deploy")
   - Champion vs Challenger R²: bar chart showing NEW CHAMPION superiority

### What to Show

- Quick pan through each dashboard (don't linger)
- Focus on KEY METRICS ONLY:
  - Drift: 50% (detected)
  - Champion R²: 0.78 (improved from OLD CHAMPION)
  - Training success: 100%
  - Deployment decision: DEPLOY

### Voice-over / Text Overlay

> "Real-time monitoring with Grafana - metrics updated with NEW CHAMPION:"
>
> "📊 Overview: Drift 50% detected → Retraining triggered"
>
> "📈 Performance: Champion R² improved to 0.78, RMSE 32.5"
>
> "🔍 Drift: 15 features drifted → Model adapted automatically"
>
> "✅ Training: 100% success rate, champion promoted seamlessly"

---

## FINAL SCENE: Summary (2:15 - 2:20)

### Text Overlay (no voice, just text on black screen)

```
✅ Production MLOps Pipeline - Full Champion/Challenger Lifecycle

Architecture:
• 3 Airflow DAGs (ingestion, predictions, training)
• Champion/Challenger system with automated promotion
• Sliding window training (660K baseline + 1.6K fresh data)
• Double evaluation (test_baseline + test_current)

Key Features Demonstrated:
• OLD CHAMPION validated on production data (last 7 days)
• CHALLENGER trained with hybrid drift strategy (reactive + proactive)
• Automatic promotion: CHALLENGER → NEW CHAMPION (if metrics improved)
• NEW CHAMPION validated immediately after promotion
• Zero-downtime: Next predictions automatically use NEW CHAMPION

Monitoring & Alerting:
• Real-time metrics (Prometheus + Grafana)
• Discord alerts for champion promotions, drift, and failures
• BigQuery audit logs track OLD → NEW champion transitions

📊 All metrics tracked • 🏆 Zero-downtime deployments • 🔄 Automated lifecycle
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

3. **DAG 3 training** (0:45-1:45):
   - Speed: 16x for training task (it takes ~20 minutes in reality)
   - Normal speed for:
     - Decision tasks and branch logic
     - `fine_tune_model` logs showing OLD → NEW champion transition
     - `validate_new_champion` task execution and logs
     - Discord champion promotion alert
     - BigQuery audit log view

4. **DAG 2 re-run** (1:45-1:55):
   - Speed: 8x for task execution
   - Normal speed for: logs showing NEW CHAMPION loaded, BigQuery predictions

5. **Grafana dashboards** (1:55-2:15):
   - Speed: 1.5x (slight speed-up for smooth flow)
   - Smooth transitions between dashboards (crossfade 0.5s)
   - Highlight champion R² improvement in charts

6. **Final summary** (2:15-2:20):
   - Fade to black with text overlay (5 seconds)

### Camera Focus / Screen Capture

- **Full screen captures** for:
  - Terminal (startup)
  - Airflow UI (DAGs, task graph, task logs)
  - Discord (champion promotion alerts)
  - BigQuery (raw data, predictions, audit logs)
  - Grafana (4 dashboards)
  - FastAPI Docs (optional - model_summary endpoint)

- **Picture-in-picture** (optional):
  - Show terminal logs in corner while showing Airflow UI
  - Show Discord notification overlay when champion is promoted

- **Key moments to capture** (close-up / zoom):
  - Airflow task graph showing `validate_new_champion` task (green)
  - `fine_tune_model` logs: "PROMOTING CHALLENGER TO NEW CHAMPION"
  - `validate_new_champion` logs: "NEW CHAMPION Validation Metrics"
  - BigQuery audit log row with `deployment_decision: "deploy"`
  - Discord alert with champion promotion details

### Audio

- **Background music**: Low-volume tech/data science track
- **Voice-over**: Clear, concise technical narration (OR text overlays if no voice)
- **Sound effects** (optional):
  - Success chime for Discord alerts
  - Soft "whoosh" for dashboard transitions

---

## Demo Strategy: Two Scenarios

### Scenario A: Full Champion Promotion (Recommended for demo)

**Use when**: You want to showcase the complete champion/challenger lifecycle

**Configuration**:
```json
{
  "force_fine_tune": true,
  "test_mode": false
}
```

**What happens**:
1. DAG 3 trains CHALLENGER
2. CHALLENGER beats OLD CHAMPION → automatic promotion
3. `validate_new_champion` runs → fresh metrics
4. DAG 2 re-run loads NEW CHAMPION
5. Grafana shows improved metrics

**Pros**: Shows the full MLOps automation (most impressive)
**Cons**: Takes ~20 minutes (need fast-forward editing)

---

### Scenario B: No Retraining (Skip scenario)

**Use when**: You want to show monitoring without training

**Configuration**: Don't set `force_fine_tune` flag (or set to `false`)

**What happens**:
1. DAG 3 validates OLD CHAMPION
2. No drift detected (or drift but metrics still good)
3. Decision: SKIP fine-tuning
4. `end_monitoring` logs "no retraining needed"
5. DAG 2 re-run still uses OLD CHAMPION

**Pros**: Faster to record (no 20-minute training)
**Cons**: Doesn't showcase champion promotion

---

**Recommendation**: Use Scenario A (full promotion) for the demo to highlight the automated champion/challenger system, which is the key differentiator of this MLOps pipeline.

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
- [ ] Total duration: 2:20 ± 5 seconds
- [ ] Highlight key moments:
  - [ ] `validate_new_champion` task in Airflow graph
  - [ ] Champion promotion logs in `fine_tune_model`
  - [ ] NEW CHAMPION loaded in DAG 2 re-run
  - [ ] Discord champion promotion alert
  - [ ] BigQuery audit log with deployment decision

---

**Good luck with the recording! 🎬**
