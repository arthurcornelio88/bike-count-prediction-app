# Drift Management Strategy

## Overview

This document explains the hybrid drift management strategy implemented in the `monitor_and_fine_tune` DAG.

## Problem Statement

### Challenge: New Bike Counters

The Paris bike counter network continuously expands with new counters. When new counters appear in production data:

1. **Model behavior with `handle_unknown='ignore'`:**
   - OneHotEncoder maps unknown compteurs to zero vectors: `[0, 0, 0, ...]`
   - Model relies solely on geographic (lat/lon) and temporal features
   - Predictions are **approximate** but not catastrophically wrong
   - Performance degrades **gradually** as proportion of new compteurs increases

2. **The dilemma:**
   - **Retrain too often:** High compute cost, unnecessary if model still performs well
   - **Retrain too late:** Extended period of suboptimal predictions

## Solution: Hybrid Strategy

Our strategy combines **proactive** (preventive) and **reactive** (corrective) triggers.

### Decision Matrix

| R² Score | Drift Share | Decision | Rationale |
|----------|-------------|----------|-----------|
| < 0.65 | Any | **RETRAIN (Reactive)** | Critical performance issue |
| 0.65-0.70 | ≥ 50% | **RETRAIN (Proactive)** | High drift + declining metrics |
| 0.65-0.70 | 30-50% | **WAIT** | Moderate drift, metrics acceptable |
| ≥ 0.70 | ≥ 30% | **WAIT** | Model handles drift well |
| ≥ 0.70 | < 30% | **ALL GOOD** | Continue monitoring |

### Thresholds Explained

```python
# Performance thresholds
R2_CRITICAL = 0.65     # Below this → immediate action (reactive)
R2_WARNING = 0.70      # Below this + high drift → preventive action (proactive)
RMSE_THRESHOLD = 60.0  # Above this → immediate action

# Drift thresholds
DRIFT_CRITICAL = 0.5   # 50%+ drift share → critical level
DRIFT_WARNING = 0.3    # 30%+ drift share → warning level
```

### Decision Logic (Priority Order)

#### Priority 0: Force Flag

```python
if force_fine_tune:
    return "fine_tune_model"
```

**Use case:** Testing, manual override

---

#### Priority 1: REACTIVE (Critical Metrics)

```python
if r2 < 0.65 or rmse > 60:
    return "fine_tune_model"
```

**Trigger:** Performance critically poor
**Action:** Immediate retraining
**Rationale:** Model is failing, must fix now regardless of drift

**Example scenario:**

- R² = 0.62 (below critical threshold)
- Drift = 40%
- **Decision:** RETRAIN immediately

---

#### Priority 2: PROACTIVE (High Drift + Declining)

```python
if drift and drift_share >= 0.5 and r2 < 0.70:
    return "fine_tune_model"
```

**Trigger:** High drift (≥50%) + metrics declining
**Action:** Preventive retraining
**Rationale:** Catch degradation early before it becomes critical

**Example scenario:**

- R² = 0.68 (not critical yet, but declining)
- Drift = 52% (critical level)
- **Decision:** RETRAIN proactively to prevent further decline

---

#### Priority 3: WAIT (Significant Drift but OK Metrics)

```python
if drift and drift_share >= 0.3 and r2 >= 0.70:
    return "end_monitoring"
```

**Trigger:** Moderate-to-high drift but good metrics
**Action:** Monitor closely, no retraining yet
**Rationale:** Model handles drift via `handle_unknown='ignore'`, avoid unnecessary cost

**Example scenario:**

- R² = 0.72 (still good)
- Drift = 50% (high)
- **Decision:** WAIT - model performs well despite drift

---

#### Priority 4: ALL GOOD

```python
else:
    return "end_monitoring"
```

**Trigger:** Low drift or no drift, good metrics
**Action:** Continue monitoring
**Rationale:** Everything working as expected

---

## Real-World Example

### Current Production Scenario (2025-11-03)

**Metrics:**

- R² on production data: **0.72** ✅
- RMSE: **32.25** ✅
- Drift detected: **Yes** (50%) ⚠️
- R² on test_baseline: 0.31 (not used for decision)

**Analysis:**

1. Champion R² (0.72) is **above warning threshold** (0.70) ✅
2. RMSE (32.25) is **well below threshold** (60) ✅
3. Drift share (50%) is **at critical level** ⚠️
4. But metrics are **still good** on production data ✅

**Decision:** **WAIT** (Priority 3)

- Model handles new compteurs adequately via `handle_unknown='ignore'`
- Performance remains acceptable despite 50% drift
- Will retrain if:
  - R² drops below 0.70 (proactive trigger)
  - R² drops below 0.65 (reactive trigger)

**Why not retrain now?**

- Retraining is expensive (~20 minutes compute)
- Current model still performs well (R² = 0.72)
- New compteurs are handled (not optimally, but acceptably)
- Cost-benefit analysis favors waiting

---

## Why test_baseline R² is Different

**Question:** "Why is R² on test_baseline (0.31) so different from production R² (0.72)?"

**Answer:** Distribution mismatch between evaluation sets:

| Dataset | R² | Why? |
|---------|-----|------|
| **test_baseline.csv** | 0.31 | Contains **old compteurs** from training time; many no longer active |
| **Production data (BQ)** | 0.72 | Contains **current compteurs** including new ones handled by model |

**Key insight:**

- `test_baseline.csv` is a **fixed reference** for detecting regression
- **Production metrics** (from `validate_model` task) are what matter for decision-making
- We use test_baseline for **comparison** (did new model regress?), not for **decision** (should we retrain?)

---

## Monitoring Best Practices

### 1. Track Trends

Monitor these metrics over time:

- R² on production data (weekly)
- Drift share (weekly)
- Proportion of unknown compteurs

### 2. Adjust Thresholds

If you notice:

- **Frequent unnecessary retraining:** Increase DRIFT_CRITICAL (e.g., 0.6)
- **Late detection of degradation:** Increase R2_WARNING (e.g., 0.75)
- **Too much tolerance for poor performance:** Increase R2_CRITICAL (e.g., 0.70)

### 3. Log Everything

The `monitoring_audit.logs` table in BigQuery tracks:

- Drift detected and drift_share
- R² and RMSE on production
- Retraining decisions and outcomes
- Model improvements

Query for insights:

```sql
SELECT
  timestamp,
  drift_detected,
  r2,
  fine_tune_triggered,
  deployment_decision
FROM `monitoring_audit.logs`
ORDER BY timestamp DESC
LIMIT 10;
```

---

## Future Improvements

### 1. Better Handling of Unknown Compteurs

Instead of `handle_unknown='ignore'`, consider:

- **Target encoding:** Encode compteurs by their average bike traffic
- **Geographic clustering:** Group similar compteurs by location
- **Meta-features:** Use compteur metadata (installation date, location type)

### 2. Adaptive Thresholds

Learn thresholds from historical data:

- Track correlation between drift_share and R² degradation
- Adjust DRIFT_CRITICAL dynamically

### 3. Cost-Aware Decision

Include retraining cost in decision logic:

```python
retraining_cost = compute_hours * cost_per_hour
performance_gain = (new_r2 - current_r2) * business_value
if performance_gain > retraining_cost:
    retrain()
```

---

## Summary

**Key Takeaways:**

1. ✅ **Hybrid strategy** balances cost and performance
2. ✅ **Production metrics** (not test_baseline) drive decisions
3. ✅ **Proactive retraining** catches issues early (50% drift + R² < 0.70)
4. ✅ **Reactive retraining** fixes critical issues (R² < 0.65)
5. ✅ **WAIT decision** avoids unnecessary retraining when model handles drift well

**Current Status:**

- Model performs well (R² = 0.72) despite 50% drift
- Using WAIT strategy to monitor before retraining
- Will retrain proactively if R² drops below 0.70
