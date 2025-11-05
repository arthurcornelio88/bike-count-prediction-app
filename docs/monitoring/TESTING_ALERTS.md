# Testing Grafana Alerts with Mock Metrics

Complete end-to-end alert testing via Prometheus Pushgateway.

## Architecture

```text
Python script → Pushgateway → Prometheus → Grafana → Discord webhook
```

## Setup

1. **Start Pushgateway**:

   ```bash
   docker compose --profile monitoring up -d pushgateway
   ```

2. **Install dependencies**:

   ```bash
   pip install prometheus-client
   ```

3. **Configure Discord contact point in Grafana**:
   - Go to <http://localhost:3000/alerting/notifications>
   - Add contact point: Discord webhook
   - URL: `${DISCORD_WEBHOOK_URL}` (from env var)

## Usage

**Run all tests**:

```bash
python scripts/test_grafana_alerts.py
```

**Run specific test**:

```bash
python scripts/test_grafana_alerts.py --test drift
python scripts/test_grafana_alerts.py --test r2
python scripts/test_grafana_alerts.py --test combined
```

**Restore normal values**:

```bash
python scripts/test_grafana_alerts.py --test restore
```

## Available Tests

| Test | Metric | Threshold | Alert Time |
|------|--------|-----------|------------|
| `drift` | `bike_drift_share=0.65` | >0.5 | 15min |
| `r2` | `bike_model_r2_champion_current=0.58` | <0.65 | 5min |
| `rmse` | `bike_model_rmse_production=85` | >70 | 5min |
| `combined` | drift=0.55 + r2=0.68 | Combined | 5min |
| `api` | Error rate=12% | >5% | 5min |
| `service` | `up=0` | <1 | 1min |
| `ingestion` | `bike_records_ingested_total=0` | =0 | 36h |

## Monitoring

1. **Pushgateway metrics**: <http://localhost:9091>
2. **Prometheus targets**: <http://localhost:9090/targets>
3. **Prometheus queries**: <http://localhost:9090/graph>
4. **Grafana alerts**: <http://localhost:3000/alerting/list>
5. **Discord channel**: Check webhook notifications

## Example Flow

```bash
# 1. Inject high drift
python scripts/test_grafana_alerts.py --test drift

# 2. Check Prometheus (5s scrape)
curl http://localhost:9090/api/v1/query?query=bike_drift_share
# Returns: 0.65

# 3. Wait 15min for Grafana evaluation

# 4. Check Grafana alerts UI
# Status: Firing → Discord notification sent

# 5. Restore normal
python scripts/test_grafana_alerts.py --test restore
```

## Troubleshooting

**Metrics not appearing in Prometheus**:

```bash
docker compose ps pushgateway  # Check running
curl http://localhost:9091/metrics | grep bike_drift_share
```

**Alerts not firing**:

- Check evaluation interval in [rules.yml](../../monitoring/grafana/provisioning/alerting/rules.yml)
- Verify contact point configured in Grafana
- Check Grafana logs: `docker compose logs grafana`

**Discord not receiving notifications**:

- Verify `DISCORD_WEBHOOK_URL` in `.env`
- Check contact point test in Grafana UI
- Inspect alert history in Grafana
