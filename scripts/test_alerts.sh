#!/bin/bash
# Test Discord alerts directly

source .env
WEBHOOK="$DISCORD_WEBHOOK_URL"

if [ -z "$WEBHOOK" ]; then
    echo "ERROR: DISCORD_WEBHOOK_URL not set"
    exit 1
fi

echo "=== Testing Discord Alerts ==="

# Test 1: High drift
echo -e "\n1. HIGH DRIFT (>50%)"
curl -X POST "$WEBHOOK" -H "Content-Type: application/json" -d '{
  "embeds": [{
    "title": "🚨 HIGH DRIFT DETECTED",
    "description": "Drift share is at 65%, exceeding threshold of 50%",
    "color": 15158332,
    "fields": [
      {"name": "Severity", "value": "Critical", "inline": true},
      {"name": "Drift Share", "value": "65%", "inline": true}
    ]
  }]
}'

sleep 2

# Test 2: Low R2
echo -e "\n\n2. LOW R2 (<0.65)"
curl -X POST "$WEBHOOK" -H "Content-Type: application/json" -d '{
  "embeds": [{
    "title": "⚠️ LOW MODEL R² SCORE",
    "description": "Champion R² is 0.58, below threshold of 0.65",
    "color": 15158332,
    "fields": [
      {"name": "Severity", "value": "Critical", "inline": true},
      {"name": "R² Score", "value": "0.58", "inline": true}
    ]
  }]
}'

sleep 2

# Test 3: Training failure
echo -e "\n\n3. TRAINING FAILURE"
curl -X POST "$WEBHOOK" -H "Content-Type: application/json" -d '{
  "embeds": [{
    "title": "⚠️ TRAINING FAILURE",
    "description": "3 consecutive training failures detected",
    "color": 16776960,
    "fields": [
      {"name": "Severity", "value": "Warning", "inline": true},
      {"name": "Failures", "value": "3", "inline": true}
    ]
  }]
}'

sleep 2

# Test 4: API errors
echo -e "\n\n4. API ERRORS (>5%)"
curl -X POST "$WEBHOOK" -H "Content-Type: application/json" -d '{
  "embeds": [{
    "title": "⚠️ HIGH API ERROR RATE",
    "description": "FastAPI error rate at 12%, threshold is 5%",
    "color": 16776960,
    "fields": [
      {"name": "Severity", "value": "Warning", "inline": true},
      {"name": "Error Rate", "value": "12%", "inline": true}
    ]
  }]
}'

sleep 2

# Test 5: Service down
echo -e "\n\n5. SERVICE DOWN"
curl -X POST "$WEBHOOK" -H "Content-Type: application/json" -d '{
  "embeds": [{
    "title": "🔴 SERVICE DOWN",
    "description": "FastAPI service has been down for 2 minutes",
    "color": 15158332,
    "fields": [
      {"name": "Severity", "value": "Critical", "inline": true},
      {"name": "Service", "value": "FastAPI", "inline": true}
    ]
  }]
}'

echo -e "\n\n=== All tests sent. Check Discord channel ==="
