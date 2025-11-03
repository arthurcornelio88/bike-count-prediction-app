#!/bin/bash
# Test script for double evaluation in monitor_and_fine_tune DAG
# Usage: ./test_double_evaluation.sh

set -e

echo "=============================================="
echo "🧪 TESTING DOUBLE EVALUATION"
echo "=============================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "\n${YELLOW}Step 1: Checking Airflow status...${NC}"
if ! docker ps | grep -q airflow-webserver; then
    echo -e "${RED}❌ Airflow is not running. Please start it first.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Airflow is running${NC}"

echo -e "\n${YELLOW}Step 2: Triggering DAG with test_mode=true and force_fine_tune=true...${NC}"
echo "This will:"
echo "  - Force fine-tuning (skip drift/validation checks)"
echo "  - Use test mode (skip large test_baseline.csv evaluation)"
echo "  - Enable double evaluation on current data only"

docker exec airflow-webserver airflow dags trigger monitor_and_fine_tune \
  --conf '{"force_fine_tune": true, "test_mode": true}'

echo -e "${GREEN}✅ DAG triggered successfully${NC}"

echo -e "\n${YELLOW}Step 3: Waiting for DAG to start (10 seconds)...${NC}"
sleep 10

echo -e "\n${YELLOW}Step 4: Monitoring DAG execution...${NC}"
echo "Run this command to follow logs:"
echo "  docker logs -f airflow-scheduler"
echo ""
echo "Or check the Airflow UI at:"
echo "  http://localhost:8080"
echo ""
echo "Or check DAG status with:"
echo "  docker exec airflow-webserver airflow dags state monitor_and_fine_tune"

echo -e "\n${YELLOW}Step 5: After completion, check BigQuery audit logs...${NC}"
echo "Expected new columns in monitoring_audit.logs:"
echo "  - double_evaluation_enabled (BOOL)"
echo "  - baseline_regression (BOOL)"
echo "  - r2_baseline (FLOAT)"
echo "  - r2_current (FLOAT)"
echo "  - r2_train (FLOAT)"
echo "  - deployment_decision (STRING)"
echo "  - model_uri (STRING)"
echo "  - run_id (STRING)"

echo -e "\n${GREEN}=============================================="
echo "🎯 Test initiated successfully!"
echo "=============================================="
echo -e "${NC}"
echo "Next steps:"
echo "1. Wait for DAG to complete (~2-3 minutes with test_mode)"
echo "2. Check logs: docker logs -f airflow-scheduler"
echo "3. Verify audit table in BigQuery Console"
echo "4. Check MLflow UI at http://localhost:5000 for run details"
