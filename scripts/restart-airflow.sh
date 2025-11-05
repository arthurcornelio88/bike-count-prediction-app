#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Restarting Airflow Services${NC}"
echo -e "${GREEN}========================================${NC}"

# Stop Airflow services only
echo -e "${YELLOW}Stopping Airflow services...${NC}"
docker compose stop airflow-webserver airflow-scheduler airflow-worker flower airflow-init

# Remove containers
echo -e "${YELLOW}Removing old containers...${NC}"
docker compose rm -f airflow-webserver airflow-scheduler airflow-worker flower airflow-init

# Recreate airflow-init
echo -e "${GREEN}Recreating Airflow init...${NC}"
docker compose up -d airflow-init

# Wait for airflow-init to be healthy
echo -e "${YELLOW}Waiting for Airflow database initialization...${NC}"
MAX_WAIT=90
COUNTER=0

until docker compose ps airflow-init --format json 2>/dev/null | grep -q '"Health":"healthy"' || [ $COUNTER -eq $MAX_WAIT ]; do
    echo -n "."
    sleep 2
    COUNTER=$((COUNTER+1))
done

if [ $COUNTER -eq $MAX_WAIT ]; then
    echo -e "${RED}ERROR: Airflow initialization timed out!${NC}"
    docker compose logs --tail=30 airflow-init
    exit 1
fi

echo -e "${GREEN}Airflow database initialized!${NC}"

# Start Airflow services
echo -e "${GREEN}Starting Airflow services...${NC}"
docker compose up -d airflow-webserver airflow-scheduler airflow-worker flower

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Airflow services restarted!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${GREEN}Airflow UI:${NC} ${YELLOW}http://localhost:8081${NC} (admin/admin)"
echo -e "${GREEN}Flower:${NC}     ${YELLOW}http://localhost:5555${NC}"
echo ""
