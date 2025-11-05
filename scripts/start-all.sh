#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Starting MLOps Stack${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if .env files exist
if [ ! -f .env ]; then
    echo -e "${RED}ERROR: .env file not found!${NC}"
    exit 1
fi

if [ ! -f backend/regmodel/.env ]; then
    echo -e "${YELLOW}WARNING: backend/regmodel/.env not found${NC}"
fi

# Stop any existing containers
echo -e "${YELLOW}Stopping existing containers...${NC}"
docker compose down

# Clean up orphaned containers
echo -e "${YELLOW}Removing orphaned containers...${NC}"
docker compose down --remove-orphans

# Start infrastructure services first (without monitoring profile by default)
echo -e "${GREEN}Starting infrastructure services...${NC}"
docker compose up -d cloud-sql-proxy postgres-airflow redis-airflow

# Wait for postgres to be healthy
echo -e "${YELLOW}Waiting for PostgreSQL to be ready...${NC}"
until docker compose exec -T postgres-airflow pg_isready -U airflow &>/dev/null; do
    echo -n "."
    sleep 1
done
echo -e "${GREEN}PostgreSQL is ready!${NC}"

# Start MLflow and RegModel API
echo -e "${GREEN}Starting MLflow and RegModel API...${NC}"
docker compose up -d mlflow regmodel-backend

# Wait a bit for MLflow to be accessible
echo -e "${YELLOW}Waiting for MLflow to be ready...${NC}"
sleep 10

# Start Airflow initialization
echo -e "${GREEN}Starting Airflow initialization...${NC}"
docker compose up -d airflow-init

# Wait for airflow-init to be healthy
echo -e "${YELLOW}Waiting for Airflow database initialization (this may take 1-2 minutes)...${NC}"
MAX_WAIT=90
COUNTER=0

# Show logs in background
docker compose logs -f airflow-init &
LOG_PID=$!

until docker compose ps airflow-init --format json 2>/dev/null | grep -q '"Health":"healthy"' || [ $COUNTER -eq $MAX_WAIT ]; do
    sleep 2
    COUNTER=$((COUNTER+1))

    # Show progress every 10 seconds
    if [ $((COUNTER % 5)) -eq 0 ]; then
        echo -e "${YELLOW}Still waiting... ($((COUNTER*2))s elapsed)${NC}"
    fi
done

# Stop log streaming
kill $LOG_PID 2>/dev/null || true

if [ $COUNTER -eq $MAX_WAIT ]; then
    echo -e "${RED}ERROR: Airflow initialization timed out after $((MAX_WAIT*2)) seconds!${NC}"
    echo -e "${YELLOW}Last logs from airflow-init:${NC}"
    docker compose logs --tail=50 airflow-init
    exit 1
fi

echo -e "${GREEN}Airflow database initialized!${NC}"

# Start all Airflow services
echo -e "${GREEN}Starting Airflow services...${NC}"
docker compose up -d airflow-webserver airflow-scheduler airflow-worker flower

# Check if monitoring profile is requested
if [ "$1" == "--with-monitoring" ] || [ "$1" == "-m" ]; then
    echo -e "${GREEN}Starting monitoring stack (Prometheus, Grafana, Airflow Exporter)...${NC}"
    docker compose --profile monitoring up -d
fi

# Display status
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  All services started!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${GREEN}Services available at:${NC}"
echo -e "  - MLflow:         ${YELLOW}http://localhost:5000${NC}"
echo -e "  - RegModel API:   ${YELLOW}http://localhost:8000${NC}"
echo -e "  - Airflow UI:     ${YELLOW}http://localhost:8081${NC} (admin/admin)"
echo -e "  - Flower:         ${YELLOW}http://localhost:5555${NC}"

if [ "$1" == "--with-monitoring" ] || [ "$1" == "-m" ]; then
    echo -e "  - Prometheus:     ${YELLOW}http://localhost:9090${NC}"
    echo -e "  - Grafana:        ${YELLOW}http://localhost:3000${NC}"
fi

echo ""
echo -e "${YELLOW}View logs with:${NC} docker compose logs -f [service-name]"
echo -e "${YELLOW}Stop all with:${NC} docker compose down"
echo ""
