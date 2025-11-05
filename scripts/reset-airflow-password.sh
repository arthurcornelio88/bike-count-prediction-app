#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Reset Airflow Admin Password${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if webserver is running
if ! docker compose ps airflow-webserver | grep -q "Up"; then
    echo -e "${RED}ERROR: airflow-webserver is not running!${NC}"
    echo -e "${YELLOW}Start services first with: ./scripts/start-all.sh${NC}"
    exit 1
fi

# Get username and password (or use defaults)
USERNAME=${1:-admin}
PASSWORD=${2:-admin}

echo -e "${YELLOW}Resetting password for user: ${USERNAME}${NC}"

# Try to reset password (if user exists)
if docker compose exec -T airflow-webserver airflow users reset-password \
    --username "$USERNAME" \
    --password "$PASSWORD" 2>/dev/null; then
    echo -e "${GREEN}Password reset successfully!${NC}"
else
    # User doesn't exist, create it
    echo -e "${YELLOW}User doesn't exist, creating...${NC}"
    docker compose exec -T airflow-webserver airflow users create \
        --username "$USERNAME" \
        --firstname "Admin" \
        --lastname "User" \
        --role "Admin" \
        --email "${USERNAME}@example.com" \
        --password "$PASSWORD"
    echo -e "${GREEN}User created successfully!${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}You can now login at:${NC}"
echo -e "${YELLOW}http://localhost:8081${NC}"
echo -e "${GREEN}Username:${NC} $USERNAME"
echo -e "${GREEN}Password:${NC} $PASSWORD"
echo -e "${GREEN}========================================${NC}"
