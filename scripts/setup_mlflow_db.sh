#!/bin/bash
# Setup MLflow Cloud SQL Database
# Run this after Cloud SQL instance is created

set -e

INSTANCE_NAME="mlflow-metadata"
DB_NAME="mlflow"
DB_USER="mlflow_user"
DB_PASSWORD=$(openssl rand -base64 32 | tr -d '\n')

echo "🔧 Setting up MLflow database..."

# Create database
echo "Creating database: $DB_NAME"
gcloud sql databases create $DB_NAME \
    --instance=$INSTANCE_NAME

# Create user
echo "Creating user: $DB_USER"
gcloud sql users create $DB_USER \
    --instance=$INSTANCE_NAME \
    --password="$DB_PASSWORD"

# Get connection name
CONNECTION_NAME=$(gcloud sql instances describe $INSTANCE_NAME \
    --format='value(connectionName)')

echo ""
echo "✅ Database setup complete!"
echo ""
echo "📋 Configuration:"
echo "  Instance: $INSTANCE_NAME"
echo "  Database: $DB_NAME"
echo "  User: $DB_USER"
echo "  Password: $DB_PASSWORD"
echo "  Connection: $CONNECTION_NAME"
echo ""
echo "🔐 Save password to Secret Manager:"
echo "echo -n '$DB_PASSWORD' | gcloud secrets create mlflow-db-password --data-file=-"
echo ""
echo "🐳 Docker connection string:"
echo "postgresql://$DB_USER:$DB_PASSWORD@//$CONNECTION_NAME/$DB_NAME"
echo ""
echo "For Cloud SQL Proxy:"
echo "./cloud-sql-proxy $CONNECTION_NAME"
