#!/bin/bash
set -e

echo "========================================"
echo "Starting Airflow Initialization"
echo "========================================"

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
while ! nc -z postgres 5432; do
  echo "PostgreSQL is unavailable - sleeping"
  sleep 2
done
echo "✓ PostgreSQL is ready!"

# Initialize Airflow database
echo "Initializing Airflow database..."
airflow db migrate

# Create admin user if it doesn't exist
echo "Creating/updating admin user..."
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin123 \
    2>/dev/null || echo "Admin user already exists or created"

echo "========================================"
echo "Airflow Initialization Complete"
echo "========================================"
echo "Starting Airflow Webserver and Scheduler..."
echo ""

# Start webserver in background
airflow webserver --port 8080 &

# Start scheduler in foreground (this keeps container running)
exec airflow scheduler
