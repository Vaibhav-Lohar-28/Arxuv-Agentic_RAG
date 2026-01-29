FROM python:3.12-slim

# Set environment variables
ENV AIRFLOW_HOME=/opt/airflow
ENV AIRFLOW_VERSION=2.10.3
ENV PYTHON_VERSION=3.12
ENV CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

# Install system dependencies (including netcat for healthcheck)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        curl \
        git \
        libpq-dev \
        poppler-utils \
        tesseract-ocr \
        netcat-traditional \
        && rm -rf /var/lib/apt/lists/*

# Create airflow user with UID/GID 50000 for cross-platform compatibility
RUN groupadd -r -g 50000 airflow && useradd -r -u 50000 -g airflow -d ${AIRFLOW_HOME} -s /bin/bash airflow

# Create airflow directories with proper ownership
RUN mkdir -p ${AIRFLOW_HOME} && \
    mkdir -p ${AIRFLOW_HOME}/dags && \
    mkdir -p ${AIRFLOW_HOME}/logs && \
    mkdir -p ${AIRFLOW_HOME}/plugins && \
    chown -R 50000:50000 ${AIRFLOW_HOME} && \
    chmod -R 755 ${AIRFLOW_HOME}

# Install Airflow with PostgreSQL support
RUN pip install --no-cache-dir \
    "apache-airflow[postgres]==${AIRFLOW_VERSION}" \
    --constraint "${CONSTRAINT_URL}" \
    psycopg2-binary

# Copy requirements and install project dependencies
COPY requirements-airflow.txt /tmp/requirements-airflow.txt
RUN pip install --no-cache-dir -r /tmp/requirements-airflow.txt

# Create and set up entrypoint script inline
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "========================================"\n\
echo "Starting Airflow Initialization"\n\
echo "========================================"\n\
\n\
# Wait for PostgreSQL to be ready\n\
echo "Waiting for PostgreSQL to be ready..."\n\
while ! nc -z postgres 5432; do\n\
  echo "PostgreSQL is unavailable - sleeping"\n\
  sleep 2\n\
done\n\
echo "✓ PostgreSQL is ready!"\n\
\n\
# Initialize Airflow database\n\
echo "Initializing Airflow database..."\n\
airflow db migrate\n\
\n\
# Create admin user if it does not exist\n\
echo "Creating/updating admin user..."\n\
airflow users create \\\n\
    --username admin \\\n\
    --firstname Admin \\\n\
    --lastname User \\\n\
    --role Admin \\\n\
    --email admin@example.com \\\n\
    --password admin123 \\\n\
    2>/dev/null || echo "Admin user already exists or created"\n\
\n\
echo "========================================"\n\
echo "Airflow Initialization Complete"\n\
echo "========================================"\n\
echo "Starting Airflow Webserver and Scheduler..."\n\
echo ""\n\
\n\
# Start webserver in background\n\
airflow webserver --port 8080 &\n\
\n\
# Start scheduler in foreground (this keeps container running)\n\
exec airflow scheduler\n\
' > /entrypoint.sh && chmod +x /entrypoint.sh

# Switch to airflow user and set working directory
USER airflow
WORKDIR ${AIRFLOW_HOME}

# Expose port
EXPOSE 8080

ENTRYPOINT ["/entrypoint.sh"]
