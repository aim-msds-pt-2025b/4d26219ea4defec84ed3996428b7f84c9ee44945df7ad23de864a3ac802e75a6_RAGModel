# Dockerfile for ML Pipeline with Apache Airflow
FROM apache/airflow:2.9.3-python3.12

# Set working directory
WORKDIR /opt/airflow

# Copy requirements file first for better caching
COPY requirements.txt .

# Install ML dependencies using Airflow's recommended approach
USER airflow
RUN pip install --no-cache-dir -r requirements.txt

# Switch to root for file operations
USER root

# Install uv for potential future use
COPY --from=ghcr.io/astral-sh/uv:0.4.18 /uv /bin/uv

# Copy dependency files for documentation
COPY pyproject.toml uv.lock ./

# Copy source code
COPY src/ ./src/
COPY main.py ./
COPY deploy/airflow/dags/ ./dags/
COPY deploy/airflow/plugins/ ./plugins/

# Create necessary directories for data persistence
RUN mkdir -p data/raw data/processed models reports && \
    chown -R airflow:root /opt/airflow/data /opt/airflow/models /opt/airflow/reports /opt/airflow/src /opt/airflow/dags /opt/airflow/plugins /opt/airflow/main.py

# Set Python path to include src directory
ENV PYTHONPATH="/opt/airflow:/opt/airflow/src"

# Switch back to airflow user
USER airflow
