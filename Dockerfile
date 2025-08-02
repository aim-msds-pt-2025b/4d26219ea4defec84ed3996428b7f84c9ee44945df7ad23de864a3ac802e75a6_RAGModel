# Dockerfile for ML Pipeline with Apache Airflow
FROM apache/airflow:2.9.3-python3.12

# Set working directory
WORKDIR /opt/airflow

# Switch to root to install additional dependencies
USER root

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install additional Python packages for ML pipeline
USER airflow
RUN pip install --no-cache-dir datasets==2.20.0 scikit-learn==1.5.1 pandas==2.2.2
USER root

# Copy source code
COPY src/ ./src/
COPY main.py ./

# Create necessary directories for data persistence
RUN mkdir -p data/raw data/processed models reports && \
    chown -R airflow:root /opt/airflow/data /opt/airflow/models /opt/airflow/reports /opt/airflow/src /opt/airflow/main.py

# Set Python path to include src directory
ENV PYTHONPATH="/opt/airflow:/opt/airflow/src"

# Switch back to airflow user
USER airflow
