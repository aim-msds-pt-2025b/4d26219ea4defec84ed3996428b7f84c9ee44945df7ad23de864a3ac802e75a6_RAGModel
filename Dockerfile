# Multi-stage Dockerfile for ML Pipeline
# Stage 1: Build stage with uv for dependency resolution
FROM python:3.12-slim as builder

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files first (for better layer caching)
COPY pyproject.toml uv.lock ./

# Create virtual environment and install dependencies
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies using uv (faster than pip)
RUN uv sync --frozen --no-dev

# Stage 2: Runtime stage - minimal image with only necessary components
FROM python:3.12-slim as runtime

# Create non-root user for security
RUN groupadd -r mluser && useradd -r -g mluser mluser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy source code
COPY src/ ./src/
COPY main.py ./

# Create necessary directories for data persistence
RUN mkdir -p data/raw data/processed models reports && \
    chown -R mluser:mluser /app

# Switch to non-root user
USER mluser

# Set Python path to include src directory
ENV PYTHONPATH="/app"

# Health check to verify container is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import src.config; print('Container healthy')" || exit 1

# Default command to run the ML pipeline
# Can be overridden at runtime for specific tasks
CMD ["python", "src/run_pipeline.py"]
