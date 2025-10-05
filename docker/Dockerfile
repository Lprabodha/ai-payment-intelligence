# Multi-stage build for better caching
FROM python:3.11-slim AS base

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/src

# Set working directory
WORKDIR /src

# Install system dependencies
RUN apt-get update && apt-get install -y \
    cron \
    && rm -rf /var/lib/apt/lists/*

# Dependencies stage - this layer is cached unless requirements.txt changes
FROM base AS dependencies

# Copy requirements first for better caching
COPY requirements.txt /src/requirements.txt

# Install Python dependencies with cache optimization
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Application stage
FROM dependencies AS application

# Create necessary directories
RUN mkdir -p /src/data/models /src/model /src/logs

# Copy Docker-specific files
COPY docker/entrypoint.sh /entrypoint.sh
COPY docker/cron /cron

# Make entrypoint executable and setup cron
RUN chmod +x /entrypoint.sh && \
    echo "" >> /cron && \
    crontab /cron

# Copy source code (this layer changes most frequently)
COPY src/ /src/

# Note: Environment variables should be passed via docker run -e or docker-compose
# COPY .env* /src/  # Uncomment if you have a .env file

# Expose port
EXPOSE 8010

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]
