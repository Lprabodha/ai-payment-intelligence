#!/bin/bash

echo "🚀 Starting AI Payment Intelligence API..."

# Set proper permissions for appuser
chown -R appuser:appuser /src/data /src/logs /src/tmp 2>/dev/null || true

# Start cron daemon
echo "📅 Starting cron daemon..."
cron

# Wait a moment for cron to start
sleep 2

# Start the FastAPI application
echo "🌐 Starting FastAPI server..."
# Change to /src directory and run uvicorn from there
cd /src
exec uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8010 \
    --workers 1 \
    --access-log \
    --log-level info
