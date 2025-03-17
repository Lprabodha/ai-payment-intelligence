#!/bin/sh

echo "🚀 Starting Cron and FastAPI..."

cron &

uvicorn src.api.routes:app --host 0.0.0.0 --port 8010 --reload
