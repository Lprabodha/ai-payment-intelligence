#!/bin/sh

echo "🚀 Starting Cron and FastAPI..."

cron &

uvicorn api.main:app --host 0.0.0.0 --port 8010 --reload
