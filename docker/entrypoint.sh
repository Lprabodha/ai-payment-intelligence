#!/bin/sh

echo "🚀 Starting cron and FastAPI..."

cron &

uvicorn main:app --host 0.0.0.0 --port 8010 --reload
