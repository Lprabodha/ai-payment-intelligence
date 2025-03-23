#!/bin/sh

echo "🚀 Starting Cron and FastAPI..."

touch /src/logs/cron.log
ls -h /sr

cron &


uvicorn api.main:app --host 0.0.0.0 --port 8010 --reload || (pip install --upgrade pip && pip install -r requirements.txt && uvicorn api.main:app --host 0.0.0.0 --port 8010 --reload)