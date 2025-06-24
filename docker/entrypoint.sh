#!/bin/sh

echo "🚀 Starting Cron and FastAPI..."

cron &

service cron start

python3 /src/gateways/solidgate_client.py

uvicorn api.main:app --host 0.0.0.0 --port 8010 --reload
