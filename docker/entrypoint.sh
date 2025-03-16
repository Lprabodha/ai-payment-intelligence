#!/bin/bash
service cron start

python3 /src/ai_models/subscription_revenue_forecasting.py

tail -f /dev/null

uvicorn src.api:app --host 0.0.0.0 --port 8010 --reload