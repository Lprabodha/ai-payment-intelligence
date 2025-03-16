#!/bin/bash
service cron start

python3 /src/ai_models/chargeback_prediction.py

tail -f /dev/null

uvicorn src.api:app --host 0.0.0.0 --port 8010 --reload