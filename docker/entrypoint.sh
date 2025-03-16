#!/bin/bash
service cron start

python3 /src/gateways/stripe_client.py

tail -f /dev/null

uvicorn src.api:app --host 0.0.0.0 --port 8010 --reload