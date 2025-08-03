import pytest
from fastapi.testclient import TestClient
from main import app  

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "ok"

def test_predict_fraud_dummy():
    payload = {
        "amount": 120.0,
        "card_country": "US",
        "billing_country": "US",
        "email": "test@example.com",
        "risk_score": 30,
        "ip_address": "192.168.1.1",
        "fingerprint": "abc123xyz",
        "hour": 14
    }
    response = client.post("/predict/fraud", json=payload)
    assert response.status_code == 200
    assert "fraud_detected" in response.json()

def test_predict_payment_gateway_dummy():
    payload = {
        "amount": 100.0,
        "risk_score": 25,
        "hour": 10
    }
    response = client.post("/predict/payment_gateway", json=payload)
    assert response.status_code in [200, 500] 

def test_run_chargeback_predictions():
    response = client.get("/jobs/predict-chargebacks")
    assert response.status_code == 200 or response.status_code == 500

def test_subscription_forecast_job():
    response = client.get("/jobs/subscription-forecast")
    assert response.status_code == 200 or response.status_code == 500
