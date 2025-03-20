from fastapi import FastAPI
import numpy as np
from datetime import datetime
from fastapi import APIRouter

import os

router = APIRouter()

app = FastAPI(title="AI Payment Intelligence API")

@app.get("/")
def root():
    return {"message": "AI Payment Intelligence API!"}

@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@router.get("/predict/fraud")
def predict_fraud():
    return {"fraud_detected": False, "confidence": 0.85}

@router.get("/predict/chargeback")
def predict_chargeback():
    return {"chargeback_risk": "low", "probability": 0.20}