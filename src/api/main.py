import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime
from tensorflow.keras.models import load_model
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

mongo_client = MongoClient(MONGO_URI)
db = mongo_client['payment_intelligence']

router = APIRouter()
app = FastAPI(title="AI Payment Intelligence API")

MODEL_PATH = "/src/data/models/"

def load_ai_model(model_name):
    model_file = os.path.join(MODEL_PATH, model_name)
    if os.path.exists(model_file):
        return joblib.load(model_file)
    else:
        print(f"⚠️ Warning: {model_file} not found!")
        return None  # Avoid crashing FastAPI if model is missing

fraud_model = load_ai_model("fraud_detection_model_final.pkl")
fraud_scaler = load_ai_model("fraud_detection_scaler_final.pkl")

chargeback_model = load_ai_model("chargeback_prediction_model.pkl")
chargeback_scaler = load_ai_model("chargeback_prediction_scaler.pkl")

subscription_model = load_ai_model("subscription_revenue_forecasting_model.pkl")
subscription_scaler = load_ai_model("subscription_revenue_scaler.pkl")

smart_routing_model_path = os.path.join(MODEL_PATH, "smart_payment_routing_model.h5")
if os.path.exists(smart_routing_model_path):
    smart_routing_model = load_model(smart_routing_model_path)
else:
    print(f"⚠️ Warning: {smart_routing_model_path} not found!")
    smart_routing_model = None

class TransactionRequest(BaseModel):
    amount: float
    card_country: str
    billing_country: str
    email: str
    risk_score: float
    ip_address: str
    fingerprint: str
    hour: int


@app.get("/")
def root():
    return {"message": "AI Payment Intelligence API!"}

@router.post("/predict/fraud")
def predict_fraud(req: TransactionRequest):
    if fraud_model is None or fraud_scaler is None:
        raise HTTPException(status_code=500, detail="Fraud model is missing!")
    
    X = fraud_scaler.transform([[req.amount, req.risk_score, req.hour]])
    prediction = fraud_model.predict(X)[0]
    return {"fraud_detected": bool(prediction)}

@router.post("/predict/chargeback")
def predict_chargeback(req: TransactionRequest):
    if chargeback_model is None or chargeback_scaler is None:
        raise HTTPException(status_code=500, detail="Chargeback model is missing!")
    
    X = chargeback_scaler.transform([[req.amount, req.risk_score, req.hour]])
    prediction = chargeback_model.predict(X)[0]
    return {"chargeback_likelihood": float(prediction)}

@router.post("/predict/subscription_revenue")
def predict_subscription_revenue(req: TransactionRequest):
    if subscription_model is None or subscription_scaler is None:
        raise HTTPException(status_code=500, detail="Subscription model is missing!")
    
    X = subscription_scaler.transform([[req.amount, req.risk_score, req.hour]])
    revenue = subscription_model.predict(X)[0]
    return {"expected_revenue": float(revenue)}

@router.post("/predict/payment_gateway")
def predict_payment_gateway(req: TransactionRequest):
    if smart_routing_model is None:
        raise HTTPException(status_code=500, detail="Smart Payment Routing model is missing!")

    X = np.array([req.amount, req.risk_score, req.hour]).reshape(1, -1)
    gateway_idx = np.argmax(smart_routing_model.predict(X, verbose=0)[0])
    gateway_map = {0: "Stripe", 1: "PayPal", 2: "Adyen"}
    return {"recommended_gateway": gateway_map[gateway_idx]}

@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

app.include_router(router)
