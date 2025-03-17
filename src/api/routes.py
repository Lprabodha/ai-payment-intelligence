# src/api/routes.py

from fastapi import APIRouter
from pydantic import BaseModel
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from pymongo import MongoClient
from datetime import datetime
import os

fraud_model = joblib.load("/src/data/models/fraud_detection_model_final.pkl")
fraud_scaler = joblib.load("/src/data/models/fraud_detection_scaler_final.pkl")

chargeback_model = joblib.load("/src/data/models/chargeback_prediction_model.pkl")
chargeback_scaler = joblib.load("/src/data/models/chargeback_prediction_scaler.pkl")

subscription_model = joblib.load("/src/data/models/subscription_revenue_forecasting_model.pkl")
subscription_scaler = joblib.load("/src/data/models/subscription_revenue_scaler.pkl")

smart_routing_model = load_model("/src/data/models/smart_payment_routing_model.h5")

MONGO_URI = os.getenv("MONGO_URI")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client['payment_intelligence']
transactions_collection = db['transactions']
subscriptions_collection = db['subscriptions']

router = APIRouter()

class GenericRequest(BaseModel):
    features: list
    customer_id: str = None  # Optional, for customer-based analysis

@router.post("/predict/fraud")
def predict_fraud(req: GenericRequest):
    X = fraud_scaler.transform([req.features])
    prediction = fraud_model.predict(X)[0]
    return {"fraud_detected": bool(prediction)}

@router.post("/predict/chargeback")
def predict_chargeback(req: GenericRequest):
    X = chargeback_scaler.transform([req.features])
    prediction = chargeback_model.predict(X)[0]
    return {"chargeback_likelihood": float(prediction)}

@router.post("/predict/subscription_revenue")
def predict_subscription(req: GenericRequest):
    X = subscription_scaler.transform([req.features])
    revenue = subscription_model.predict(X)[0]
    return {"expected_revenue": float(revenue)}

@router.post("/predict/payment_gateway")
def predict_payment_gateway(req: GenericRequest):
    X = np.array(req.features).reshape(1, -1)
    gateway_idx = np.argmax(smart_routing_model.predict(X, verbose=0)[0])
    gateway_map = {0: "Stripe", 1: "PayPal", 2: "Adyen"}
    return {"recommended_gateway": gateway_map[gateway_idx]}

@router.get("/stats/revenue")
def get_revenue_stats():
    pipeline = [
        {"$group": {"_id": {"$month": "$created_at"}, "total": {"$sum": "$price_amount"}}},
        {"$sort": {"_id": 1}}
    ]
    revenue_stats = list(subscriptions_collection.aggregate(pipeline))
    return {"monthly_revenue": revenue_stats}

@router.get("/stats/fraud_trends")
def get_fraud_stats():
    pipeline = [
        {"$group": {"_id": {"$month": "$created_at"}, "fraud_count": {"$sum": "$disputed"}}},
        {"$sort": {"_id": 1}}
    ]
    fraud_stats = list(transactions_collection.aggregate(pipeline))
    return {"monthly_fraud": fraud_stats}

@router.get("/stats/gateway_usage")
def get_gateway_usage():
    pipeline = [
        {"$group": {"_id": "$gateway", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    gateways = list(transactions_collection.aggregate(pipeline))
    return {"gateway_usage": gateways}

@router.get("/predict/customer_revenue/{customer_id}")
def predict_customer_revenue(customer_id: str):
    customer_subs = list(subscriptions_collection.find({"email": customer_id}))
    total_value = sum([sub['price_amount'] for sub in customer_subs])
    likely_churn = len(customer_subs) < 2  # Mock logic: if <2 subscriptions, at ris
    return {
        "customer_id": customer_id,
        "total_subscription_value": total_value,
        "likely_churn": likely_churn
    }

@router.get("/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}
