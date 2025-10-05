"""
Prediction API routes
"""
from fastapi import APIRouter, HTTPException
from models.schemas import (
    TransactionRequest, 
    FraudPredictionResponse, 
    ChargebackPredictionResponse
)
from predictions.fraud import run_fraud_prediction
from predictions.chargeback import predict_chargeback

router = APIRouter(prefix="/predict", tags=["predictions"])

@router.post("/fraud", response_model=FraudPredictionResponse)
def predict_fraud(req: TransactionRequest):
    """Predict fraud for a transaction"""
    try:
        result = run_fraud_prediction(req)
        return FraudPredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chargeback", response_model=ChargebackPredictionResponse)
def predict_chargeback_endpoint(req: TransactionRequest):
    """Predict chargeback for a transaction"""
    try:
        result = predict_chargeback(req)
        return ChargebackPredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/jobs/predict-chargebacks")
def run_chargeback_predictions_job():
    """Background job to predict chargebacks for recent transactions"""
    try:
        from database.connection import db
        from predictions.chargeback import predict_chargeback
        from models.schemas import TransactionRequest
        from datetime import datetime, timedelta
        
        # Get recent transactions that don't have chargeback predictions
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        recent_txns = list(db["transactions"].find({
            "created_at": {"$gte": cutoff_time},
            "transaction_id": {"$nin": [doc["transaction_id"] for doc in db["chargeback_predictions"].find({}, {"transaction_id": 1})]}
        }))
        
        predictions_made = 0
        for txn in recent_txns:
            try:
                req = TransactionRequest(**{
                    "amount": txn.get("amount", 0.0),
                    "currency": txn.get("currency", "usd"),
                    "email": txn.get("email", ""),
                    "ip_address": txn.get("ip_address", ""),
                    "card_country": txn.get("card_country", ""),
                    "billing_country": txn.get("billing_country", ""),
                    "card_brand": txn.get("card_brand", ""),
                    "funding_type": txn.get("funding_type", ""),
                    "fingerprint": txn.get("fingerprint", ""),
                    "risk_score": txn.get("risk_score"),
                    "three_d_secure": txn.get("three_d_secure"),
                    "cvc_check": txn.get("cvc_check"),
                    "address_line1_check": txn.get("address_line1_check"),
                    "postal_code_check": txn.get("postal_code_check"),
                    "outcome_type": txn.get("outcome_type"),
                    "seller_message": txn.get("seller_message"),
                    "network_status": txn.get("network_status")
                })
                
                result = predict_chargeback(req)
                
                # Store prediction
                db["chargeback_predictions"].update_one(
                    {"transaction_id": txn["transaction_id"]},
                    {"$set": {
                        "transaction_id": txn["transaction_id"],
                        "email": txn["email"],
                        "chargeback_predicted": result["chargeback_predicted"],
                        "confidence_score": result["confidence_score"],
                        "chargeback_reason": result["chargeback_reason"],
                        "model_type": result["model_type"],
                        "created_at": datetime.utcnow()
                    }},
                    upsert=True
                )
                
                predictions_made += 1
                
            except Exception as e:
                print(f"Error processing transaction {txn.get('transaction_id')}: {e}")
                continue
        
        return {
            "message": f"Processed {predictions_made} transactions for chargeback prediction",
            "total_transactions": len(recent_txns),
            "predictions_made": predictions_made
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
