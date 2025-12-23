"""
Fraud detection prediction logic
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from predictions.models import model_manager
from database.connection import db
from utils.helpers import _email_domain_risk, classify_risk_level
from models.schemas import TransactionRequest, FraudPredictionResponse

def process_fraud_workflow(transaction):
    """Process fraud detection workflow for a transaction"""
    try:
        print("Running fraud check for:", transaction.get("transaction_id"))
        
        # Create transaction request from transaction data
        req = TransactionRequest(**{
            "amount": transaction.get("amount", 0.0),
            "currency": transaction.get("currency", "usd"),
            "email": transaction.get("email", ""),
            "ip_address": transaction.get("ip_address", ""),
            "card_country": transaction.get("card_country", ""),
            "billing_country": transaction.get("billing_country", ""),
            "card_brand": transaction.get("card_brand", ""),
            "funding_type": transaction.get("funding_type", ""),
            "fingerprint": transaction.get("fingerprint", ""),
            "risk_score": transaction.get("risk_score"),
            "three_d_secure": transaction.get("three_d_secure"),
            "cvc_check": transaction.get("cvc_check"),
            "address_line1_check": transaction.get("address_line1_check"),
            "postal_code_check": transaction.get("postal_code_check"),
            "outcome_type": transaction.get("outcome_type"),
            "seller_message": transaction.get("seller_message"),
            "network_status": transaction.get("network_status")
        })
        
        # Run fraud prediction
        result = run_fraud_prediction(req)
        
        # Store result in database
        fraud_result = {
            "transaction_id": transaction.get("transaction_id"),
            "is_fraud": result["is_fraud"],
            "confidence_score": result["confidence_score"],
            "risk_level": result["risk_level"],
            "fraud_reasons": result["fraud_reasons"],
            "model_type": result["model_type"],
            "created_at": datetime.utcnow()
        }
        
        db["fraud_results"].update_one(
            {"transaction_id": transaction.get("transaction_id")},
            {"$set": fraud_result},
            upsert=True
        )
        
        print("Fraud result saved.")
        return fraud_result
        
    except Exception as e:
        print(f"Fraud workflow error: {e}")
        return None

def run_fraud_prediction(req: TransactionRequest) -> dict:
    """Run fraud prediction on transaction request"""
    try:
        # Get customer transaction history
        customer_txns = list(db["transactions"].find({"email": req.email}))
        
        if not customer_txns:
            # No history - use default prediction
            return {
                "is_fraud": False,
                "confidence_score": 0.5,
                "risk_level": "medium",
                "fraud_reasons": ["No transaction history available"],
                "model_type": "default"
            }
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(customer_txns)
        
        # Feature engineering
        features = _extract_fraud_features(req, df)
        
        # Get models
        fraud_pipeline = model_manager.get_model('fraud_pipeline')
        fraud_model = model_manager.get_model('fraud_model')
        fraud_scaler = model_manager.get_model('fraud_scaler')
        
        if fraud_pipeline:
            # Use ensemble pipeline
            prediction = fraud_pipeline.predict([features])[0]
            confidence = fraud_pipeline.predict_proba([features])[0][1]
            model_type = "ensemble"
        elif fraud_model and fraud_scaler:
            # Use individual model with scaler
            features_scaled = fraud_scaler.transform([features])
            prediction = fraud_model.predict(features_scaled)[0]
            confidence = fraud_model.predict_proba(features_scaled)[0][1]
            model_type = "legacy"
        else:
            # Fallback prediction
            return {
                "is_fraud": False,
                "confidence_score": 0.5,
                "risk_level": "medium",
                "fraud_reasons": ["Model not available"],
                "model_type": "default"
            }
        
        # Ensure confidence is valid (handle NaN, None, inf)
        import math
        if confidence is None or math.isnan(confidence) or math.isinf(confidence):
            confidence = 0.0
        
        # Clamp confidence between 0 and 1
        confidence = max(0.0, min(1.0, float(confidence)))
        
        # Generate fraud reasons
        fraud_reasons = _generate_fraud_reasons(features, confidence)
        
        # Classify risk level
        risk_level = classify_risk_level(confidence)
        
        return {
            "is_fraud": bool(prediction),
            "confidence_score": round(float(confidence), 4),
            "risk_level": risk_level,
            "fraud_reasons": fraud_reasons,
            "model_type": model_type
        }
        
    except Exception as e:
        print(f"Fraud prediction error: {e}")
        return {
            "is_fraud": False,
            "confidence_score": 0.0,
            "risk_level": "unknown",
            "fraud_reasons": [f"Prediction error: {str(e)}"],
            "model_type": "error"
        }

def _extract_fraud_features(req: TransactionRequest, df: pd.DataFrame) -> list:
    """Extract fraud detection features from transaction and history"""
    try:
        # Basic transaction features
        amount = float(req.amount)
        hour = datetime.utcnow().hour
        day_of_week = datetime.utcnow().weekday()
        
        # Customer history features
        total_txns = len(df)
        avg_amount = df['amount'].mean() if total_txns > 0 else 0
        max_amount = df['amount'].max() if total_txns > 0 else 0
        
        # Time-based features
        recent_txns = df[df['created_at'] >= datetime.utcnow() - timedelta(hours=24)]
        recent_count = len(recent_txns)
        
        # Risk features
        email_domain_risk = _email_domain_risk(req.email)
        
        # Build feature vector (matching model expectations)
        features = [
            np.log1p(amount),                    # amount_log
            hour / 24.0,                         # hour_normalized
            float(day_of_week >= 5),            # is_weekend
            float(total_txns),                   # total_transactions
            float(avg_amount),                   # avg_amount
            float(max_amount),                   # max_amount
            float(recent_count),                 # recent_transactions_24h
            float(amount > avg_amount * 2),      # amount_above_avg
            float(req.risk_score or 0) / 100.0, # risk_score_normalized
            float(email_domain_risk),            # email_domain_risk
            float(req.card_country != req.billing_country), # country_mismatch
            float(len(set(df['ip_address'])) > 1), # multiple_ip_addresses
        ]
        
        return features
        
    except Exception as e:
        print(f"Feature extraction error: {e}")
        # Return default features
        return [0.0] * 12

def _generate_fraud_reasons(features: list, confidence: float) -> list:
    """Generate human-readable fraud reasons based on features"""
    reasons = []
    
    # Amount-based reasons
    if features[0] > 8:  # High amount (log scale)
        reasons.append("Transaction amount is unusually high")
    
    # Time-based reasons
    if features[6] > 5:  # Many recent transactions
        reasons.append("High transaction frequency in last 24 hours")
    
    # Risk-based reasons
    if features[8] > 0.8:  # High risk score
        reasons.append("Payment processor risk score is high")
    
    # Email domain reasons
    if features[9] > 0:  # Suspicious email domain
        reasons.append("Email domain appears suspicious")
    
    # Country mismatch
    if features[10] > 0:  # Country mismatch
        reasons.append("Card country and billing country don't match")
    
    # Multiple IP addresses
    if features[11] > 0:  # Multiple IPs
        reasons.append("Customer has used multiple IP addresses")
    
    # Default reason if no specific indicators
    if not reasons:
        reasons.append("No specific fraud indicators detected")
    
    return reasons
