"""
Chargeback prediction logic
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from predictions.models import model_manager
from database.connection import db
from models.schemas import TransactionRequest, ChargebackPredictionResponse

def predict_chargeback(req: TransactionRequest) -> dict:
    """Predict chargeback risk for a transaction"""
    try:
        # Get customer transaction history
        customer_txns = list(db["transactions"].find({"email": req.email}))
        
        if not customer_txns:
            return {
                "chargeback_predicted": False,
                "confidence_score": 0.0,
                "chargeback_reason": "No transaction history available",
                "model_type": "default"
            }
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(customer_txns)
        
        # Feature engineering
        features = _extract_chargeback_features(req, df)
        
        # Get models
        chargeback_pipeline = model_manager.get_model('chargeback_pipeline')
        chargeback_model = model_manager.get_model('chargeback_model')
        chargeback_scaler = model_manager.get_model('chargeback_scaler')
        
        if not chargeback_pipeline and not chargeback_model:
            return {
                "chargeback_predicted": False,
                "confidence_score": 0.0,
                "chargeback_reason": "Model not available - using default prediction",
                "model_type": "default"
            }
        
        # Prepare features for model
        feature_names = [
            'amount_log', 'hour', 'is_weekend', 'past_tx_count_email',
            'past_tx_count_10m', 'past_tx_count_1h', 'past_tx_count_24h',
            'time_between_transactions', 'past_refund_count',
            'customer_refund_ratio_past', 'past_avg_amount',
            'transaction_amount_diff', 'past_chargebacks',
            'country_mismatch', 'ip_address_reuse_before',
            'fingerprint_reuse_before', 'device_ip_pair_reuse_before',
            'email_domain_risk', 'risk_score'
        ]
        
        X = pd.DataFrame([{k: features.get(k, 0) for k in feature_names}], columns=feature_names)
        X = X.replace([np.inf, -np.inf], 0).fillna(0)
        
        # Make prediction
        if chargeback_pipeline:
            confidence = float(chargeback_pipeline.predict_proba(X)[0][1])
            prediction = int(chargeback_pipeline.predict(X)[0])
            model_type = "ensemble"
        else:
            X_scaled = chargeback_scaler.transform(X)
            confidence = float(chargeback_model.predict_proba(X_scaled)[0][1])
            prediction = int(chargeback_model.predict(X_scaled)[0])
            model_type = "legacy"
        
        # Generate reasons
        reasons = _generate_chargeback_reasons(features)
        
        return {
            "chargeback_predicted": bool(prediction),
            "confidence_score": round(confidence, 4),
            "chargeback_reason": ", ".join(reasons) if reasons else "No strong indicators",
            "model_type": model_type
        }
        
    except Exception as e:
        print(f"Chargeback prediction error: {e}")
        return {
            "chargeback_predicted": False,
            "confidence_score": 0.0,
            "chargeback_reason": f"Prediction error: {str(e)}",
            "model_type": "error"
        }

def _extract_chargeback_features(req: TransactionRequest, df: pd.DataFrame) -> dict:
    """Extract chargeback prediction features"""
    try:
        # Basic transaction features
        amount = float(req.amount)
        hour = datetime.utcnow().hour
        day_of_week = datetime.utcnow().weekday()
        
        # Customer history features
        email_txn_count = len(df)
        email_dispute_count = df['disputed'].sum() if 'disputed' in df.columns else 0
        email_refund_count = df['refunded'].sum() if 'refunded' in df.columns else 0
        avg_amount_email = df['amount'].mean() if email_txn_count > 0 else 0
        
        # Time-based features
        now = datetime.utcnow()
        past_10m = now - timedelta(minutes=10)
        past_1h = now - timedelta(hours=1)
        past_24h = now - timedelta(hours=24)
        
        if len(df) > 0:
            df['created_at'] = pd.to_datetime(df['created_at'])
            past_tx_10m = len(df[df['created_at'] >= past_10m])
            past_tx_1h = len(df[df['created_at'] >= past_1h])
            past_tx_24h = len(df[df['created_at'] >= past_24h])
        else:
            past_tx_10m = past_tx_1h = past_tx_24h = 0
        
        # Time between transactions
        if len(df) > 1:
            df_sorted = df.sort_values('created_at')
            time_diffs = df_sorted['created_at'].diff().dt.total_seconds()
            time_between = time_diffs.mean() if len(time_diffs) > 0 else 0
        else:
            time_between = 0
        
        # Risk features
        refund_ratio = (email_refund_count / email_txn_count) if email_txn_count > 0 else 0
        
        # Build features dictionary
        features = {
            'amount_log': float(np.log1p(amount)),
            'hour': int(hour),
            'is_weekend': int(day_of_week >= 5),
            'past_tx_count_email': email_txn_count,
            'past_tx_count_10m': past_tx_10m,
            'past_tx_count_1h': past_tx_1h,
            'past_tx_count_24h': past_tx_24h,
            'time_between_transactions': float(time_between),
            'past_refund_count': email_refund_count,
            'customer_refund_ratio_past': float(refund_ratio),
            'past_avg_amount': float(avg_amount_email),
            'transaction_amount_diff': float(abs(amount - avg_amount_email)),
            'past_chargebacks': int(email_dispute_count),
            'country_mismatch': int(req.card_country != req.billing_country),
            'ip_address_reuse_before': 0,  # Simplified for now
            'fingerprint_reuse_before': 0,  # Simplified for now
            'device_ip_pair_reuse_before': 0,  # Simplified for now
            'email_domain_risk': 0,  # Simplified for now
            'risk_score': float(req.risk_score or 0),
        }
        
        return features
        
    except Exception as e:
        print(f"Chargeback feature extraction error: {e}")
        return {k: 0 for k in [
            'amount_log', 'hour', 'is_weekend', 'past_tx_count_email',
            'past_tx_count_10m', 'past_tx_count_1h', 'past_tx_count_24h',
            'time_between_transactions', 'past_refund_count',
            'customer_refund_ratio_past', 'past_avg_amount',
            'transaction_amount_diff', 'past_chargebacks',
            'country_mismatch', 'ip_address_reuse_before',
            'fingerprint_reuse_before', 'device_ip_pair_reuse_before',
            'email_domain_risk', 'risk_score'
        ]}

def _generate_chargeback_reasons(features: dict) -> list:
    """Generate chargeback reasons based on features"""
    reasons = []
    
    if features['customer_refund_ratio_past'] > 0.5:
        reasons.append("Customer has a high historical refund ratio (>50%)")
    
    if features['device_ip_pair_reuse_before'] > 3:
        reasons.append("Device/IP pair has been reused in multiple transactions")
    
    if features['country_mismatch'] == 1:
        reasons.append("Card country and billing country do not match")
    
    if features['past_tx_count_email'] > 10:
        reasons.append("Email has an unusually high number of transactions")
    
    if features['risk_score'] > 70:
        reasons.append("Stripe risk score is high (>70)")
    
    if features['time_between_transactions'] < 60:
        reasons.append("Very short time between transactions (<60s)")
    
    if features.get('past_chargebacks', 0) > 0:
        reasons.append("Previous chargebacks found for this user")
    
    if features['amount_log'] > 8:
        reasons.append("Transaction amount is extremely high")
    
    if features['past_tx_count_1h'] > 5:
        reasons.append("High transaction frequency in the last hour")
    
    return reasons
