"""
Chargeback prediction logic with real-world patterns
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from predictions.models import model_manager
from database.connection import db
from models.schemas import TransactionRequest, ChargebackPredictionResponse
from typing import Dict, List, Optional, Tuple
import re

def predict_chargeback(req: TransactionRequest) -> dict:
    """Predict chargeback risk with enhanced validation and security checks"""
    try:
        # Enhanced validation and security checks
        from utils.security_utils import (
            validate_and_sanitize_request,
            calculate_security_score,
            is_high_risk_transaction
        )
        
        # Validate and sanitize request data
        is_valid, sanitized_data, error = validate_and_sanitize_request(req.dict())
        
        if not is_valid:
            return {
                "chargeback_predicted": False,
                "confidence_score": 0.0,
                "chargeback_reason": f"Validation failed: {error}",
                "model_type": "validation_error"
            }
        
        # Check for high-risk transaction patterns
        is_high_risk, risk_reason = is_high_risk_transaction(sanitized_data)
        
        if is_high_risk:
            return {
                "chargeback_predicted": True,
                "confidence_score": 0.8,
                "chargeback_reason": f"High risk transaction detected: {risk_reason}",
                "model_type": "security_block"
            }
        
        # Get customer transaction history
        customer_txns = list(db["transactions"].find({"email": req.email}))
        
        if not customer_txns:
            # No history - use default prediction with security score
            security_score = calculate_security_score(sanitized_data)
            return {
                "chargeback_predicted": security_score > 0.6,
                "confidence_score": security_score,
                "chargeback_reason": f"No transaction history available. Security score: {security_score:.2f}",
                "model_type": "default_with_security"
            }
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(customer_txns)
        
        # Feature engineering with enhanced validation
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

def _calculate_chargeback_risk_factors(req: TransactionRequest, df: pd.DataFrame, now: datetime) -> float:
    """Calculate chargeback-specific risk factors"""
    try:
        risk_score = 0.0
        
        # Transaction timing patterns
        hour = now.hour
        day_of_week = now.weekday()
        
        # Weekend transactions have higher chargeback risk
        if day_of_week >= 5:  # Weekend
            risk_score += 0.1
        
        # Night transactions (higher fraud risk)
        if hour >= 22 or hour <= 6:
            risk_score += 0.15
        
        # Amount patterns
        amount = float(req.amount)
        
        # Round amounts (often used in testing)
        if amount % 100 == 0 and amount > 100:
            risk_score += 0.2
        
        # Common testing amounts
        if amount in [1.00, 5.00, 10.00, 50.00, 100.00]:
            risk_score += 0.1
        
        # Digital goods pricing patterns
        if amount % 0.99 == 0:
            risk_score += 0.05
        
        # Currency patterns
        if req.currency.upper() not in ['USD', 'EUR', 'GBP']:
            risk_score += 0.1
        
        # Customer history patterns
        if len(df) > 0:
            # High refund ratio
            refunds = df['refunded'].sum() if 'refunded' in df.columns else 0
            refund_ratio = refunds / len(df)
            if refund_ratio > 0.3:
                risk_score += 0.3
            elif refund_ratio > 0.1:
                risk_score += 0.1
            
            # Previous chargebacks
            chargebacks = df['disputed'].sum() if 'disputed' in df.columns else 0
            if chargebacks > 0:
                risk_score += 0.4
            
            # High transaction frequency
            recent_24h = df[df['created_at'] >= now - timedelta(hours=24)] if 'created_at' in df.columns else pd.DataFrame()
            if len(recent_24h) > 10:
                risk_score += 0.2
        
        return min(1.0, risk_score)
        
    except Exception as e:
        print(f"Chargeback risk factors calculation error: {e}")
        return 0.2

def _calculate_merchant_chargeback_risk(req: TransactionRequest, df: pd.DataFrame) -> float:
    """Calculate merchant-specific chargeback risk"""
    try:
        risk_score = 0.0
        
        # Industry-specific risk factors
        amount = float(req.amount)
        
        # Digital goods vs physical goods
        if amount < 50 and req.currency.upper() == 'USD':
            risk_score += 0.1  # Digital goods often have higher chargeback rates
        
        # Subscription patterns
        if 'subscription' in str(req).lower():
            risk_score += 0.15  # Subscriptions have higher dispute rates
        
        # Card brand patterns
        if req.card_brand == 'AMEX':
            risk_score += 0.05  # AMEX has different chargeback rules
        
        # Payment method risk
        if req.funding_type == 'debit':
            risk_score += 0.1  # Debit cards have different protections
        
        return min(1.0, risk_score)
        
    except Exception as e:
        print(f"Merchant chargeback risk calculation error: {e}")
        return 0.1

def _calculate_satisfaction_risk(req: TransactionRequest, df: pd.DataFrame) -> float:
    """Calculate customer satisfaction risk indicators"""
    try:
        risk_score = 0.0
        
        if len(df) == 0:
            return 0.2  # New customer risk
        
        # Refund patterns
        refunds = df['refunded'].sum() if 'refunded' in df.columns else 0
        refund_ratio = refunds / len(df)
        
        if refund_ratio > 0.4:
            risk_score += 0.4  # Very high refund rate
        elif refund_ratio > 0.2:
            risk_score += 0.2  # High refund rate
        
        # Customer tenure (new customers more likely to dispute)
        if 'created_at' in df.columns:
            first_txn = df['created_at'].min()
            days_since_first = (datetime.utcnow() - first_txn).days
            if days_since_first < 30:
                risk_score += 0.2  # New customer
            elif days_since_first < 90:
                risk_score += 0.1  # Recent customer
        
        # Transaction frequency patterns
        if len(df) > 0:
            avg_time_between = df['created_at'].diff().mean().total_seconds() / 3600 if len(df) > 1 else 24
            if avg_time_between < 1:  # Very frequent transactions
                risk_score += 0.1
        
        return min(1.0, risk_score)
        
    except Exception as e:
        print(f"Satisfaction risk calculation error: {e}")
        return 0.1

def _calculate_payment_method_risk(req: TransactionRequest, df: pd.DataFrame) -> float:
    """Calculate payment method specific risk"""
    try:
        risk_score = 0.0
        
        # Card brand risk
        if req.card_brand == 'AMEX':
            risk_score += 0.1  # Different chargeback rules
        elif req.card_brand == 'DINERS':
            risk_score += 0.15  # Less common, higher risk
        
        # Funding type risk
        if req.funding_type == 'debit':
            risk_score += 0.1  # Different protections
        elif req.funding_type == 'prepaid':
            risk_score += 0.2  # Higher fraud risk
        
        # 3D Secure status
        if req.three_d_secure == 'not_supported':
            risk_score += 0.2
        elif req.three_d_secure == 'failed':
            risk_score += 0.3
        elif req.three_d_secure == 'authenticated':
            risk_score -= 0.1  # Reduces risk
        
        # CVC check
        if req.cvc_check == 'fail':
            risk_score += 0.2
        elif req.cvc_check == 'pass':
            risk_score -= 0.05
        
        # Address verification
        if req.address_line1_check == 'fail':
            risk_score += 0.1
        if req.postal_code_check == 'fail':
            risk_score += 0.1
        
        return max(0.0, min(1.0, risk_score))
        
    except Exception as e:
        print(f"Payment method risk calculation error: {e}")
        return 0.1

def _calculate_geographic_risk(req: TransactionRequest, df: pd.DataFrame) -> float:
    """Calculate geographic risk factors"""
    try:
        risk_score = 0.0
        
        # Country mismatch
        if req.card_country != req.billing_country:
            risk_score += 0.2
        
        # High-risk countries (simplified list)
        high_risk_countries = ['NG', 'PK', 'BD', 'VN', 'ID', 'PH', 'KE', 'GH', 'MA']
        if req.card_country in high_risk_countries:
            risk_score += 0.2
        
        # IP geolocation mismatch
        if req.ip_address:
            # Simplified IP risk (in real implementation, use IP geolocation service)
            if req.card_country == 'US' and not req.ip_address.startswith(('192.168.', '10.', '172.')):
                # Non-US IP with US card
                risk_score += 0.1
        
        return min(1.0, risk_score)
        
    except Exception as e:
        print(f"Geographic risk calculation error: {e}")
        return 0.1

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
    
    # Enhanced risk factors (if available)
    if features.get('chargeback_risk_factors', 0) > 0.7:
        reasons.append("Multiple chargeback risk factors detected")
    elif features.get('chargeback_risk_factors', 0) > 0.4:
        reasons.append("Several chargeback risk factors present")
    
    if features.get('merchant_chargeback_risk', 0) > 0.3:
        reasons.append("Merchant-specific chargeback risk indicators")
    
    if features.get('satisfaction_risk', 0) > 0.5:
        reasons.append("Customer satisfaction risk indicators present")
    
    if features.get('payment_method_risk', 0) > 0.3:
        reasons.append("Payment method risk factors detected")
    
    if features.get('geographic_risk', 0) > 0.4:
        reasons.append("Geographic risk factors present")
    
    # Default reasons
    if not reasons:
        reasons.append("No strong chargeback indicators detected")
    
    # Limit to most significant reasons
    return reasons[:5]
