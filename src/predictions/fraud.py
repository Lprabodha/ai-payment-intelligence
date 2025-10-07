"""
Fraud detection prediction logic with real-world patterns
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from predictions.models import model_manager
from database.connection import db
from utils.helpers import _email_domain_risk, classify_risk_level
from models.schemas import TransactionRequest, FraudPredictionResponse
from utils.validation import (
    validate_transaction_data, 
    EmailValidator, 
    IPValidator, 
    AmountValidator,
    SecurityValidator
)
import hashlib
import re
from typing import Dict, List, Optional, Tuple

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
    """Run fraud prediction with enhanced validation and security checks"""
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
                "is_fraud": False,
                "confidence_score": 0.0,
                "risk_level": "low",
                "fraud_reasons": [f"Validation failed: {error}"],
                "model_type": "validation_error"
            }
        
        # Check for high-risk transaction patterns
        is_high_risk, risk_reason = is_high_risk_transaction(sanitized_data)
        
        if is_high_risk:
            return {
                "is_fraud": True,
                "confidence_score": 0.9,
                "risk_level": "high",
                "fraud_reasons": [f"High risk transaction detected: {risk_reason}"],
                "model_type": "security_block"
            }
        
        # Get customer transaction history
        customer_txns = list(db["transactions"].find({"email": req.email}))
        
        if not customer_txns:
            # No history - use default prediction with security score
            security_score = calculate_security_score(sanitized_data)
            return {
                "is_fraud": security_score > 0.7,
                "confidence_score": security_score,
                "risk_level": "high" if security_score > 0.7 else "medium",
                "fraud_reasons": ["No transaction history available", f"Security score: {security_score:.2f}"],
                "model_type": "default_with_security"
            }
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(customer_txns)
        
        # Feature engineering with enhanced validation
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
    """Extract comprehensive fraud detection features with real-world patterns"""
    try:
        # Basic transaction features
        amount = float(req.amount)
        hour = datetime.utcnow().hour
        day_of_week = datetime.utcnow().weekday()
        minute = datetime.utcnow().minute
        
        # Customer history features
        total_txns = len(df)
        avg_amount = df['amount'].mean() if total_txns > 0 else 0
        max_amount = df['amount'].max() if total_txns > 0 else 0
        min_amount = df['amount'].min() if total_txns > 0 else 0
        
        # Time-based features with real-world patterns
        now = datetime.utcnow()
        recent_1h = df[df['created_at'] >= now - timedelta(hours=1)]
        recent_24h = df[df['created_at'] >= now - timedelta(hours=24)]
        recent_7d = df[df['created_at'] >= now - timedelta(days=7)]
        recent_30d = df[df['created_at'] >= now - timedelta(days=30)]
        
        recent_1h_count = len(recent_1h)
        recent_24h_count = len(recent_24h)
        recent_7d_count = len(recent_7d)
        recent_30d_count = len(recent_30d)
        
        # Advanced velocity patterns
        amount_1h = recent_1h['amount'].sum() if recent_1h_count > 0 else 0
        amount_24h = recent_24h['amount'].sum() if recent_24h_count > 0 else 0
        amount_7d = recent_7d['amount'].sum() if recent_7d_count > 0 else 0
        amount_30d = recent_30d['amount'].sum() if recent_30d_count > 0 else 0
        
        # Real-world risk features
        email_domain_risk = _email_domain_risk(req.email)
        device_fingerprint_risk = _calculate_device_risk(req.fingerprint, req.ip_address, df)
        behavioral_pattern_risk = _calculate_behavioral_risk(req, df, now)
        
        # Enhanced pattern detection
        ip_reputation_risk = _calculate_ip_reputation_risk(req.ip_address)
        amount_testing_risk = _detect_amount_testing_pattern(req, df)
        time_anomaly_risk = _detect_time_anomaly_pattern(req, now)
        card_bin_risk = _calculate_card_bin_risk(req.card_brand)
        
        # Geographic and device diversity
        unique_countries = len(set(df['card_country'].dropna())) if total_txns > 0 else 0
        unique_ips = len(set(df['ip_address'].dropna())) if total_txns > 0 else 0
        unique_cards = len(set(df['fingerprint'].dropna())) if total_txns > 0 else 0
        
        # Time pattern analysis
        hours_since_last = 24  # Default
        if total_txns > 0 and 'created_at' in df.columns:
            last_txn_time = df['created_at'].max()
            hours_since_last = (now - last_txn_time).total_seconds() / 3600
        
        # Merchant-specific patterns
        merchant_risk = _calculate_merchant_risk(req, df)
        
        # Build enhanced feature vector (39 features)
        features = [
            # Basic transaction features (1-5)
            np.log1p(amount),                    # 1: amount_log
            amount,                              # 2: amount_raw
            hour,                                # 3: hour
            day_of_week,                         # 4: day_of_week
            minute,                              # 5: minute
            
            # Time-based features (6-12)
            float(day_of_week >= 5),            # 6: is_weekend
            float(hour >= 9 and hour <= 17),    # 7: is_business_hours
            float(hour >= 18 or hour <= 6),     # 8: is_evening_night
            float(hour < 6),                     # 9: is_night
            float(hour >= 6 and hour < 12),     # 10: is_morning
            float(hour >= 12 and hour < 18),    # 11: is_afternoon
            float(hour >= 18),                   # 12: is_evening
            
            # Customer history features (13-20)
            float(total_txns),                   # 13: total_transactions
            float(avg_amount),                   # 14: avg_amount
            float(max_amount),                   # 15: max_amount
            float(min_amount),                   # 16: min_amount
            float(amount / max(avg_amount, 1)),  # 17: amount_vs_avg_ratio
            float(amount > avg_amount * 2),      # 18: amount_above_avg_2x
            float(amount > avg_amount * 5),      # 19: amount_above_avg_5x
            float(amount > max_amount),          # 20: amount_above_max
            
            # Recent transaction counts (21-27)
            float(recent_1h_count),              # 21: transactions_1h
            float(recent_24h_count),             # 22: transactions_24h
            float(recent_7d_count),              # 23: transactions_7d
            float(recent_1h_count / max(hour + 1, 1)), # 24: tx_rate_1h
            float(recent_24h_count / 24),        # 25: tx_rate_24h
            float(recent_7d_count / 7),          # 26: tx_rate_7d
            float(hours_since_last),             # 27: hours_since_last
            
            # Recent amount features (28-32)
            float(amount_1h),                    # 28: amount_1h
            float(amount_24h),                   # 29: amount_24h
            float(amount_7d),                    # 30: amount_7d
            float(amount_1h / max(recent_1h_count, 1)), # 31: avg_amount_1h
            float(amount_24h / max(recent_24h_count, 1)), # 32: avg_amount_24h
            
            # Enhanced risk features (33-39)
            float(req.risk_score or 0) / 100.0, # 33: risk_score_normalized
            float(email_domain_risk),            # 34: email_domain_risk
            float(req.card_country != req.billing_country), # 35: country_mismatch
            float(device_fingerprint_risk),      # 36: device_risk
            float(behavioral_pattern_risk),      # 37: behavioral_risk
            float(merchant_risk),                # 38: merchant_risk
            float(unique_countries),             # 39: geographic_diversity
        ]
        
        return features
        
    except Exception as e:
        print(f"Feature extraction error: {e}")
        # Return default features with 39 elements
        return [0.0] * 39

def _calculate_device_risk(fingerprint: str, ip_address: str, df: pd.DataFrame) -> float:
    """Calculate device fingerprint risk based on real-world patterns"""
    try:
        if not fingerprint or not ip_address:
            return 0.5  # Medium risk for missing data
        
        # Device reuse patterns
        same_device_count = len(df[df['fingerprint'] == fingerprint]) if len(df) > 0 else 0
        same_ip_count = len(df[df['ip_address'] == ip_address]) if len(df) > 0 else 0
        
        # Device-IP pairing analysis
        device_ip_pairs = df.groupby(['fingerprint', 'ip_address']).size()
        current_pair_count = device_ip_pairs.get((fingerprint, ip_address), 0)
        
        # Risk scoring based on patterns
        risk_score = 0.0
        
        # High risk: New device with known IP (possible device spoofing)
        if same_device_count == 0 and same_ip_count > 3:
            risk_score += 0.4
        
        # High risk: Known device with new IP (possible account takeover)
        if same_device_count > 5 and same_ip_count == 0:
            risk_score += 0.5
        
        # Medium risk: Multiple devices from same IP (shared computer)
        if same_ip_count > 5 and len(df['fingerprint'].unique()) > 3:
            risk_score += 0.3
        
        # Low risk: Consistent device-IP pairing
        if current_pair_count > 2 and same_device_count > 1:
            risk_score -= 0.2
        
        # Normalize to 0-1 range
        return max(0.0, min(1.0, risk_score))
        
    except Exception as e:
        print(f"Device risk calculation error: {e}")
        return 0.5

def _calculate_behavioral_risk(req: TransactionRequest, df: pd.DataFrame, now: datetime) -> float:
    """Calculate behavioral pattern risk based on user behavior"""
    try:
        if len(df) == 0:
            return 0.3  # New user risk
        
        risk_score = 0.0
        
        # Time pattern analysis
        hour = now.hour
        day_of_week = now.weekday()
        
        # Check if user typically transacts at this time
        user_hours = df['created_at'].dt.hour if 'created_at' in df.columns else pd.Series()
        user_days = df['created_at'].dt.dayofweek if 'created_at' in df.columns else pd.Series()
        
        if len(user_hours) > 0:
            # Unusual time pattern
            if hour not in user_hours.value_counts().head(3).index:
                risk_score += 0.2
            
            # Weekend vs weekday pattern
            is_weekend = day_of_week >= 5
            weekend_txns = len(user_days[user_days >= 5])
            weekday_txns = len(user_days[user_days < 5])
            
            if is_weekend and weekend_txns == 0 and weekday_txns > 0:
                risk_score += 0.3  # First weekend transaction
        
        # Amount pattern analysis
        amounts = df['amount'] if 'amount' in df.columns else pd.Series()
        if len(amounts) > 0:
            current_amount = float(req.amount)
            avg_amount = amounts.mean()
            std_amount = amounts.std()
            
            # Unusual amount (outside 2 standard deviations)
            if std_amount > 0 and abs(current_amount - avg_amount) > 2 * std_amount:
                risk_score += 0.2
            
            # Testing pattern (small amounts followed by large)
            recent_amounts = amounts.tail(3)
            if len(recent_amounts) >= 2:
                if all(amt < avg_amount * 0.5 for amt in recent_amounts[:-1]) and current_amount > avg_amount * 2:
                    risk_score += 0.4  # Testing pattern detected
        
        # Velocity pattern analysis
        recent_1h = df[df['created_at'] >= now - timedelta(hours=1)] if 'created_at' in df.columns else pd.DataFrame()
        recent_24h = df[df['created_at'] >= now - timedelta(hours=24)] if 'created_at' in df.columns else pd.DataFrame()
        
        if len(recent_1h) > 5:  # Unusually high velocity
            risk_score += 0.3
        elif len(recent_24h) > 20:  # Very high daily velocity
            risk_score += 0.2
        
        # Payment method consistency
        if 'card_brand' in df.columns and len(df) > 0:
            user_brands = df['card_brand'].value_counts()
            current_brand = req.card_brand
            
            if current_brand not in user_brands.index and len(user_brands) == 1:
                risk_score += 0.1  # New card brand for single-brand user
        
        return max(0.0, min(1.0, risk_score))
        
    except Exception as e:
        print(f"Behavioral risk calculation error: {e}")
        return 0.3

def _calculate_merchant_risk(req: TransactionRequest, df: pd.DataFrame) -> float:
    """Calculate merchant-specific risk patterns"""
    try:
        if len(df) == 0:
            return 0.2  # New merchant risk
        
        risk_score = 0.0
        
        # Industry-specific patterns (based on amount ranges)
        amount = float(req.amount)
        
        # Digital goods vs physical goods patterns
        if amount % 0.99 == 0 or amount in [9.99, 19.99, 29.99, 49.99, 99.99]:
            risk_score += 0.1  # Common digital pricing
        
        # Round number risk (often used in testing)
        if amount % 100 == 0 and amount > 100:
            risk_score += 0.2
        
        # Unusual decimal places
        decimal_places = len(str(amount).split('.')[-1]) if '.' in str(amount) else 0
        if decimal_places > 2:
            risk_score += 0.1  # Unusual precision
        
        # Currency patterns
        if req.currency.upper() not in ['USD', 'EUR', 'GBP', 'CAD']:
            risk_score += 0.1  # Less common currency
        
        # Card brand patterns
        if req.card_brand == 'AMEX' and amount < 10:
            risk_score += 0.2  # Unusual AMEX usage pattern
        
        return max(0.0, min(1.0, risk_score))
        
    except Exception as e:
        print(f"Merchant risk calculation error: {e}")
        return 0.2

def _calculate_ip_reputation_risk(ip_address: str) -> float:
    """Calculate IP reputation risk based on real-world patterns"""
    try:
        if not ip_address:
            return 0.5
        
        # High-risk IP patterns (simplified)
        high_risk_ips = [
            "185.220.101",  # Known VPN/proxy ranges
            "192.168.1.1",  # Local network (suspicious for production)
            "10.0.0.1",     # Local network
            "172.16.0.1"    # Local network
        ]
        
        for risk_ip in high_risk_ips:
            if ip_address.startswith(risk_ip):
                return 0.8
        
        # Check for suspicious patterns
        if ip_address.count('.') != 3:  # Invalid IP format
            return 0.9
        
        return 0.2  # Default low risk
        
    except Exception:
        return 0.5

def _detect_amount_testing_pattern(req: TransactionRequest, df: pd.DataFrame) -> float:
    """Detect amount testing patterns (small amounts followed by large)"""
    try:
        if len(df) == 0:
            return 0.0
        
        current_amount = float(req.amount)
        
        # Check if recent transactions were small amounts
        recent_amounts = df['amount'].tail(3) if len(df) >= 3 else df['amount']
        
        if len(recent_amounts) >= 2:
            avg_recent = recent_amounts.mean()
            # If recent amounts were small but current is large
            if avg_recent < 50 and current_amount > avg_recent * 5:
                return 0.7  # High testing pattern risk
        
        # Common testing amounts
        testing_amounts = [1.00, 5.00, 10.00, 50.00, 100.00]
        if current_amount in testing_amounts:
            return 0.4
        
        return 0.0
        
    except Exception:
        return 0.0

def _detect_time_anomaly_pattern(req: TransactionRequest, now: datetime) -> float:
    """Detect unusual transaction times"""
    try:
        hour = now.hour
        day_of_week = now.weekday()
        
        # High-risk hours (2 AM - 6 AM)
        if 2 <= hour <= 6:
            return 0.8
        
        # Weekend late night (Friday/Saturday 11 PM - 3 AM)
        if day_of_week in [4, 5] and hour >= 23:
            return 0.6
        
        # Very early morning (before 7 AM)
        if hour < 7:
            return 0.4
        
        return 0.0
        
    except Exception:
        return 0.0

def _calculate_card_bin_risk(card_brand: str) -> float:
    """Calculate card BIN risk based on brand"""
    try:
        # Simplified risk scoring by card brand
        risk_scores = {
            "VISA": 0.2,
            "MASTERCARD": 0.2,
            "AMEX": 0.3,  # Higher chargeback rates
            "DISCOVER": 0.4,  # Less common
            "DINERS": 0.6,  # Less common, higher risk
            "JCB": 0.7,  # International, higher risk
            "UNIONPAY": 0.8  # International, higher risk
        }
        
        return risk_scores.get(card_brand.upper(), 0.5)
        
    except Exception:
        return 0.5

def _generate_fraud_reasons(features: list, confidence: float) -> list:
    """Generate comprehensive fraud reasons based on real-world patterns"""
    reasons = []
    
    # Amount-based reasons (enhanced)
    if features[0] > 8:  # High amount (log scale)
        reasons.append("Transaction amount significantly exceeds customer average")
    elif features[17] == 1:  # Amount above 2x average
        reasons.append("Transaction amount is 2x higher than customer average")
    elif features[18] == 1:  # Amount above 5x average
        reasons.append("Transaction amount is 5x higher than customer average")
    
    # Time-based reasons (enhanced)
    if features[6] == 0 and features[7] == 1:  # Weekend and evening/night
        reasons.append("Unusual transaction time (weekend evening/night)")
    elif features[8] == 1:  # Night time (2-6 AM)
        reasons.append("Transaction during high-risk hours (2-6 AM)")
    elif features[11] == 1:  # Evening
        reasons.append("Transaction during evening hours")
    
    # Velocity-based reasons (enhanced)
    if features[20] > 5:  # Very high transaction count in 1 hour
        reasons.append("Extremely high transaction frequency (5+ per hour)")
    elif features[20] > 3:  # High transaction count in 1 hour
        reasons.append("High transaction frequency detected (3+ per hour)")
    elif features[21] > 15:  # High daily velocity
        reasons.append("Unusual daily transaction volume (15+ transactions)")
    
    # Behavioral pattern reasons
    if features[36] > 0.7:  # High behavioral risk
        reasons.append("Transaction pattern inconsistent with customer behavior")
    elif features[36] > 0.5:  # Medium behavioral risk
        reasons.append("Some behavioral anomalies detected")
    
    # Device fingerprint reasons
    if features[35] > 0.7:  # High device risk
        reasons.append("Device fingerprint shows signs of compromise")
    elif features[35] > 0.5:  # Medium device risk
        reasons.append("Device patterns indicate potential fraud")
    
    # Merchant-specific reasons
    if features[37] > 0.6:  # High merchant risk
        reasons.append("Transaction characteristics match known fraud patterns")
    
    # Geographic diversity reasons
    if features[38] > 3:  # Multiple countries
        reasons.append("Customer using cards from multiple countries")
    
    # Risk score reasons
    if features[32] > 0.8:  # Very high risk score
        reasons.append("Payment processor flagged transaction as very high risk")
    elif features[32] > 0.6:  # High risk score
        reasons.append("Payment processor risk score is elevated")
    
    # Email domain reasons
    if features[33] > 0.7:  # High email domain risk
        reasons.append("Email domain strongly associated with fraudulent activity")
    elif features[33] > 0.4:  # Medium email domain risk
        reasons.append("Email domain has some fraud associations")
    
    # Country mismatch
    if features[34] == 1:  # Country mismatch
        reasons.append("Card country differs from billing country")
    
    # IP reuse patterns
    if features[35] > 0.6:  # High device risk (includes IP reuse)
        reasons.append("IP address patterns suggest potential account takeover")
    
    # Time since last transaction
    if features[26] < 0.1:  # Very short time since last transaction
        reasons.append("Transaction occurs immediately after previous transaction")
    elif features[26] < 1:  # Short time since last transaction
        reasons.append("Very short time between transactions")
    
    # Default reasons based on confidence
    if not reasons:
        if confidence > 0.8:
            reasons.append("Multiple high-risk indicators detected")
        elif confidence > 0.6:
            reasons.append("Several risk factors present")
        elif confidence > 0.4:
            reasons.append("Some risk indicators detected")
        elif confidence > 0.2:
            reasons.append("Low-level risk factors present")
        else:
            reasons.append("Transaction appears legitimate")
    
    # Limit to most significant reasons
    return reasons[:5]
