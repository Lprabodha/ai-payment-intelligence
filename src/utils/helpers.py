"""
Utility functions and helpers
"""
import numpy as np
import pandas as pd
from datetime import datetime
from config.settings import settings

def sanitize_for_mongo(obj):
    """Sanitize data for MongoDB storage"""
    if isinstance(obj, dict):
        return {k: sanitize_for_mongo(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_mongo(i) for i in obj]
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, datetime):
        return obj
    elif isinstance(obj, str):
        return obj
    else:
        return str(obj)

def to_native(value):
    """Convert numpy types to native Python types"""
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    elif isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    elif isinstance(value, np.bool_):
        return bool(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    else:
        return value

def classify_risk_level(confidence, high=0.85, medium=0.5):
    """Classify risk level based on confidence score"""
    if confidence >= high:
        return "high"
    elif confidence >= medium:
        return "medium"
    else:
        return "low"

def _email_domain_risk(email: str) -> int:
    """Check if email domain is common or risky"""
    if not email or '@' not in email:
        return 1
    
    domain = email.split('@')[1].lower()
    
    # Common domains are less risky
    if domain in settings.COMMON_DOMAINS:
        return 0
    
    # Suspicious domains
    suspicious_domains = [
        'tempmail', 'temp-mail', '10minutemail', 'guerrillamail',
        'mailinator', 'yopmail', 'throwaway', 'trashmail'
    ]
    
    if any(sus in domain for sus in suspicious_domains):
        return 1
    
    return 0

def build_recommendations(transaction: dict,
                         fraud_result: dict = None,
                         chargeback_result: dict = None,
                         routing_result: dict = None) -> list:
    """Build recommendations based on prediction results"""
    recommendations = []
    
    # Fraud-based recommendations
    if fraud_result and fraud_result.get('is_fraud', False):
        confidence = fraud_result.get('confidence_score', 0)
        if confidence > 0.8:
            recommendations.append({
                "action": "block_transaction",
                "priority": "high",
                "reason": "High fraud confidence",
                "confidence": confidence
            })
        elif confidence > 0.6:
            recommendations.append({
                "action": "require_manual_review",
                "priority": "medium",
                "reason": "Moderate fraud risk",
                "confidence": confidence
            })
    
    # Chargeback-based recommendations
    if chargeback_result and chargeback_result.get('chargeback_predicted', False):
        confidence = chargeback_result.get('confidence_score', 0)
        if confidence > 0.7:
            recommendations.append({
                "action": "require_additional_verification",
                "priority": "high",
                "reason": "High chargeback risk",
                "confidence": confidence
            })
    
    # Routing-based recommendations
    if routing_result and routing_result.get('recommended_gateway'):
        current_gateway = transaction.get('gateway', 'unknown')
        recommended_gateway = routing_result.get('recommended_gateway')
        
        if current_gateway != recommended_gateway:
            recommendations.append({
                "action": "consider_gateway_switch",
                "priority": "medium",
                "reason": f"Better gateway available: {recommended_gateway}",
                "confidence": routing_result.get('confidence', 0)
            })
    
    # Amount-based recommendations
    amount = transaction.get('amount', 0)
    if amount > 10000:
        recommendations.append({
            "action": "require_manager_approval",
            "priority": "medium",
            "reason": "High-value transaction",
            "confidence": 1.0
        })
    
    # Default recommendation if no specific issues
    if not recommendations:
        recommendations.append({
            "action": "approve_transaction",
            "priority": "low",
            "reason": "No significant risk indicators",
            "confidence": 1.0
        })
    
    return recommendations
