"""
Security utilities for transaction validation and fraud detection
"""
import re
from typing import Dict, List, Tuple, Optional, Any
from utils.validation import (
    EmailValidator, IPValidator, CardValidator, AmountValidator, SecurityValidator
)
# Security constants
DISPOSABLE_EMAIL_DOMAINS = {
    "mailinator.com", "tempmail.com", "guerrillamail.com", "10minutemail.com",
    "yopmail.com", "trashmail.com", "fakemail.com", "temp-mail.org"
}

TEST_CARD_PREFIXES = {
    "411111", "400000", "545454", "555555", "378282", "340000", "601100"
}

HIGH_RISK_THRESHOLD = 0.7

def validate_and_sanitize_request(data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Validate and sanitize transaction request data
    
    Args:
        data: Transaction request data dictionary
        
    Returns:
        Tuple of (is_valid, sanitized_data, error_message)
    """
    try:
        sanitized_data = {}
        
        # Validate email
        if 'email' in data:
            email = data['email'].strip().lower()
            if not EmailValidator.validate_email(email):
                return False, {}, "Invalid email format"
            if EmailValidator.is_disposable_email(email):
                return False, {}, "Disposable email addresses not allowed"
            sanitized_data['email'] = email
        
        # Validate IP address
        if 'ip_address' in data:
            ip = data['ip_address'].strip()
            if not IPValidator.validate_ip(ip):
                return False, {}, "Invalid IP address format"
            if IPValidator.is_private_ip(ip):
                # Private IPs are allowed but flagged
                sanitized_data['ip_address'] = ip
                sanitized_data['is_private_ip'] = True
            else:
                sanitized_data['ip_address'] = ip
                sanitized_data['is_private_ip'] = False
        
        # Validate amount
        if 'amount' in data:
            amount = float(data['amount'])
            if not AmountValidator.validate_amount(amount):
                return False, {}, "Invalid transaction amount"
            if AmountValidator.is_testing_amount(amount):
                sanitized_data['amount'] = amount
                sanitized_data['is_testing_amount'] = True
            else:
                sanitized_data['amount'] = amount
                sanitized_data['is_testing_amount'] = False
        
        # Validate card brand
        if 'card_brand' in data:
            card_brand = data['card_brand'].strip().upper()
            sanitized_data['card_brand'] = card_brand
        
        # Validate countries
        if 'card_country' in data:
            country = data['card_country'].strip().upper()
            if len(country) != 2 or not country.isalpha():
                return False, {}, "Invalid country code format"
            sanitized_data['card_country'] = country
            
        if 'billing_country' in data:
            country = data['billing_country'].strip().upper()
            if len(country) != 2 or not country.isalpha():
                return False, {}, "Invalid billing country code format"
            sanitized_data['billing_country'] = country
        
        # Check for security patterns in text fields
        text_fields = ['fingerprint', 'outcome_type', 'seller_message', 'network_status']
        for field in text_fields:
            if field in data and data[field]:
                text = str(data[field])
                if SecurityValidator.detect_sql_injection(text):
                    return False, {}, f"Potential SQL injection detected in {field}"
                if SecurityValidator.detect_xss(text):
                    return False, {}, f"Potential XSS detected in {field}"
                if SecurityValidator.detect_command_injection(text):
                    return False, {}, f"Potential command injection detected in {field}"
                sanitized_data[field] = SecurityValidator.sanitize_input(text)
        
        # Copy other fields
        for key, value in data.items():
            if key not in sanitized_data:
                sanitized_data[key] = value
        
        return True, sanitized_data, None
        
    except Exception as e:
        return False, {}, f"Validation error: {str(e)}"

def calculate_security_score(data: Dict[str, Any]) -> float:
    """
    Calculate security risk score for transaction data
    
    Args:
        data: Sanitized transaction data
        
    Returns:
        Security score between 0.0 and 1.0
    """
    score = 0.0
    
    # Email risk
    if 'email' in data:
        score += EmailValidator.get_email_risk_score(data['email']) * 0.2
    
    # IP risk
    if 'ip_address' in data:
        score += IPValidator.get_ip_risk_score(data['ip_address']) * 0.3
    
    # Amount risk
    if 'amount' in data:
        score += AmountValidator.get_amount_risk_score(data['amount']) * 0.2
    
    # Private IP risk
    if data.get('is_private_ip', False):
        score += 0.1
    
    # Testing amount risk
    if data.get('is_testing_amount', False):
        score += 0.2
    
    # Country mismatch risk
    if 'card_country' in data and 'billing_country' in data:
        if data['card_country'] != data['billing_country']:
            score += 0.1
    
    return min(1.0, score)

def is_high_risk_transaction(data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Check if transaction is high risk based on security patterns
    
    Args:
        data: Transaction data
        
    Returns:
        Tuple of (is_high_risk, reason)
    """
    # Check for multiple high-risk indicators
    risk_indicators = []
    
    # High security score
    security_score = calculate_security_score(data)
    if security_score > HIGH_RISK_THRESHOLD:
        risk_indicators.append(f"High security score: {security_score:.2f}")
    
    # Disposable email
    if 'email' in data and EmailValidator.is_disposable_email(data['email']):
        risk_indicators.append("Disposable email detected")
    
    # Private IP
    if data.get('is_private_ip', False):
        risk_indicators.append("Private IP address")
    
    # Testing amount
    if data.get('is_testing_amount', False):
        risk_indicators.append("Testing amount detected")
    
    # Country mismatch
    if ('card_country' in data and 'billing_country' in data and 
        data['card_country'] != data['billing_country']):
        risk_indicators.append("Country mismatch")
    
    # High amount
    if 'amount' in data and data['amount'] > 10000:
        risk_indicators.append("High transaction amount")
    
    if len(risk_indicators) >= 2:
        return True, "; ".join(risk_indicators)
    
    return False, ""

def enhance_fraud_prediction_with_security(
    prediction_result: Dict[str, Any], 
    security_score: float
) -> Dict[str, Any]:
    """
    Enhance fraud prediction result with security insights
    
    Args:
        prediction_result: Original prediction result
        security_score: Calculated security score
        
    Returns:
        Enhanced prediction result
    """
    enhanced_result = prediction_result.copy()
    
    # Adjust confidence based on security score
    original_confidence = enhanced_result.get('confidence', 0.0)
    adjusted_confidence = min(1.0, original_confidence + (security_score * 0.3))
    enhanced_result['confidence'] = adjusted_confidence
    
    # Add security insights
    enhanced_result['security_score'] = security_score
    enhanced_result['security_enhanced'] = True
    
    # Update fraud decision if security score is very high
    if security_score > 0.8 and not enhanced_result.get('is_fraud', False):
        enhanced_result['is_fraud'] = True
        enhanced_result['fraud_reasons'].append("High security risk detected")
    
    return enhanced_result

def detect_velocity_anomalies(transactions: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    Detect velocity anomalies in transaction history
    
    Args:
        transactions: List of transaction dictionaries
        
    Returns:
        Tuple of (has_anomaly, description)
    """
    if len(transactions) < 3:
        return False, "Insufficient transaction history"
    
    # Check for rapid successive transactions
    amounts = [tx.get('amount', 0) for tx in transactions[-10:]]
    if len(amounts) >= 3:
        # Check for similar amounts (possible testing)
        if len(set(amounts)) <= 2:
            return True, "Repeated transaction amounts detected"
        
        # Check for rapid escalation
        if len(amounts) >= 5:
            recent_avg = sum(amounts[-5:]) / 5
            older_avg = sum(amounts[-10:-5]) / 5 if len(amounts) >= 10 else recent_avg
            if recent_avg > older_avg * 3:
                return True, "Rapid transaction amount escalation"
    
    return False, ""

def get_risk_factors(data: Dict[str, Any]) -> List[str]:
    """
    Get list of risk factors for transaction
    
    Args:
        data: Transaction data
        
    Returns:
        List of risk factor descriptions
    """
    risk_factors = []
    
    # Email risk
    if 'email' in data:
        email_risk = EmailValidator.get_email_risk_score(data['email'])
        if email_risk > 0.5:
            risk_factors.append(f"High-risk email domain (score: {email_risk:.2f})")
    
    # IP risk
    if 'ip_address' in data:
        ip_risk = IPValidator.get_ip_risk_score(data['ip_address'])
        if ip_risk > 0.5:
            risk_factors.append(f"High-risk IP address (score: {ip_risk:.2f})")
    
    # Amount risk
    if 'amount' in data:
        amount_risk = AmountValidator.get_amount_risk_score(data['amount'])
        if amount_risk > 0.5:
            risk_factors.append(f"High-risk amount (score: {amount_risk:.2f})")
    
    # Private IP
    if data.get('is_private_ip', False):
        risk_factors.append("Private IP address detected")
    
    # Testing amount
    if data.get('is_testing_amount', False):
        risk_factors.append("Testing amount detected")
    
    # Country mismatch
    if ('card_country' in data and 'billing_country' in data and 
        data['card_country'] != data['billing_country']):
        risk_factors.append("Card and billing country mismatch")
    
    return risk_factors