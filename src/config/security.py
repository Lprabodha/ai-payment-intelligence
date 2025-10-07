"""
Security configuration for TransactIQ
Defines security policies, thresholds, and validation rules
"""

import os
from typing import Set, Dict, List


class SecurityConfig:
    """Security configuration settings"""
    
    # Risk Score Thresholds
    HIGH_RISK_THRESHOLD = 0.7
    MEDIUM_RISK_THRESHOLD = 0.4
    LOW_RISK_THRESHOLD = 0.1
    
    # Email Security
    DISPOSABLE_EMAIL_DOMAINS = {
        '10minutemail.com', 'tempmail.org', 'guerrillamail.com',
        'mailinator.com', 'throwaway.email', 'temp-mail.org',
        'yopmail.com', 'maildrop.cc', 'sharklasers.com',
        'guerrillamailblock.com', 'pokemail.net', 'spam4.me',
        'mailnesia.com', 'mailcatch.com', 'mailme.lv',
        'inboxalias.com', 'mailin8r.com', 'mailinator2.com'
    }
    
    TRUSTED_EMAIL_DOMAINS = {
        'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
        'icloud.com', 'aol.com', 'protonmail.com', 'zoho.com',
        'mail.com', 'gmx.com', 'yandex.com', 'mail.ru'
    }
    
    # Amount Validation
    MAX_TRANSACTION_AMOUNT = 1000000  # $1M
    MIN_TRANSACTION_AMOUNT = 0.01     # $0.01
    SUSPICIOUS_AMOUNTS = {
        0.01, 0.99, 1.00, 1.01, 10.00, 100.00, 1000.00,
        99.99, 999.99, 9999.99, 10000.00
    }
    
    # IP Address Security
    PRIVATE_IP_RANGES = [
        '10.0.0.0/8',
        '172.16.0.0/12', 
        '192.168.0.0/16',
        '127.0.0.0/8'
    ]
    
    HIGH_RISK_COUNTRIES = {
        'CN', 'RU', 'KP', 'IR', 'SY', 'VE', 'CU', 'MM'
    }
    
    # Card Validation
    SUPPORTED_CARD_BRANDS = {
        'Visa', 'Mastercard', 'American Express', 'Discover'
    }
    
    SUPPORTED_CURRENCIES = {
        'USD', 'EUR', 'GBP', 'CAD', 'AUD', 'JPY', 'CHF', 'SEK', 'NOK', 'DKK'
    }
    
    # Security Patterns
    SQL_INJECTION_PATTERNS = [
        r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)',
        r'(\b(OR|AND)\s+\d+\s*=\s*\d+)',
        r'(\b(OR|AND)\s+[\'"][^\'"]*[\'"]\s*=\s*[\'"][^\'"]*[\'"])',
        r'(;\s*(DROP|DELETE|INSERT|UPDATE))',
        r'(\b(UNION|SELECT)\s+.*\s+FROM)',
        r'(\b(OR|AND)\s+.*\s+LIKE)',
        r'(\b(OR|AND)\s+.*\s+IN\s*\()',
    ]
    
    XSS_PATTERNS = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'on\w+\s*=',
        r'<iframe[^>]*>',
        r'<object[^>]*>',
        r'<embed[^>]*>',
        r'<link[^>]*>',
        r'<meta[^>]*>',
        r'<style[^>]*>.*?</style>',
        r'<img[^>]*onerror',
        r'<svg[^>]*onload',
        r'<body[^>]*onload',
        r'<form[^>]*onload',
    ]
    
    COMMAND_INJECTION_PATTERNS = [
        r'[;&|`$]',
        r'(\b(rm|del|format|shutdown|reboot|kill|ps|net|whoami)\b)',
        r'(\b(cat|ls|dir|type|more|less|head|tail)\b)',
        r'(\b(wget|curl|nc|telnet|ftp|ssh)\b)',
        r'(\b(eval|exec|system|shell_exec|passthru)\b)',
    ]
    
    # Rate Limiting
    RATE_LIMITS = {
        'fraud_prediction': 100,      # per minute
        'chargeback_prediction': 100,  # per minute
        'risk_score': 200,            # per minute
        'webhook': 1000,              # per minute
        'health_check': 300,          # per minute
    }
    
    # Request Size Limits
    MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_FIELD_LENGTH = 255
    
    # Security Headers
    SECURITY_HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'Referrer-Policy': 'strict-origin-when-cross-origin',
        'Content-Security-Policy': "default-src 'self'",
    }
    
    # Validation Rules
    EMAIL_REGEX = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    COUNTRY_CODE_REGEX = r'^[A-Z]{2}$'
    CARD_LAST_FOUR_REGEX = r'^\d{4}$'
    CURRENCY_REGEX = r'^[A-Z]{3}$'
    
    # Risk Scoring Weights
    RISK_WEIGHTS = {
        'email_risk': 0.25,
        'ip_risk': 0.25,
        'amount_risk': 0.20,
        'security_risk': 0.30
    }
    
    @classmethod
    def get_risk_threshold(cls, risk_type: str) -> float:
        """Get risk threshold for specific type"""
        thresholds = {
            'high': cls.HIGH_RISK_THRESHOLD,
            'medium': cls.MEDIUM_RISK_THRESHOLD,
            'low': cls.LOW_RISK_THRESHOLD
        }
        return thresholds.get(risk_type, cls.MEDIUM_RISK_THRESHOLD)
    
    @classmethod
    def is_high_risk_country(cls, country_code: str) -> bool:
        """Check if country is considered high risk"""
        return country_code.upper() in cls.HIGH_RISK_COUNTRIES
    
    @classmethod
    def is_disposable_email_domain(cls, domain: str) -> bool:
        """Check if email domain is disposable"""
        return domain.lower() in cls.DISPOSABLE_EMAIL_DOMAINS
    
    @classmethod
    def is_trusted_email_domain(cls, domain: str) -> bool:
        """Check if email domain is trusted"""
        return domain.lower() in cls.TRUSTED_EMAIL_DOMAINS
    
    @classmethod
    def is_suspicious_amount(cls, amount: float) -> bool:
        """Check if amount is suspicious"""
        return amount in cls.SUSPICIOUS_AMOUNTS
    
    @classmethod
    def get_rate_limit(cls, endpoint: str) -> int:
        """Get rate limit for endpoint"""
        return cls.RATE_LIMITS.get(endpoint, 60)  # Default 60 per minute
    
    @classmethod
    def is_supported_currency(cls, currency: str) -> bool:
        """Check if currency is supported"""
        return currency.upper() in cls.SUPPORTED_CURRENCIES
    
    @classmethod
    def is_supported_card_brand(cls, brand: str) -> bool:
        """Check if card brand is supported"""
        return brand.title() in cls.SUPPORTED_CARD_BRANDS


# Global security configuration instance
security_config = SecurityConfig()
