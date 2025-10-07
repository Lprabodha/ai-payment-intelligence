"""
Enhanced data validation and security utilities for TransactIQ
Implements comprehensive input validation, sanitization, and security checks
"""

import re
import ipaddress
from typing import Any, Dict, List, Optional, Union
from decimal import Decimal, InvalidOperation
from datetime import datetime, date
from pydantic import BaseModel, Field, field_validator, model_validator
import bleach
import html


class ValidationError(Exception):
    """Custom validation error for TransactIQ"""
    pass


class EmailValidator:
    """Advanced email validation with security checks"""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format and check for suspicious patterns"""
        try:
            # Basic format validation
            if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
                return False
            
            # Check for suspicious patterns
            suspicious_patterns = [
                r'\+.*\+',  # Multiple plus signs
                r'\.{2,}',  # Multiple consecutive dots
                r'@.*@',    # Multiple @ symbols
                r'^\.',     # Starts with dot
                r'\.$',     # Ends with dot
                r'\.\.',    # Consecutive dots
                r'[<>"\']', # HTML/SQL injection attempts
            ]
            
            for pattern in suspicious_patterns:
                if re.search(pattern, email):
                    return False
            
            return True
        except Exception:
            return False
    
    @staticmethod
    def get_email_domain(email: str) -> Optional[str]:
        """Extract domain from email address"""
        try:
            return email.split('@')[1].lower()
        except Exception:
            return None
    
    @staticmethod
    def is_disposable_email(email: str) -> bool:
        """Check if email is from disposable email service"""
        disposable_domains = {
            '10minutemail.com', 'tempmail.org', 'guerrillamail.com',
            'mailinator.com', 'throwaway.email', 'temp-mail.org',
            'yopmail.com', 'maildrop.cc', 'sharklasers.com',
            'guerrillamailblock.com', 'pokemail.net', 'spam4.me'
        }
        
        domain = EmailValidator.get_email_domain(email)
        return domain in disposable_domains
    
    @staticmethod
    def get_email_risk_score(email: str) -> float:
        """Calculate email risk score based on various factors"""
        risk_score = 0.0
        
        # Check for disposable email
        if EmailValidator.is_disposable_email(email):
            risk_score += 0.8
        
        # Check for suspicious patterns
        if re.search(r'\d{4,}', email):  # Multiple consecutive digits
            risk_score += 0.3
        
        if re.search(r'[a-z]{6,}', email):  # Long consecutive letters
            risk_score += 0.2
        
        # Check domain age and reputation (simplified)
        domain = EmailValidator.get_email_domain(email)
        if domain:
            # Common legitimate domains have lower risk
            legitimate_domains = {
                'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
                'icloud.com', 'aol.com', 'protonmail.com', 'zoho.com'
            }
            if domain not in legitimate_domains:
                risk_score += 0.2
        
        return min(risk_score, 1.0)


class IPValidator:
    """IP address validation and security analysis"""
    
    @staticmethod
    def validate_ip(ip: str) -> bool:
        """Validate IP address format"""
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def is_private_ip(ip: str) -> bool:
        """Check if IP address is private"""
        try:
            return ipaddress.ip_address(ip).is_private
        except ValueError:
            return False
    
    @staticmethod
    def is_reserved_ip(ip: str) -> bool:
        """Check if IP address is reserved"""
        try:
            return ipaddress.ip_address(ip).is_reserved
        except ValueError:
            return False
    
    @staticmethod
    def get_ip_risk_score(ip: str) -> float:
        """Calculate IP risk score based on various factors"""
        risk_score = 0.0
        
        if not IPValidator.validate_ip(ip):
            return 1.0  # Invalid IP is high risk
        
        # Private IPs have lower risk
        if IPValidator.is_private_ip(ip):
            risk_score += 0.1
        
        # Reserved IPs have higher risk
        if IPValidator.is_reserved_ip(ip):
            risk_score += 0.7
        
        # Check for suspicious IP patterns
        if re.match(r'^0\.', ip):  # IPs starting with 0.
            risk_score += 0.5
        
        if re.match(r'^127\.', ip):  # Localhost variations
            risk_score += 0.3
        
        return min(risk_score, 1.0)


class AmountValidator:
    """Financial amount validation and fraud detection"""
    
    @staticmethod
    def validate_amount(amount: Union[str, int, float, Decimal]) -> bool:
        """Validate transaction amount"""
        try:
            # Convert to Decimal for precise validation
            if isinstance(amount, str):
                amount = Decimal(amount)
            elif isinstance(amount, (int, float)):
                amount = Decimal(str(amount))
            
            # Check range
            if amount <= 0:
                return False
            
            # Check for suspicious amounts
            if amount > Decimal('1000000'):  # $1M limit
                return False
            
            # Check decimal places (max 2 for currency)
            if amount.as_tuple().exponent < -2:
                return False
            
            return True
        except (InvalidOperation, ValueError, TypeError):
            return False
    
    @staticmethod
    def is_round_amount(amount: Union[str, int, float, Decimal]) -> bool:
        """Check if amount is a round number (potential testing)"""
        try:
            if isinstance(amount, str):
                amount = Decimal(amount)
            elif isinstance(amount, (int, float)):
                amount = Decimal(str(amount))
            
            # Check if amount is divisible by 100
            return amount % Decimal('100') == 0
        except Exception:
            return False
    
    @staticmethod
    def is_testing_amount(amount: Union[str, int, float, Decimal]) -> bool:
        """Check if amount matches common testing patterns"""
        try:
            if isinstance(amount, str):
                amount = Decimal(amount)
            elif isinstance(amount, (int, float)):
                amount = Decimal(str(amount))
            
            # Common testing amounts
            testing_amounts = {
                Decimal('0.01'), Decimal('0.99'), Decimal('1.00'), Decimal('1.01'),
                Decimal('10.00'), Decimal('100.00'), Decimal('1000.00'),
                Decimal('99.99'), Decimal('999.99'), Decimal('9999.99')
            }
            
            return amount in testing_amounts
        except Exception:
            return False
    
    @staticmethod
    def get_amount_risk_score(amount: Union[str, int, float, Decimal]) -> float:
        """Calculate amount risk score"""
        risk_score = 0.0
        
        if not AmountValidator.validate_amount(amount):
            return 1.0
        
        try:
            if isinstance(amount, str):
                amount = Decimal(amount)
            elif isinstance(amount, (int, float)):
                amount = Decimal(str(amount))
            
            # Round amounts are suspicious
            if AmountValidator.is_round_amount(amount):
                risk_score += 0.3
            
            # Testing amounts are suspicious
            if AmountValidator.is_testing_amount(amount):
                risk_score += 0.5
            
            # Very high amounts are risky
            if amount > Decimal('10000'):
                risk_score += 0.4
            
            # Very low amounts might be testing
            if amount < Decimal('1'):
                risk_score += 0.2
            
        except Exception:
            risk_score = 1.0
        
        return min(risk_score, 1.0)


class CardValidator:
    """Credit card validation and fraud detection"""
    
    @staticmethod
    def validate_card_number(card_number: str) -> bool:
        """Validate credit card number using Luhn algorithm"""
        try:
            # Remove spaces and dashes
            card_number = re.sub(r'[\s-]', '', card_number)
            
            # Check if all digits
            if not card_number.isdigit():
                return False
            
            # Check length (13-19 digits)
            if len(card_number) < 13 or len(card_number) > 19:
                return False
            
            # Luhn algorithm
            def luhn_checksum(card_num):
                def digits_of(n):
                    return [int(d) for d in str(n)]
                
                digits = digits_of(card_num)
                odd_digits = digits[-1::-2]
                even_digits = digits[-2::-2]
                checksum = sum(odd_digits)
                for d in even_digits:
                    checksum += sum(digits_of(d * 2))
                return checksum % 10
            
            return luhn_checksum(card_number) == 0
        except Exception:
            return False
    
    @staticmethod
    def get_card_brand(card_number: str) -> str:
        """Identify card brand from number"""
        try:
            card_number = re.sub(r'[\s-]', '', card_number)
            
            if card_number.startswith('4'):
                return 'Visa'
            elif card_number.startswith('5') or card_number.startswith('2'):
                return 'Mastercard'
            elif card_number.startswith('3'):
                return 'American Express'
            elif card_number.startswith('6'):
                return 'Discover'
            else:
                return 'Unknown'
        except Exception:
            return 'Unknown'
    
    @staticmethod
    def is_test_card(card_number: str) -> bool:
        """Check if card number is a test card"""
        test_patterns = [
            r'^4\d{12}(\d{3})?$',  # Visa test
            r'^5[1-5]\d{14}$',     # Mastercard test
            r'^3[47]\d{13}$',      # Amex test
            r'^6\d{15}$',          # Discover test
        ]
        
        card_number = re.sub(r'[\s-]', '', card_number)
        
        for pattern in test_patterns:
            if re.match(pattern, card_number):
                return True
        
        return False


class Sanitizer:
    """Data sanitization utilities for security"""
    
    @staticmethod
    def sanitize_string(value: str, max_length: int = 255) -> str:
        """Sanitize string input to prevent XSS and injection attacks"""
        if not isinstance(value, str):
            return ""
        
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # HTML encode to prevent XSS
        value = html.escape(value)
        
        # Remove HTML tags
        value = bleach.clean(value, tags=[], strip=True)
        
        # Remove SQL injection patterns
        sql_patterns = [
            r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)',
            r'(\b(OR|AND)\s+\d+\s*=\s*\d+)',
            r'(\b(OR|AND)\s+[\'"][^\'"]*[\'"]\s*=\s*[\'"][^\'"]*[\'"])',
            r'(;\s*(DROP|DELETE|INSERT|UPDATE))',
        ]
        
        for pattern in sql_patterns:
            value = re.sub(pattern, '', value, flags=re.IGNORECASE)
        
        # Truncate if too long
        if len(value) > max_length:
            value = value[:max_length]
        
        return value.strip()
    
    @staticmethod
    def sanitize_json(data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize JSON data recursively"""
        sanitized = {}
        
        for key, value in data.items():
            # Sanitize key
            clean_key = Sanitizer.sanitize_string(str(key), 100)
            
            if isinstance(value, str):
                sanitized[clean_key] = Sanitizer.sanitize_string(value)
            elif isinstance(value, dict):
                sanitized[clean_key] = Sanitizer.sanitize_json(value)
            elif isinstance(value, list):
                sanitized[clean_key] = [
                    Sanitizer.sanitize_string(str(item)) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                sanitized[clean_key] = value
        
        return sanitized


class SecurityValidator:
    """Security-focused validation and threat detection"""
    
    @staticmethod
    def detect_sql_injection(value: str) -> bool:
        """Detect potential SQL injection patterns"""
        sql_patterns = [
            r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)',
            r'(\b(OR|AND)\s+\d+\s*=\s*\d+)',
            r'(\b(OR|AND)\s+[\'"][^\'"]*[\'"]\s*=\s*[\'"][^\'"]*[\'"])',
            r'(\b(OR|AND)\s+[\'"]\s*=\s*[\'"])',
            r'(;\s*(DROP|DELETE|INSERT|UPDATE))',
            r'(\b(UNION|SELECT)\s+.*\s+FROM)',
            r'(\b(OR|AND)\s+.*\s+LIKE)',
            r'(\b(OR|AND)\s+.*\s+IN\s*\()',
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        
        return False
    
    @staticmethod
    def detect_xss(value: str) -> bool:
        """Detect potential XSS patterns"""
        xss_patterns = [
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
        
        for pattern in xss_patterns:
            if re.search(pattern, value, re.IGNORECASE | re.DOTALL):
                return True
        
        return False
    
    @staticmethod
    def detect_command_injection(value: str) -> bool:
        """Detect potential command injection patterns"""
        cmd_patterns = [
            r'[;&|`$]',
            r'(\b(rm|del|format|shutdown|reboot|kill|ps|net|whoami)\b)',
            r'(\b(cat|ls|dir|type|more|less|head|tail)\b)',
            r'(\b(wget|curl|nc|telnet|ftp|ssh)\b)',
            r'(\b(eval|exec|system|shell_exec|passthru)\b)',
        ]
        
        for pattern in cmd_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        
        return False
    
    @staticmethod
    def validate_input_security(data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive security validation of input data"""
        security_issues = []
        risk_score = 0.0
        
        for key, value in data.items():
            if isinstance(value, str):
                # Check for SQL injection
                if SecurityValidator.detect_sql_injection(value):
                    security_issues.append(f"SQL injection pattern detected in {key}")
                    risk_score += 0.3
                
                # Check for XSS
                if SecurityValidator.detect_xss(value):
                    security_issues.append(f"XSS pattern detected in {key}")
                    risk_score += 0.3
                
                # Check for command injection
                if SecurityValidator.detect_command_injection(value):
                    security_issues.append(f"Command injection pattern detected in {key}")
                    risk_score += 0.4
        
        return {
            "is_secure": len(security_issues) == 0,
            "risk_score": min(risk_score, 1.0),
            "issues": security_issues,
            "recommendation": "Block request" if risk_score > 0.5 else "Allow with monitoring"
        }


class EnhancedTransactionRequest(BaseModel):
    """Enhanced transaction request with comprehensive validation"""
    
    # Basic transaction data
    amount: Decimal = Field(..., gt=0, le=1000000, description="Transaction amount")
    currency: str = Field(default="USD", pattern=r'^[A-Z]{3}$', description="Currency code")
    
    # Customer information
    email: str = Field(..., description="Customer email address")
    card_country: str = Field(..., min_length=2, max_length=2, description="Card country code")
    billing_country: str = Field(..., min_length=2, max_length=2, description="Billing country code")
    
    # Technical data
    ip_address: str = Field(..., description="Customer IP address")
    fingerprint: str = Field(..., min_length=1, max_length=255, description="Device fingerprint")
    
    # Risk and timing
    risk_score: float = Field(default=0.0, ge=0, le=100, description="Risk score")
    hour: int = Field(..., ge=0, le=23, description="Hour of transaction")
    
    # Optional fields
    card_brand: Optional[str] = Field(None, max_length=50)
    card_last_four: Optional[str] = Field(None, pattern=r'^\d{4}$')
    user_agent: Optional[str] = Field(None, max_length=500)
    
    @validator('amount')
    def validate_amount(cls, v):
        if not AmountValidator.validate_amount(v):
            raise ValueError('Invalid transaction amount')
        return v
    
    @validator('email')
    def validate_email(cls, v):
        if not EmailValidator.validate_email(v):
            raise ValueError('Invalid email format')
        if EmailValidator.is_disposable_email(v):
            raise ValueError('Disposable email addresses not allowed')
        return v
    
    @validator('ip_address')
    def validate_ip(cls, v):
        if not IPValidator.validate_ip(v):
            raise ValueError('Invalid IP address')
        return v
    
    @validator('fingerprint')
    def sanitize_fingerprint(cls, v):
        return Sanitizer.sanitize_string(v, 255)
    
    @validator('card_country', 'billing_country')
    def validate_country_codes(cls, v):
        if not re.match(r'^[A-Z]{2}$', v):
            raise ValueError('Invalid country code format')
        return v.upper()
    
    @validator('card_brand')
    def validate_card_brand(cls, v):
        if v is not None:
            valid_brands = {'Visa', 'Mastercard', 'American Express', 'Discover'}
            if v not in valid_brands:
                raise ValueError('Invalid card brand')
        return v
    
    @root_validator
    def validate_country_consistency(cls, values):
        card_country = values.get('card_country')
        billing_country = values.get('billing_country')
        
        # Flag potential mismatch for review
        if card_country != billing_country:
            values['country_mismatch'] = True
        else:
            values['country_mismatch'] = False
        
        return values
    
    @root_validator
    def validate_security(cls, values):
        # Perform security validation
        security_check = SecurityValidator.validate_input_security(values)
        
        if not security_check['is_secure']:
            raise ValueError(f"Security validation failed: {security_check['issues']}")
        
        # Add risk scores
        values['email_risk_score'] = EmailValidator.get_email_risk_score(values.get('email', ''))
        values['ip_risk_score'] = IPValidator.get_ip_risk_score(values.get('ip_address', ''))
        values['amount_risk_score'] = AmountValidator.get_amount_risk_score(values.get('amount', 0))
        
        return values


# Global validator instances
email_validator = EmailValidator()
ip_validator = IPValidator()
amount_validator = AmountValidator()
card_validator = CardValidator()
sanitizer = Sanitizer()
security_validator = SecurityValidator()


def validate_transaction_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize transaction data"""
    try:
        # Sanitize input data
        sanitized_data = sanitizer.sanitize_json(data)
        
        # Perform security validation
        security_result = security_validator.validate_input_security(sanitized_data)
        
        # Create validated request
        validated_request = EnhancedTransactionRequest(**sanitized_data)
        
        return {
            "is_valid": True,
            "data": validated_request.dict(),
            "security": security_result
        }
        
    except Exception as e:
        return {
            "is_valid": False,
            "error": str(e),
            "security": {"is_secure": False, "risk_score": 1.0, "issues": [str(e)]}
        }
