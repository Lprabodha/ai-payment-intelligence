"""
Simplified data validation utilities for TransactIQ
"""
import re
import ipaddress
from typing import Any, Dict, List, Optional, Union
from decimal import Decimal, InvalidOperation

class EmailValidator:
    """Enhanced email validation with security checks"""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format using regex"""
        if not email or not isinstance(email, str):
            return False
        
        # Basic email regex pattern
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def is_disposable_email(email: str) -> bool:
        """Check if email is from a disposable email service"""
        disposable_domains = {
            "mailinator", "10minutemail", "guerrillamail", "tempmail", "yopmail", 
            "trashmail", "temp-mail", "throwaway", "tempmail.org", "10minutemail.com"
        }
        
        domain = email.split('@')[1].lower() if '@' in email else ''
        return any(disposable in domain for disposable in disposable_domains)
    
    @staticmethod
    def get_email_risk_score(email: str) -> float:
        """Calculate email risk score"""
        if not email:
            return 1.0
        
        score = 0.1  # Base score
        
        # Disposable email
        if EmailValidator.is_disposable_email(email):
            score += 0.7
        
        # Unknown domain
        common_domains = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com"}
        domain = email.split('@')[1].lower() if '@' in email else ''
        if domain not in common_domains:
            score += 0.3
        
        return min(1.0, score)

class IPValidator:
    """IP address validation and risk assessment"""
    
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
        """Check if IP is private/reserved"""
        try:
            ip_obj = ipaddress.ip_address(ip)
            return ip_obj.is_private or ip_obj.is_reserved
        except ValueError:
            return False
    
    @staticmethod
    def get_ip_risk_score(ip: str) -> float:
        """Calculate IP risk score"""
        if not IPValidator.validate_ip(ip):
            return 1.0
        
        score = 0.2  # Base score
        
        # Private IP
        if IPValidator.is_private_ip(ip):
            score += 0.1
        
        # Known high-risk IPs (simplified)
        high_risk_ips = {"185.239.240.0", "185.239.241.0"}
        if ip in high_risk_ips:
            score += 0.6
        
        return min(1.0, score)

class CardValidator:
    """Credit card validation utilities"""
    
    @staticmethod
    def validate_card_number(card_number: str) -> bool:
        """Validate card number using Luhn algorithm"""
        if not card_number or not card_number.isdigit():
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
                checksum += sum(digits_of(d*2))
            return checksum % 10
        
        return luhn_checksum(card_number) == 0
    
    @staticmethod
    def get_card_brand(card_number: str) -> str:
        """Detect card brand from number"""
        if not card_number:
            return "Unknown"
        
        if card_number.startswith('4'):
            return "Visa"
        elif card_number.startswith(('51', '52', '53', '54', '55')):
            return "Mastercard"
        elif card_number.startswith(('34', '37')):
            return "American Express"
        elif card_number.startswith('6'):
            return "Discover"
        else:
            return "Unknown"
    
    @staticmethod
    def is_test_card(card_number: str) -> bool:
        """Check if card is a test card"""
        test_prefixes = {"411111", "400000", "545454", "555555", "378282", "340000"}
        return any(card_number.startswith(prefix) for prefix in test_prefixes)

class AmountValidator:
    """Transaction amount validation"""
    
    @staticmethod
    def validate_amount(amount: float) -> bool:
        """Validate transaction amount"""
        try:
            amount_float = float(amount)
            return 0.01 <= amount_float <= 1000000.0
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def is_testing_amount(amount: float) -> bool:
        """Check if amount is a testing amount"""
        testing_amounts = {0.01, 1.00, 10.00, 100.00}
        return amount in testing_amounts
    
    @staticmethod
    def get_amount_risk_score(amount: float) -> float:
        """Calculate amount risk score"""
        if not AmountValidator.validate_amount(amount):
            return 1.0
        
        score = 0.1  # Base score
        
        # Testing amount
        if AmountValidator.is_testing_amount(amount):
            score += 0.6
        
        # High amount
        if amount > 10000:
            score += 0.3
        
        # Round numbers
        if amount == int(amount):
            score += 0.1
        
        return min(1.0, score)

class SecurityValidator:
    """Security pattern detection"""
    
    @staticmethod
    def detect_sql_injection(text: str) -> bool:
        """Detect potential SQL injection patterns"""
        if not text:
            return False
        
        sql_patterns = [
            r"(\b(union|select|insert|update|delete|drop|create|alter)\b)",
            r"(--|\#|\/\*|\*\/)",
            r"(\b(or|and)\b\s+\d+\s*=\s*\d+)",
            r"(\bexec\b|\bexecute\b)",
            r"(\bscript\b)",
        ]
        
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in sql_patterns)
    
    @staticmethod
    def detect_xss(text: str) -> bool:
        """Detect potential XSS patterns"""
        if not text:
            return False
        
        xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>",
            r"<object[^>]*>",
            r"<embed[^>]*>",
        ]
        
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in xss_patterns)
    
    @staticmethod
    def detect_command_injection(text: str) -> bool:
        """Detect potential command injection patterns"""
        if not text:
            return False
        
        cmd_patterns = [
            r"[;&|`$]",
            r"(\brm\b|\bdel\b|\bformat\b)",
            r"(\bcat\b|\btype\b).*(\||>)",
            r"(\bwget\b|\bcurl\b).*(\||>)",
        ]
        
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in cmd_patterns)
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        """Basic input sanitization"""
        if not text:
            return ""
        
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', ';', '|', '`', '$']
        sanitized = text
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        return sanitized.strip()
    
    @staticmethod
    def validate_input_security(data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive security validation"""
        risks = []
        is_risky = False
        
        # Check text fields for security patterns
        text_fields = ['email', 'fingerprint', 'user_agent', 'description']
        for field in text_fields:
            if field in data and data[field]:
                text = str(data[field])
                
                if SecurityValidator.detect_sql_injection(text):
                    risks.append("sql_injection")
                    is_risky = True
                
                if SecurityValidator.detect_xss(text):
                    risks.append("xss")
                    is_risky = True
                
                if SecurityValidator.detect_command_injection(text):
                    risks.append("command_injection")
                    is_risky = True
        
        # Check email security
        if 'email' in data:
            if EmailValidator.is_disposable_email(data['email']):
                risks.append("disposable_email")
                is_risky = True
        
        # Check IP security
        if 'ip_address' in data:
            if IPValidator.is_private_ip(data['ip_address']):
                risks.append("private_ip")
            
            ip_risk = IPValidator.get_ip_risk_score(data['ip_address'])
            if ip_risk > 0.7:
                risks.append("high_risk_ip")
                is_risky = True
        
        # Check amount security
        if 'amount' in data:
            if AmountValidator.is_testing_amount(data['amount']):
                risks.append("testing_amount")
                is_risky = True
        
        return {
            'is_risky': is_risky,
            'is_secure': not is_risky,
            'risks': risks,
            'reasons': risks
        }

def validate_transaction_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Performs comprehensive validation on transaction data.
    Raises ValueError for critical validation failures.
    """
    validated_data = {}
    
    # Validate email
    if 'email' in data:
        email = data['email']
        if not EmailValidator.validate_email(email):
            raise ValueError("Invalid email format")
        if EmailValidator.is_disposable_email(email):
            raise ValueError("Disposable email addresses are not allowed")
        validated_data['email'] = email.lower().strip()
    
    # Validate IP address
    if 'ip_address' in data:
        ip = data['ip_address']
        if not IPValidator.validate_ip(ip):
            raise ValueError("Invalid IP address format")
        validated_data['ip_address'] = ip
    
    # Validate amount
    if 'amount' in data:
        amount = float(data['amount'])
        if not AmountValidator.validate_amount(amount):
            raise ValueError("Invalid transaction amount")
        validated_data['amount'] = amount
    
    # Validate countries
    for field in ['card_country', 'billing_country']:
        if field in data:
            country = data[field]
            if not (isinstance(country, str) and len(country) == 2 and country.isalpha()):
                raise ValueError(f"Invalid {field} format")
            validated_data[field] = country.upper()
    
    # Check for security patterns
    security_check = SecurityValidator.validate_input_security(data)
    if security_check['is_risky']:
        raise ValueError(f"Security validation failed: {', '.join(security_check['reasons'])}")
    
    # Copy other fields
    for key, value in data.items():
        if key not in validated_data:
            validated_data[key] = value
    
    return validated_data
