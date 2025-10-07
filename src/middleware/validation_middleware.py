"""
Data validation middleware for TransactIQ
Implements comprehensive input validation and security checks
"""

import json
from typing import Dict, Any, Optional
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from utils.validation import (
    validate_transaction_data,
    SecurityValidator,
    Sanitizer,
    EmailValidator,
    IPValidator,
    AmountValidator,
    CardValidator
)


class ValidationMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive data validation and security checks"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        # Endpoints that require enhanced validation
        self.validation_endpoints = {
            "/predict/fraud",
            "/predict/chargeback", 
            "/predict/payment_gateway",
            "/risk/score",
            "/webhook/stripe",
            "/webhook/solidgate"
        }
    
    async def dispatch(self, request: Request, call_next):
        """Process request with validation middleware"""
        
        # Skip validation for non-POST requests
        if request.method != "POST":
            return await call_next(request)
        
        # Skip validation for non-targeted endpoints
        if request.url.path not in self.validation_endpoints:
            return await call_next(request)
        
        try:
            # Get request body
            body = await request.body()
            
            if not body:
                return await call_next(request)
            
            # Parse JSON
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"detail": "Invalid JSON format"}
                )
            
            # Perform security validation
            security_result = SecurityValidator.validate_input_security(data)
            
            # Check if request is secure
            if not security_result["is_secure"]:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "detail": "Security validation failed",
                        "issues": security_result["issues"],
                        "risk_score": security_result["risk_score"]
                    }
                )
            
            # Sanitize data
            sanitized_data = Sanitizer.sanitize_json(data)
            
            # Validate transaction data for prediction endpoints
            if request.url.path.startswith("/predict/") or request.url.path.startswith("/risk/"):
                validation_result = validate_transaction_data(sanitized_data)
                
                if not validation_result["is_valid"]:
                    return JSONResponse(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        content={
                            "detail": "Data validation failed",
                            "error": validation_result["error"]
                        }
                    )
                
                # Add validation metadata to request
                request.state.validation_metadata = {
                    "is_validated": True,
                    "security_score": security_result["risk_score"],
                    "validation_metadata": validation_result.get("security", {})
                }
            
            # Create new request with sanitized data
            new_body = json.dumps(sanitized_data).encode()
            
            # Create new request with sanitized body
            async def receive():
                return {
                    "type": "http.request",
                    "body": new_body
                }
            
            # Store original receive function
            original_receive = request._receive
            
            # Replace receive function
            request._receive = receive
            
            # Process request
            response = await call_next(request)
            
            # Restore original receive function
            request._receive = original_receive
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": f"Validation middleware error: {str(e)}"}
            )


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses"""
    
    async def dispatch(self, request: Request, call_next):
        """Add security headers to response"""
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        # Remove server information
        if "server" in response.headers:
            del response.headers["server"]
        
        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to limit request body size"""
    
    def __init__(self, app: ASGIApp, max_size: int = 10 * 1024 * 1024):  # 10MB default
        super().__init__(app)
        self.max_size = max_size
    
    async def dispatch(self, request: Request, call_next):
        """Check request body size"""
        content_length = request.headers.get("content-length")
        
        if content_length:
            try:
                size = int(content_length)
                if size > self.max_size:
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content={"detail": f"Request body too large. Maximum size: {self.max_size} bytes"}
                    )
            except ValueError:
                pass
        
        return await call_next(request)


class ContentTypeValidationMiddleware(BaseHTTPMiddleware):
    """Middleware to validate content type for POST requests"""
    
    async def dispatch(self, request: Request, call_next):
        """Validate content type"""
        if request.method == "POST":
            content_type = request.headers.get("content-type", "")
            
            # Allow JSON content types
            allowed_types = [
                "application/json",
                "application/json; charset=utf-8"
            ]
            
            if not any(content_type.startswith(allowed) for allowed in allowed_types):
                return JSONResponse(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    content={"detail": "Content-Type must be application/json"}
                )
        
        return await call_next(request)


class InputSanitizationMiddleware(BaseHTTPMiddleware):
    """Middleware for input sanitization and validation"""
    
    async def dispatch(self, request: Request, call_next):
        """Sanitize and validate input data"""
        
        # Only process POST requests with JSON content
        if (request.method == "POST" and 
            request.headers.get("content-type", "").startswith("application/json")):
            
            try:
                # Get request body
                body = await request.body()
                
                if body:
                    # Parse JSON
                    data = json.loads(body)
                    
                    # Sanitize data
                    sanitized_data = Sanitizer.sanitize_json(data)
                    
                    # Check for security threats
                    security_check = SecurityValidator.validate_input_security(sanitized_data)
                    
                    if not security_check["is_secure"]:
                        return JSONResponse(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            content={
                                "detail": "Input contains potentially malicious content",
                                "issues": security_check["issues"]
                            }
                        )
                    
                    # Create new request with sanitized data
                    new_body = json.dumps(sanitized_data).encode()
                    
                    # Create new request with sanitized body
                    async def receive():
                        return {
                            "type": "http.request",
                            "body": new_body
                        }
                    
                    # Store original receive function
                    original_receive = request._receive
                    
                    # Replace receive function
                    request._receive = receive
                    
                    # Process request
                    response = await call_next(request)
                    
                    # Restore original receive function
                    request._receive = original_receive
                    
                    return response
            
            except json.JSONDecodeError:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"detail": "Invalid JSON format"}
                )
            except Exception as e:
                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={"detail": f"Input sanitization error: {str(e)}"}
                )
        
        return await call_next(request)
