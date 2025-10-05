"""
Pydantic models for API requests and responses
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class TransactionRequest(BaseModel):
    """Request model for transaction prediction"""
    amount: float = Field(..., description="Transaction amount")
    currency: str = Field(..., description="Currency code")
    email: str = Field(..., description="Customer email")
    ip_address: str = Field(..., description="Customer IP address")
    card_country: str = Field(..., description="Card issuing country")
    billing_country: str = Field(..., description="Billing country")
    card_brand: str = Field(..., description="Card brand (VISA, MASTERCARD, etc.)")
    funding_type: str = Field(..., description="Funding type (credit, debit)")
    fingerprint: str = Field(..., description="Device fingerprint")
    risk_score: Optional[float] = Field(None, description="Risk score from payment processor")
    three_d_secure: Optional[str] = Field(None, description="3D Secure status")
    cvc_check: Optional[str] = Field(None, description="CVC check status")
    address_line1_check: Optional[str] = Field(None, description="Address line 1 check")
    postal_code_check: Optional[str] = Field(None, description="Postal code check")
    outcome_type: Optional[str] = Field(None, description="Outcome type")
    seller_message: Optional[str] = Field(None, description="Seller message")
    network_status: Optional[str] = Field(None, description="Network status")

class FraudPredictionResponse(BaseModel):
    """Response model for fraud prediction"""
    is_fraud: bool = Field(..., description="Whether transaction is fraudulent")
    confidence_score: float = Field(..., description="Confidence score (0-1)")
    risk_level: str = Field(..., description="Risk level (low, medium, high)")
    fraud_reasons: List[str] = Field(..., description="List of fraud indicators")
    model_type: str = Field(..., description="Model used for prediction")

class ChargebackPredictionResponse(BaseModel):
    """Response model for chargeback prediction"""
    chargeback_predicted: bool = Field(..., description="Whether chargeback is predicted")
    confidence_score: float = Field(..., description="Confidence score (0-1)")
    chargeback_reason: str = Field(..., description="Reason for chargeback prediction")
    model_type: str = Field(..., description="Model used for prediction")

class RoutingPredictionResponse(BaseModel):
    """Response model for smart routing prediction"""
    recommended_gateway: str = Field(..., description="Recommended payment gateway")
    confidence: float = Field(..., description="Confidence score (0-1)")
    all_scores: Dict[str, float] = Field(..., description="Scores for all gateways")
    current_gateway: str = Field(..., description="Current gateway used")
    error: Optional[str] = Field(None, description="Error message if any")

class RevenuePredictionResponse(BaseModel):
    """Response model for subscription revenue prediction"""
    predicted_revenue: float = Field(..., description="Predicted revenue amount")
    current_revenue: float = Field(..., description="Current revenue amount")
    growth_rate: float = Field(..., description="Growth rate percentage")
    error: Optional[str] = Field(None, description="Error message if any")

class RecommendationResponse(BaseModel):
    """Response model for transaction recommendations"""
    transaction_id: str = Field(..., description="Transaction ID")
    recommendations: List[Dict[str, Any]] = Field(..., description="List of recommendations")
    risk_level: str = Field(..., description="Overall risk level")
    confidence: float = Field(..., description="Overall confidence score")

class WebhookResponse(BaseModel):
    """Response model for webhook processing"""
    status: int = Field(..., description="HTTP status code")
    message: str = Field(..., description="Response message")
    error: Optional[str] = Field(None, description="Error message if any")

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Response timestamp")
    version: str = Field(..., description="API version")
    database: str = Field(..., description="Database status")
    models: Dict[str, str] = Field(..., description="Model status")
