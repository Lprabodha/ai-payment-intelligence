"""
Risk scoring API routes
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from risk_engine.api import risk_scoring_api
from risk_engine.models import RiskLevel, DecisionAction

router = APIRouter(prefix="/risk", tags=["Risk Scoring"])

class TransactionRequest(BaseModel):
    """Transaction data for risk scoring"""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    email: Optional[str] = Field(None, description="User email address")
    amount: float = Field(..., description="Transaction amount")
    currency: str = Field("USD", description="Transaction currency")
    payment_method: str = Field("card", description="Payment method")
    card_brand: Optional[str] = Field(None, description="Card brand")
    card_country: Optional[str] = Field(None, description="Card country")
    ip_address: Optional[str] = Field(None, description="IP address")
    device_fingerprint: Optional[str] = Field(None, description="Device fingerprint")
    user_agent: Optional[str] = Field(None, description="User agent string")
    billing_country: Optional[str] = Field(None, description="Billing country")
    shipping_country: Optional[str] = Field(None, description="Shipping country")
    merchant_id: Optional[str] = Field(None, description="Merchant identifier")
    product_category: Optional[str] = Field(None, description="Product category")
    timestamp: Optional[datetime] = Field(None, description="Transaction timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class BatchTransactionRequest(BaseModel):
    """Batch transaction request"""
    transactions: List[TransactionRequest] = Field(..., description="List of transactions to score")

class RiskThresholdsRequest(BaseModel):
    """Risk thresholds update request"""
    low_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Low risk threshold")
    medium_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Medium risk threshold")
    high_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="High risk threshold")
    critical_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Critical risk threshold")
    approve_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Approve threshold")
    review_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Review threshold")
    decline_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Decline threshold")

class RiskAssessmentResponse(BaseModel):
    """Risk assessment response"""
    transaction_id: str
    overall_risk_score: float
    risk_level: str
    decision_action: str
    confidence: float
    reasoning: List[str]
    ml_prediction: Optional[Dict[str, Any]]
    rule_evaluation: Optional[Dict[str, Any]]
    velocity_features: Optional[Dict[str, Any]]
    ip_features: Optional[Dict[str, Any]]
    device_features: Optional[Dict[str, Any]]
    geo_features: Optional[Dict[str, Any]]
    assessment_time: str
    metadata: Optional[Dict[str, Any]]

@router.post("/score", response_model=RiskAssessmentResponse)
async def score_transaction(transaction: TransactionRequest):
    """Score a single transaction for risk"""
    try:
        result = risk_scoring_api.score_transaction(transaction.dict())
        return RiskAssessmentResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/score/batch")
async def score_transaction_batch(batch_request: BatchTransactionRequest):
    """Score multiple transactions for risk"""
    try:
        transactions = [tx.dict() for tx in batch_request.transactions]
        results = risk_scoring_api.score_transaction_batch(transactions)
        return {"results": results, "total_transactions": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/assessment/{transaction_id}")
async def get_risk_assessment(transaction_id: str):
    """Get risk assessment for a specific transaction"""
    try:
        assessment = risk_scoring_api.get_risk_assessment(transaction_id)
        if not assessment:
            raise HTTPException(status_code=404, detail="Risk assessment not found")
        return assessment
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/profile/{user_id}")
async def get_user_risk_profile(user_id: str):
    """Get risk profile for a user"""
    try:
        profile = risk_scoring_api.get_user_risk_profile(user_id)
        return profile
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/thresholds")
async def update_risk_thresholds(thresholds: RiskThresholdsRequest):
    """Update risk scoring thresholds"""
    try:
        result = risk_scoring_api.update_risk_thresholds(thresholds.dict(exclude_unset=True))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def get_risk_metrics():
    """Get risk scoring metrics and statistics"""
    try:
        metrics = risk_scoring_api.get_risk_metrics()
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check for risk scoring service"""
    try:
        # Test basic functionality
        test_transaction = {
            "transaction_id": "health_check_test",
            "amount": 100.0,
            "currency": "USD"
        }
        
        result = risk_scoring_api.score_transaction(test_transaction)
        
        return {
            "status": "healthy",
            "service": "risk_scoring_engine",
            "timestamp": datetime.utcnow().isoformat(),
            "test_assessment": {
                "transaction_id": result["transaction_id"],
                "risk_level": result["risk_level"],
                "decision_action": result["decision_action"]
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "risk_scoring_engine",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

@router.get("/rules")
async def get_risk_rules():
    """Get available risk rules"""
    try:
        rules = []
        for rule in risk_scoring_api.engine.risk_rules:
            rules.append({
                "name": rule["name"],
                "description": rule["description"],
                "score": rule["score"],
                "weight": rule["weight"]
            })
        
        return {
            "total_rules": len(rules),
            "rules": rules
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    try:
        stats = risk_scoring_api.cache.get_cache_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/cache/user/{user_id}")
async def invalidate_user_cache(user_id: str):
    """Invalidate cache for a specific user"""
    try:
        success = risk_scoring_api.cache.invalidate_user_cache(user_id)
        return {
            "success": success,
            "user_id": user_id,
            "message": "User cache invalidated" if success else "Failed to invalidate user cache"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
