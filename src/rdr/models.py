"""
RDR Data Models and Schemas
"""

from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class RDRAlertType(str, Enum):
    """Types of RDR alerts"""
    INQUIRY = "inquiry"  # Customer inquiry before dispute
    PRE_DISPUTE = "pre_dispute"  # Dispute about to be filed
    EARLY_WARNING = "early_warning"  # High chargeback risk detected
    FRAUD_SUSPECTED = "fraud_suspected"  # Fraud indicators present
    REFUND_REQUEST = "refund_request"  # Customer requested refund


class RDRDecisionType(str, Enum):
    """RDR decision types"""
    AUTO_REFUND = "auto_refund"  # Automatically issue refund
    MANUAL_REVIEW = "manual_review"  # Requires human review
    CONTACT_CUSTOMER = "contact_customer"  # Contact customer first
    DECLINE_REFUND = "decline_refund"  # Do not issue refund
    GATHER_EVIDENCE = "gather_evidence"  # Collect evidence for dispute


class RefundStatus(str, Enum):
    """Refund processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RDRAlert(BaseModel):
    """RDR Alert data model"""
    alert_id: str
    transaction_id: str
    alert_type: RDRAlertType
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Transaction details
    amount: float
    currency: str = "usd"
    customer_email: str
    customer_id: Optional[str] = None
    
    # Alert details
    dispute_reason: Optional[str] = None
    dispute_amount: Optional[float] = None
    alert_source: str = "system"  # system, verifi, ethoca, customer
    priority: str = "medium"  # low, medium, high, critical
    
    # Risk assessment
    fraud_score: Optional[float] = None
    chargeback_score: Optional[float] = None
    combined_risk_score: Optional[float] = None
    
    # Customer history
    customer_lifetime_value: Optional[float] = None
    previous_disputes: int = 0
    refund_history_count: int = 0
    account_age_days: int = 0
    
    # Status
    status: str = "new"  # new, reviewing, resolved, escalated
    resolved_at: Optional[datetime] = None
    resolution_method: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "alert_id": "rdr_alert_123",
                "transaction_id": "tx_456",
                "alert_type": "pre_dispute",
                "amount": 150.0,
                "currency": "usd",
                "customer_email": "customer@example.com",
                "dispute_reason": "Product not as described",
                "priority": "high"
            }
        }


class RDRDecision(BaseModel):
    """RDR decision model"""
    decision_id: str
    alert_id: str
    transaction_id: str
    decision_type: RDRDecisionType
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Decision rationale
    confidence: float  # 0.0 to 1.0
    reasoning: List[str]
    risk_factors: List[str]
    
    # Action details
    refund_amount: Optional[float] = None
    refund_recommended: bool = False
    auto_process: bool = False
    
    # Cost-benefit analysis
    refund_cost: float = 0.0
    chargeback_cost_avoided: float = 0.0
    expected_roi: float = 0.0
    
    # Follow-up actions
    recommended_actions: List[str] = []
    escalate_to_human: bool = False
    contact_customer: bool = False
    gather_evidence: bool = False
    
    class Config:
        json_schema_extra = {
            "example": {
                "decision_id": "dec_789",
                "alert_id": "rdr_alert_123",
                "transaction_id": "tx_456",
                "decision_type": "auto_refund",
                "confidence": 0.85,
                "refund_recommended": True,
                "auto_process": True
            }
        }


class RefundRequest(BaseModel):
    """Refund request model"""
    refund_id: str
    transaction_id: str
    alert_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Refund details
    amount: float
    currency: str = "usd"
    reason: str
    refund_type: str = "full"  # full, partial
    
    # Status tracking
    status: RefundStatus = RefundStatus.PENDING
    gateway: str  # stripe, solidgate, etc.
    gateway_refund_id: Optional[str] = None
    
    # Processing info
    processed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    failed_reason: Optional[str] = None
    retry_count: int = 0
    
    # Customer info
    customer_email: str
    customer_id: Optional[str] = None
    notify_customer: bool = True
    
    # Chargeback prevention
    prevented_chargeback: Optional[bool] = None
    chargeback_cost_avoided: Optional[float] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "refund_id": "ref_123",
                "transaction_id": "tx_456",
                "amount": 150.0,
                "currency": "usd",
                "reason": "Customer requested refund - RDR",
                "status": "pending",
                "gateway": "stripe"
            }
        }


class RDRMetrics(BaseModel):
    """RDR system metrics"""
    total_alerts: int = 0
    alerts_today: int = 0
    auto_refunds_issued: int = 0
    manual_reviews: int = 0
    chargebacks_prevented: int = 0
    total_refund_amount: float = 0.0
    total_cost_avoided: float = 0.0
    success_rate: float = 0.0
    average_response_time_seconds: float = 0.0
    
    # By alert type
    alerts_by_type: Dict[str, int] = {}
    
    # By decision type
    decisions_by_type: Dict[str, int] = {}
    
    # Time period
    period_start: datetime
    period_end: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_alerts": 145,
                "auto_refunds_issued": 98,
                "chargebacks_prevented": 87,
                "total_cost_avoided": 13050.0,
                "success_rate": 0.87
            }
        }

