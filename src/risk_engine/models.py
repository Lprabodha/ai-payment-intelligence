"""
Data models for real-time risk scoring engine
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DecisionAction(Enum):
    """Decision action enumeration"""
    APPROVE = "approve"
    REVIEW = "review"
    DECLINE = "decline"
    CHALLENGE = "challenge"

@dataclass
class TransactionData:
    """Structured transaction data for risk assessment"""
    transaction_id: str
    user_id: Optional[str] = None
    email: Optional[str] = None
    amount: float = 0.0
    currency: str = "USD"
    payment_method: str = "card"
    card_brand: Optional[str] = None
    card_country: Optional[str] = None
    ip_address: Optional[str] = None
    device_fingerprint: Optional[str] = None
    user_agent: Optional[str] = None
    billing_country: Optional[str] = None
    shipping_country: Optional[str] = None
    merchant_id: Optional[str] = None
    product_category: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VelocityFeatures:
    """User velocity features"""
    transactions_last_hour: int = 0
    transactions_last_day: int = 0
    transactions_last_week: int = 0
    amount_last_hour: float = 0.0
    amount_last_day: float = 0.0
    amount_last_week: float = 0.0
    unique_merchants_last_day: int = 0
    unique_countries_last_day: int = 0
    avg_transaction_amount: float = 0.0
    max_transaction_amount: float = 0.0

@dataclass
class IPReputationFeatures:
    """IP reputation features"""
    is_proxy: bool = False
    is_vpn: bool = False
    is_tor: bool = False
    is_datacenter: bool = False
    country_mismatch: bool = False
    risk_score: float = 0.0
    reputation_score: float = 0.0

@dataclass
class DeviceFeatures:
    """Device fingerprint features"""
    is_mobile: bool = False
    is_bot: bool = False
    browser_risk: float = 0.0
    os_risk: float = 0.0
    device_consistency: float = 0.0
    fingerprint_risk: float = 0.0

@dataclass
class GeographicFeatures:
    """Geographic velocity features"""
    distance_from_home: float = 0.0
    timezone_mismatch: bool = False
    country_risk_score: float = 0.0
    velocity_anomaly: bool = False
    location_consistency: float = 0.0

@dataclass
class MLPrediction:
    """ML model prediction results"""
    fraud_probability: float = 0.0
    confidence: float = 0.0
    model_version: str = "1.0"
    features_used: List[str] = field(default_factory=list)
    prediction_time: datetime = field(default_factory=datetime.utcnow)

@dataclass
class RuleEvaluation:
    """Rule-based evaluation results"""
    rules_triggered: List[str] = field(default_factory=list)
    rule_scores: Dict[str, float] = field(default_factory=dict)
    total_rule_score: float = 0.0
    evaluation_time: datetime = field(default_factory=datetime.utcnow)

@dataclass
class RiskAssessment:
    """Complete risk assessment result"""
    transaction_id: str
    overall_risk_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.MEDIUM
    decision_action: DecisionAction = DecisionAction.APPROVE
    ml_prediction: Optional[MLPrediction] = None
    rule_evaluation: Optional[RuleEvaluation] = None
    velocity_features: Optional[VelocityFeatures] = None
    ip_features: Optional[IPReputationFeatures] = None
    device_features: Optional[DeviceFeatures] = None
    geo_features: Optional[GeographicFeatures] = None
    reasoning: List[str] = field(default_factory=list)
    confidence: float = 0.0
    assessment_time: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RiskThresholds:
    """Risk scoring thresholds"""
    low_threshold: float = 0.3
    medium_threshold: float = 0.6
    high_threshold: float = 0.8
    critical_threshold: float = 0.9
    approve_threshold: float = 0.4
    review_threshold: float = 0.7
    decline_threshold: float = 0.9
