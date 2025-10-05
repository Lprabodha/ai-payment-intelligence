"""
Recommendation Engine for Payment Intelligence
Provides contextual recommendations for fraud and chargeback prevention
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import numpy as np


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskType(Enum):
    FRAUD = "fraud"
    CHARGEBACK = "chargeback"
    BOTH = "both"


class RecommendationEngine:
    """
    Recommendation engine that provides specific actions
    based on fraud risk, chargeback risk, transaction amount, and customer history
    """
    
    def __init__(self):
        self.risk_thresholds = {
            RiskLevel.LOW: 0.2,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.HIGH: 0.75,
            RiskLevel.CRITICAL: 0.9
        }
        
        self.amount_thresholds = {
            "low": 100,
            "medium": 500,
            "high": 2000,
            "critical": 10000
        }

    def classify_risk_level(self, confidence: float, risk_type: RiskType) -> RiskLevel:
        """Classify risk level based on confidence score and risk type"""
        if risk_type == RiskType.FRAUD:
            if confidence >= 0.85:
                return RiskLevel.CRITICAL
            elif confidence >= 0.65:
                return RiskLevel.HIGH
            elif confidence >= 0.35:
                return RiskLevel.MEDIUM
        elif risk_type == RiskType.CHARGEBACK:
            if confidence >= 0.80:
                return RiskLevel.CRITICAL
            elif confidence >= 0.60:
                return RiskLevel.HIGH
            elif confidence >= 0.30:
                return RiskLevel.MEDIUM
        
        return RiskLevel.LOW

    def get_fraud_specific_recommendations(self, 
                                         fraud_level: RiskLevel,
                                         amount: float,
                                         customer_history: Dict,
                                         transaction_features: Dict) -> List[str]:
        """Generate fraud-specific recommendations based on risk level and context"""
        
        recommendations = []
        
        if fraud_level == RiskLevel.CRITICAL:
            recommendations.extend([
                "🚨 IMMEDIATE ACTION: Block transaction and flag for manual review",
                "📞 Contact customer via verified phone number within 30 minutes",
                "🔒 Implement temporary account lockout (24-48 hours)",
                "📋 Require additional KYC documentation (government ID + utility bill)",
                "🔄 Force password reset and 2FA setup",
                "📊 Create fraud case for investigation team"
            ])
            
        elif fraud_level == RiskLevel.HIGH:
            recommendations.extend([
                "⚠️ Require 3DS authentication with step-up verification",
                "⏸️ Hold transaction for 2-hour manual review",
                "📍 Verify billing address with AVS and postal code validation",
                "🔍 Enhanced device fingerprinting and behavioral analysis",
                "📧 Send verification email to customer with confirmation link",
                "🚫 Temporarily restrict high-value transactions (>$500) for 7 days"
            ])
            
        elif fraud_level == RiskLevel.MEDIUM:
            recommendations.extend([
                "🔐 Require 3DS authentication",
                "⏳ Hold for automated review (30 minutes)",
                "📋 Standard AVS and CVC verification",
                "📊 Flag for enhanced monitoring on next 3 transactions",
                "🎯 Implement velocity checks (max 3 transactions/hour)"
            ])
            
        else:  # LOW risk
            recommendations.extend([
                "✅ Process with standard verification",
                "📈 Monitor for pattern changes",
                "🔔 Set up automated alerts for unusual activity"
            ])

        # Amount-based additional recommendations
        if amount >= self.amount_thresholds["critical"]:
            recommendations.append("💰 CRITICAL AMOUNT: Require senior manager approval")
        elif amount >= self.amount_thresholds["high"]:
            recommendations.append("💳 HIGH VALUE: Enhanced verification required")
        
        # Customer history-based recommendations
        if customer_history.get("past_chargebacks", 0) > 0:
            recommendations.append("⚠️ Previous chargeback history: Extra verification required")
        
        if customer_history.get("refund_ratio", 0) > 0.3:
            recommendations.append("🔄 High refund rate: Consider refund protection measures")
        
        if customer_history.get("account_age_days", 0) < 7:
            recommendations.append("🆕 New customer: Enhanced onboarding verification")

        return recommendations

    def get_chargeback_specific_recommendations(self,
                                              chargeback_level: RiskLevel,
                                              amount: float,
                                              customer_history: Dict,
                                              transaction_features: Dict) -> List[str]:
        """Generate chargeback-specific recommendations based on risk level and context"""
        
        recommendations = []
        
        if chargeback_level == RiskLevel.CRITICAL:
            recommendations.extend([
                "🚨 HIGH CHARGEBACK RISK: Implement pre-emptive protection",
                "📄 Collect comprehensive transaction evidence package",
                "📞 Proactive customer communication within 2 hours",
                "🔒 Require signed receipt or delivery confirmation",
                "📋 Enhanced product/service description documentation",
                "⏰ Set up 30-day monitoring for dispute indicators"
            ])
            
        elif chargeback_level == RiskLevel.HIGH:
            recommendations.extend([
                "⚠️ CHARGEBACK PREVENTION: Enhanced documentation required",
                "📧 Send detailed transaction confirmation email",
                "📋 Collect delivery confirmation and signature",
                "🎯 Implement customer satisfaction follow-up (24-48 hours)",
                "📊 Monitor customer communication channels for complaints",
                "🔄 Offer refund option before dispute window"
            ])
            
        elif chargeback_level == RiskLevel.MEDIUM:
            recommendations.extend([
                "📋 Standard chargeback protection measures",
                "📧 Transaction confirmation email",
                "⏰ Follow-up customer satisfaction check (72 hours)",
                "📊 Monitor for customer service complaints"
            ])
            
        else:  # LOW risk
            recommendations.extend([
                "✅ Standard processing with basic protection",
                "📈 Monitor customer satisfaction metrics"
            ])

        # Transaction-specific recommendations
        if transaction_features.get("country_mismatch", False):
            recommendations.append("🌍 Cross-border transaction: Enhanced documentation required")
        
        if transaction_features.get("unusual_amount", False):
            recommendations.append("💰 Unusual amount pattern: Customer communication recommended")
        
        if customer_history.get("past_chargebacks", 0) > 0:
            recommendations.append("⚠️ Previous chargeback history: Enhanced protection measures")
        
        if customer_history.get("dispute_reasons"):
            # Add specific recommendations based on past dispute reasons
            past_reasons = customer_history.get("dispute_reasons", [])
            if "product_not_delivered" in past_reasons:
                recommendations.append("📦 Delivery tracking and confirmation required")
            if "fraudulent" in past_reasons:
                recommendations.append("🔍 Enhanced identity verification required")

        return recommendations

    def get_routing_recommendations(self,
                                  fraud_level: RiskLevel,
                                  chargeback_level: RiskLevel,
                                  amount: float,
                                  customer_history: Dict) -> List[str]:
        """Generate payment gateway routing recommendations"""
        
        recommendations = []
        
        # Gateway selection based on risk profile
        if fraud_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
            recommendations.extend([
                "🛡️ Route to high-security gateway with advanced fraud detection",
                "🔒 Use gateway with strongest 3DS and authentication capabilities",
                "📊 Implement gateway-specific risk scoring"
            ])
        
        if chargeback_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
            recommendations.extend([
                "💳 Route to gateway with best chargeback protection",
                "📋 Use gateway with comprehensive dispute management tools",
                "🔄 Consider gateway with automatic refund capabilities"
            ])
        
        # Amount-based routing
        if amount >= self.amount_thresholds["high"]:
            recommendations.extend([
                "💰 High-value routing: Use premium gateway with enhanced support",
                "🏦 Consider bank direct routing for very high amounts"
            ])
        
        # Customer history-based routing
        if customer_history.get("gateway_success_rate"):
            best_gateway = max(customer_history["gateway_success_rate"], 
                             key=customer_history["gateway_success_rate"].get)
            recommendations.append(f"📈 Route to {best_gateway} (best historical success rate)")

        return recommendations

    def build_comprehensive_recommendations(self,
                                          transaction: Dict,
                                          fraud_pred: Optional[Dict],
                                          chargeback_pred: Optional[Dict]) -> Dict:
        """
        Build comprehensive, contextual recommendations based on all risk factors
        """
        
        tx_id = transaction.get("transaction_id")
        amount = float(transaction.get("amount", 0.0))
        currency = transaction.get("currency", "usd")
        
        # Extract risk information
        fraud_detected = bool(fraud_pred.get("fraud_detected", False)) if fraud_pred else False
        fraud_conf = float(fraud_pred.get("confidence", 0.0)) if fraud_pred else 0.0
        fraud_level = self.classify_risk_level(fraud_conf, RiskType.FRAUD)
        
        chargeback_predicted = bool(chargeback_pred.get("chargeback_predicted", False)) if chargeback_pred else False
        chargeback_conf = float(chargeback_pred.get("confidence_score", 0.0)) if chargeback_pred else 0.0
        chargeback_level = self.classify_risk_level(chargeback_conf, RiskType.CHARGEBACK)
        
        # Determine overall risk type
        risk_type = RiskType.BOTH if (fraud_detected or chargeback_predicted) else RiskType.FRAUD if fraud_detected else RiskType.CHARGEBACK if chargeback_predicted else None
        
        # Build customer history context
        customer_history = {
            "past_chargebacks": transaction.get("past_chargebacks", 0),
            "refund_ratio": transaction.get("customer_refund_ratio", 0.0),
            "account_age_days": transaction.get("account_age_days", 0),
            "dispute_reasons": transaction.get("past_dispute_reasons", []),
            "gateway_success_rate": transaction.get("gateway_success_rates", {})
        }
        
        # Build transaction features
        transaction_features = {
            "country_mismatch": transaction.get("country_mismatch", 0) == 1,
            "unusual_amount": transaction.get("unusual_amount_flag", 0) == 1,
            "time_pattern": transaction.get("time_between_transactions", 0),
            "device_reuse": transaction.get("device_ip_pair_reuse_before", 0)
        }
        
        # Generate specific recommendations
        fraud_recommendations = self.get_fraud_specific_recommendations(
            fraud_level, amount, customer_history, transaction_features
        )
        
        chargeback_recommendations = self.get_chargeback_specific_recommendations(
            chargeback_level, amount, customer_history, transaction_features
        )
        
        routing_recommendations = self.get_routing_recommendations(
            fraud_level, chargeback_level, amount, customer_history
        )
        
        # Combine and deduplicate recommendations
        all_recommendations = fraud_recommendations + chargeback_recommendations + routing_recommendations
        unique_recommendations = list(dict.fromkeys(all_recommendations))  # Preserve order, remove duplicates
        
        # Determine overall priority
        overall_priority = self._calculate_overall_priority(fraud_level, chargeback_level, amount)
        
        # Generate summary insights
        insights = self._generate_insights(fraud_pred, chargeback_pred, transaction_features, customer_history)
        
        return {
            "transaction_id": tx_id,
            "created_at": datetime.utcnow(),
            "overall_priority": overall_priority,
            "risk_assessment": {
                "fraud": {
                    "detected": fraud_detected,
                    "confidence": round(fraud_conf, 4),
                    "level": fraud_level.value,
                    "recommendations": fraud_recommendations
                },
                "chargeback": {
                    "predicted": chargeback_predicted,
                    "confidence": round(chargeback_conf, 4),
                    "level": chargeback_level.value,
                    "recommendations": chargeback_recommendations
                },
                "routing": {
                    "recommendations": routing_recommendations
                }
            },
            "action_plan": {
                "immediate_actions": [r for r in unique_recommendations if "🚨" in r or "⚠️" in r],
                "short_term_actions": [r for r in unique_recommendations if "📞" in r or "📧" in r or "🔐" in r],
                "long_term_actions": [r for r in unique_recommendations if "📊" in r or "🔄" in r],
                "monitoring_actions": [r for r in unique_recommendations if "📈" in r or "🔔" in r]
            },
            "insights": insights,
            "amount_context": {
                "amount": amount,
                "currency": currency,
                "tier": self._get_amount_tier(amount)
            },
            "ttl_days": 30
        }

    def _calculate_overall_priority(self, fraud_level: RiskLevel, chargeback_level: RiskLevel, amount: float) -> str:
        """Calculate overall priority based on all risk factors"""
        
        if fraud_level == RiskLevel.CRITICAL or chargeback_level == RiskLevel.CRITICAL:
            return "critical"
        elif fraud_level == RiskLevel.HIGH or chargeback_level == RiskLevel.HIGH:
            return "high"
        elif fraud_level == RiskLevel.MEDIUM or chargeback_level == RiskLevel.MEDIUM:
            return "medium"
        elif amount >= self.amount_thresholds["high"]:
            return "medium"  # High amount with low risk still needs attention
        else:
            return "low"

    def _get_amount_tier(self, amount: float) -> str:
        """Get amount tier for context"""
        if amount >= self.amount_thresholds["critical"]:
            return "critical"
        elif amount >= self.amount_thresholds["high"]:
            return "high"
        elif amount >= self.amount_thresholds["medium"]:
            return "medium"
        else:
            return "low"

    def _generate_insights(self, fraud_pred: Optional[Dict], chargeback_pred: Optional[Dict], 
                          transaction_features: Dict, customer_history: Dict) -> List[str]:
        """Generate actionable insights based on risk patterns"""
        
        insights = []
        
        if fraud_pred and fraud_pred.get("reasons"):
            insights.append(f"Fraud indicators: {', '.join(fraud_pred['reasons'][:3])}")
        
        if chargeback_pred and chargeback_pred.get("chargeback_reason"):
            insights.append(f"Chargeback risk factors: {chargeback_pred['chargeback_reason']}")
        
        if transaction_features.get("country_mismatch"):
            insights.append("Cross-border transaction detected - enhanced verification recommended")
        
        if transaction_features.get("device_reuse", 0) > 3:
            insights.append("Device reuse pattern detected - monitor for account takeover")
        
        if customer_history.get("refund_ratio", 0) > 0.2:
            insights.append("Customer has elevated refund rate - consider satisfaction improvement")
        
        if customer_history.get("account_age_days", 0) < 30:
            insights.append("New customer - implement enhanced onboarding")
        
        return insights


# Global recommendation engine instance
recommendation_engine = RecommendationEngine()
