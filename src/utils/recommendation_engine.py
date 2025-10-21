"""
Enhanced Recommendation Engine for Payment Intelligence
Provides ML-driven contextual recommendations for fraud and chargeback prevention
with effectiveness tracking and cost-benefit analysis
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskType(Enum):
    FRAUD = "fraud"
    CHARGEBACK = "chargeback"
    BOTH = "both"


class ActionCategory(Enum):
    IMMEDIATE = "immediate"  # Must be done now
    SHORT_TERM = "short_term"  # Within 24 hours
    MEDIUM_TERM = "medium_term"  # Within 7 days
    LONG_TERM = "long_term"  # Ongoing monitoring
    

class RecommendationEngine:
    """
    Enhanced recommendation engine with ML-driven insights,
    effectiveness tracking, and cost-benefit analysis
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
        
        # Cost estimates for different actions (in USD)
        self.action_costs = {
            "manual_review": 5.0,
            "phone_verification": 8.0,
            "document_verification": 15.0,
            "senior_approval": 12.0,
            "3ds_auth": 0.5,
            "enhanced_monitoring": 2.0,
            "customer_communication": 3.0,
            "refund_protection": 10.0,
        }
        
        # Expected effectiveness (reduction in fraud/chargeback rate)
        self.action_effectiveness = {
            "manual_review": 0.35,
            "phone_verification": 0.45,
            "document_verification": 0.55,
            "senior_approval": 0.25,
            "3ds_auth": 0.65,
            "enhanced_monitoring": 0.15,
            "customer_communication": 0.20,
            "refund_protection": 0.30,
        }
        
        # Track recommendation effectiveness over time
        self.recommendation_history = defaultdict(lambda: {
            "total_given": 0,
            "successful": 0,
            "failed": 0,
            "avg_fraud_prevented": 0.0,
            "avg_cost": 0.0
        })

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
        Build enhanced ML-driven recommendations with cost-benefit analysis
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
        
        # Calculate combined ML-driven risk score
        combined_risk_score = self.calculate_combined_risk_score(
            fraud_conf, chargeback_conf, amount, customer_history
        )
        
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
        unique_recommendations = list(dict.fromkeys(all_recommendations))
        
        # Calculate cost-benefit for key actions
        cost_benefit_analysis = []
        key_actions = ["manual_review", "3ds_auth", "phone_verification", "document_verification"]
        
        for action in key_actions:
            if combined_risk_score > 0.3:  # Only calculate for transactions with some risk
                roi_analysis = self.calculate_action_roi(action, amount, combined_risk_score)
                if roi_analysis["recommended"]:
                    cost_benefit_analysis.append(roi_analysis)
        
        # Prioritize actions by ROI
        cost_benefit_analysis = sorted(
            cost_benefit_analysis,
            key=lambda x: x["roi_percentage"],
            reverse=True
        )
        
        # Determine overall priority
        overall_priority = self._calculate_overall_priority(fraud_level, chargeback_level, amount)
        
        # Generate ML-driven insights
        insights = self._generate_insights(fraud_pred, chargeback_pred, transaction_features, customer_history)
        
        # Enhanced action plan with timing
        action_plan = {
            "immediate_actions": {
                "actions": [r for r in unique_recommendations if "🚨" in r or "IMMEDIATE" in r],
                "timeframe": "Within 5 minutes",
                "priority": "CRITICAL"
            },
            "short_term_actions": {
                "actions": [r for r in unique_recommendations if "📞" in r or "📧" in r or "🔐" in r],
                "timeframe": "Within 2 hours",
                "priority": "HIGH"
            },
            "medium_term_actions": {
                "actions": [r for r in unique_recommendations if "📋" in r or "🔍" in r],
                "timeframe": "Within 24 hours",
                "priority": "MEDIUM"
            },
            "long_term_actions": {
                "actions": [r for r in unique_recommendations if "📊" in r or "🔄" in r],
                "timeframe": "Within 7 days",
                "priority": "LOW"
            },
            "monitoring_actions": {
                "actions": [r for r in unique_recommendations if "📈" in r or "🔔" in r],
                "timeframe": "Ongoing",
                "priority": "CONTINUOUS"
            }
        }
        
        return {
            "transaction_id": tx_id,
            "created_at": datetime.utcnow().isoformat(),
            "model_version": "3.0.0",
            "overall_priority": overall_priority,
            "combined_risk_score": round(combined_risk_score, 4),
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
            "action_plan": action_plan,
            "cost_benefit_analysis": cost_benefit_analysis,
            "insights": insights,
            "amount_context": {
                "amount": amount,
                "currency": currency,
                "tier": self._get_amount_tier(amount)
            },
            "recommended_next_steps": [
                action["action"] for action in cost_benefit_analysis[:3]
            ] if cost_benefit_analysis else ["standard_processing"],
            "estimated_total_cost": round(sum(a["cost"] for a in cost_benefit_analysis), 2),
            "estimated_total_savings": round(sum(a["expected_savings"] for a in cost_benefit_analysis), 2),
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

    def calculate_action_roi(self, action_type: str, amount: float, risk_probability: float) -> Dict:
        """Calculate ROI for a specific action"""
        
        cost = self.action_costs.get(action_type, 5.0)
        effectiveness = self.action_effectiveness.get(action_type, 0.25)
        
        # Expected loss without action
        expected_loss = amount * risk_probability
        
        # Expected loss with action (reduced by effectiveness)
        expected_loss_with_action = expected_loss * (1 - effectiveness)
        
        # Savings from action
        savings = expected_loss - expected_loss_with_action
        
        # Net benefit
        net_benefit = savings - cost
        
        # ROI
        roi = (net_benefit / cost) * 100 if cost > 0 else 0
        
        return {
            "action": action_type,
            "cost": cost,
            "expected_savings": savings,
            "net_benefit": net_benefit,
            "roi_percentage": roi,
            "recommended": roi > 0
        }
    
    def calculate_combined_risk_score(self, fraud_conf: float, chargeback_conf: float, 
                                     amount: float, customer_history: Dict) -> float:
        """Calculate a combined ML-driven risk score"""
        
        # Base risk from predictions
        fraud_weight = 0.6
        chargeback_weight = 0.4
        base_risk = (fraud_conf * fraud_weight) + (chargeback_conf * chargeback_weight)
        
        # Amount risk multiplier
        amount_multiplier = 1.0
        if amount > self.amount_thresholds["critical"]:
            amount_multiplier = 1.5
        elif amount > self.amount_thresholds["high"]:
            amount_multiplier = 1.3
        elif amount > self.amount_thresholds["medium"]:
            amount_multiplier = 1.1
        
        # Customer history adjustment
        history_adjustment = 0.0
        if customer_history.get("past_chargebacks", 0) > 0:
            history_adjustment += 0.15
        if customer_history.get("refund_ratio", 0) > 0.3:
            history_adjustment += 0.10
        if customer_history.get("account_age_days", 0) < 7:
            history_adjustment += 0.10
        
        # Combined score
        combined_score = min(1.0, (base_risk * amount_multiplier) + history_adjustment)
        
        return combined_score
    
    def prioritize_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """Prioritize recommendations based on ROI and urgency"""
        
        # Sort by ROI (descending) and then by urgency
        sorted_recs = sorted(
            recommendations,
            key=lambda x: (x.get("roi_percentage", 0), x.get("urgency_score", 0)),
            reverse=True
        )
        
        return sorted_recs
    
    def track_recommendation_effectiveness(self, recommendation_id: str, 
                                          action_taken: str, 
                                          outcome: str,
                                          fraud_prevented: bool,
                                          cost: float):
        """Track the effectiveness of a recommendation"""
        
        self.recommendation_history[action_taken]["total_given"] += 1
        
        if fraud_prevented:
            self.recommendation_history[action_taken]["successful"] += 1
        else:
            self.recommendation_history[action_taken]["failed"] += 1
        
        # Update running averages
        total = self.recommendation_history[action_taken]["total_given"]
        
        prev_avg_prevented = self.recommendation_history[action_taken]["avg_fraud_prevented"]
        self.recommendation_history[action_taken]["avg_fraud_prevented"] = (
            (prev_avg_prevented * (total - 1) + (1 if fraud_prevented else 0)) / total
        )
        
        prev_avg_cost = self.recommendation_history[action_taken]["avg_cost"]
        self.recommendation_history[action_taken]["avg_cost"] = (
            (prev_avg_cost * (total - 1) + cost) / total
        )
    
    def get_action_effectiveness_report(self) -> Dict:
        """Get effectiveness report for all actions"""
        
        report = {}
        for action, stats in self.recommendation_history.items():
            success_rate = (stats["successful"] / stats["total_given"] * 100) if stats["total_given"] > 0 else 0
            
            report[action] = {
                "total_recommendations": stats["total_given"],
                "success_rate": round(success_rate, 2),
                "avg_fraud_prevention_rate": round(stats["avg_fraud_prevented"] * 100, 2),
                "avg_cost": round(stats["avg_cost"], 2),
                "estimated_roi": round((stats["avg_fraud_prevented"] * 100) / (stats["avg_cost"] + 0.01), 2)
            }
        
        return report

    def _generate_insights(self, fraud_pred: Optional[Dict], chargeback_pred: Optional[Dict], 
                          transaction_features: Dict, customer_history: Dict) -> List[str]:
        """Generate ML-driven actionable insights based on risk patterns"""
        
        insights = []
        
        # Fraud-specific insights with severity
        if fraud_pred and fraud_pred.get("reasons"):
            top_reasons = fraud_pred['reasons'][:3]
            fraud_conf = fraud_pred.get("confidence", 0)
            severity = "CRITICAL" if fraud_conf > 0.8 else "HIGH" if fraud_conf > 0.6 else "MODERATE"
            insights.append(f"[{severity}] Fraud indicators: {', '.join(top_reasons)}")
        
        # Chargeback-specific insights
        if chargeback_pred and chargeback_pred.get("chargeback_reason"):
            cb_conf = chargeback_pred.get("confidence_score", 0)
            severity = "CRITICAL" if cb_conf > 0.75 else "HIGH" if cb_conf > 0.55 else "MODERATE"
            insights.append(f"[{severity}] Chargeback risk: {chargeback_pred['chargeback_reason']}")
        
        # Transaction-based insights
        if transaction_features.get("country_mismatch"):
            insights.append("[MEDIUM] Cross-border transaction - enhanced verification recommended")
        
        if transaction_features.get("device_reuse", 0) > 5:
            insights.append("[HIGH] Suspicious device reuse pattern - potential account takeover")
        elif transaction_features.get("device_reuse", 0) > 3:
            insights.append("[MEDIUM] Device reuse detected - monitor for unusual activity")
        
        # Customer behavior insights
        refund_ratio = customer_history.get("refund_ratio", 0)
        if refund_ratio > 0.4:
            insights.append("[HIGH] Very high refund rate (>40%) - consider customer satisfaction review")
        elif refund_ratio > 0.2:
            insights.append("[MEDIUM] Elevated refund rate - proactive communication recommended")
        
        # Account age insights
        account_age = customer_history.get("account_age_days", 0)
        if account_age < 7:
            insights.append("[HIGH] Brand new customer - implement enhanced onboarding and verification")
        elif account_age < 30:
            insights.append("[MEDIUM] New customer (< 30 days) - additional monitoring recommended")
        
        # Historical pattern insights
        if customer_history.get("past_chargebacks", 0) > 2:
            insights.append("[CRITICAL] Multiple previous chargebacks - high-risk customer profile")
        elif customer_history.get("past_chargebacks", 0) > 0:
            insights.append("[HIGH] Previous chargeback history - implement protection measures")
        
        # Velocity-based insights
        if transaction_features.get("time_pattern", 0) < 60:
            insights.append("[HIGH] Rapid successive transactions - potential card testing or fraud")
        
        return insights


# Global recommendation engine instance
recommendation_engine = RecommendationEngine()
