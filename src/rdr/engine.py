"""
RDR (Rapid Dispute Resolution) Engine

Main engine for processing RDR alerts and making refund decisions
"""

import os
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pymongo import MongoClient
import numpy as np

from .models import RDRAlert, RDRDecision, RDRAlertType, RDRDecisionType
from utils.logger import get_logger

logger = get_logger(__name__)


class RDREngine:
    """
    RDR Engine for intelligent dispute resolution
    
    Features:
    - ML-driven refund decisions
    - Cost-benefit analysis
    - Automatic risk assessment
    - Customer lifetime value consideration
    """
    
    def __init__(self, db: MongoClient, fraud_service=None, chargeback_service=None):
        self.db = db
        self.fraud_service = fraud_service
        self.chargeback_service = chargeback_service
        
        # RDR Configuration
        self.config = {
            # Thresholds for auto-refund
            "auto_refund_amount_max": 500.0,  # Max amount for auto-refund
            "auto_refund_confidence_min": 0.75,  # Min confidence for auto-refund
            "high_value_threshold": 2000.0,  # High-value transaction threshold
            
            # Customer value thresholds
            "vip_customer_ltv": 5000.0,  # Lifetime value for VIP treatment
            "new_customer_days": 30,  # Days to consider as new customer
            
            # Chargeback cost estimates
            "chargeback_fee": 15.0,  # Average chargeback fee
            "dispute_handling_cost": 25.0,  # Internal cost to handle dispute
            "reputation_damage_cost": 50.0,  # Estimated reputation cost
            
            # Decision thresholds
            "chargeback_probability_high": 0.70,  # High chargeback risk
            "chargeback_probability_medium": 0.40,  # Medium chargeback risk
            "fraud_probability_high": 0.65,  # High fraud risk
            
            # Customer retention
            "retention_value_multiplier": 1.5,  # Customer retention value
        }
        
        logger.info("RDR Engine initialized")
    
    def process_alert(self, alert: RDRAlert) -> RDRDecision:
        """
        Process an RDR alert and make a decision
        
        Args:
            alert: RDR alert to process
            
        Returns:
            RDR decision with recommended action
        """
        logger.info(f"Processing RDR alert: {alert.alert_id} for transaction {alert.transaction_id}")
        
        # Step 1: Get transaction details
        transaction = self._get_transaction(alert.transaction_id)
        if not transaction:
            logger.warning(f"Transaction not found: {alert.transaction_id}")
            return self._create_manual_review_decision(alert, "Transaction not found")
        
        # Step 2: Assess chargeback risk
        chargeback_risk = self._assess_chargeback_risk(alert, transaction)
        
        # Step 3: Assess fraud risk
        fraud_risk = self._assess_fraud_risk(alert, transaction)
        
        # Step 4: Calculate customer value
        customer_value = self._calculate_customer_value(alert, transaction)
        
        # Step 5: Perform cost-benefit analysis
        cost_benefit = self._analyze_cost_benefit(
            alert, transaction, chargeback_risk, fraud_risk, customer_value
        )
        
        # Step 6: Make decision
        decision = self._make_decision(
            alert, transaction, chargeback_risk, fraud_risk, customer_value, cost_benefit
        )
        
        # Step 7: Save decision
        self._save_decision(decision)
        
        logger.info(f"RDR decision made: {decision.decision_type} for alert {alert.alert_id}")
        
        return decision
    
    def _get_transaction(self, transaction_id: str) -> Optional[Dict]:
        """Get transaction from database"""
        try:
            return self.db["transactions"].find_one({"transaction_id": transaction_id})
        except Exception as e:
            logger.error(f"Error fetching transaction {transaction_id}: {e}")
            return None
    
    def _assess_chargeback_risk(self, alert: RDRAlert, transaction: Dict) -> Dict:
        """Assess chargeback risk for the alert"""
        
        # Use existing chargeback service if available
        if self.chargeback_service and alert.chargeback_score is None:
            try:
                prediction = self.chargeback_service.predict_chargeback(transaction)
                chargeback_probability = prediction.get("confidence_score", 0.5)
                chargeback_reasons = [prediction.get("chargeback_reason", "Unknown")]
            except Exception as e:
                logger.warning(f"Chargeback service failed: {e}")
                chargeback_probability = 0.5
                chargeback_reasons = ["Service unavailable"]
        else:
            chargeback_probability = alert.chargeback_score or 0.5
            chargeback_reasons = [alert.dispute_reason] if alert.dispute_reason else []
        
        # Additional risk factors
        risk_factors = []
        
        # Alert type risk
        if alert.alert_type == RDRAlertType.PRE_DISPUTE:
            chargeback_probability = max(chargeback_probability, 0.80)
            risk_factors.append("Pre-dispute alert received")
        elif alert.alert_type == RDRAlertType.FRAUD_SUSPECTED:
            chargeback_probability = max(chargeback_probability, 0.70)
            risk_factors.append("Fraud suspected")
        
        # Customer history risk
        if alert.previous_disputes > 0:
            chargeback_probability = min(1.0, chargeback_probability + (alert.previous_disputes * 0.1))
            risk_factors.append(f"Customer has {alert.previous_disputes} previous disputes")
        
        # Refund history
        if alert.refund_history_count > 3:
            chargeback_probability = min(1.0, chargeback_probability + 0.10)
            risk_factors.append(f"High refund history ({alert.refund_history_count} refunds)")
        
        return {
            "probability": chargeback_probability,
            "risk_level": self._classify_risk_level(chargeback_probability),
            "reasons": chargeback_reasons,
            "risk_factors": risk_factors
        }
    
    def _assess_fraud_risk(self, alert: RDRAlert, transaction: Dict) -> Dict:
        """Assess fraud risk for the alert"""
        
        # Use existing fraud service if available
        if self.fraud_service and alert.fraud_score is None:
            try:
                prediction = self.fraud_service.predict_fraud(transaction)
                fraud_probability = prediction.get("confidence", 0.3)
                fraud_reasons = prediction.get("reasons", [])
            except Exception as e:
                logger.warning(f"Fraud service failed: {e}")
                fraud_probability = 0.3
                fraud_reasons = []
        else:
            fraud_probability = alert.fraud_score or 0.3
            fraud_reasons = []
        
        # Alert-specific fraud indicators
        if alert.alert_type == RDRAlertType.FRAUD_SUSPECTED:
            fraud_probability = max(fraud_probability, 0.75)
            fraud_reasons.append("Fraud alert triggered")
        
        return {
            "probability": fraud_probability,
            "risk_level": self._classify_risk_level(fraud_probability),
            "reasons": fraud_reasons
        }
    
    def _calculate_customer_value(self, alert: RDRAlert, transaction: Dict) -> Dict:
        """Calculate customer lifetime value and importance"""
        
        # Get customer transaction history
        customer_txns = list(
            self.db["transactions"]
            .find({"email": alert.customer_email})
            .sort("created_at", -1)
            .limit(100)
        )
        
        # Calculate metrics
        total_spent = sum(t.get("amount", 0) for t in customer_txns)
        transaction_count = len(customer_txns)
        avg_transaction_value = total_spent / transaction_count if transaction_count > 0 else 0
        
        # Calculate lifetime value
        ltv = alert.customer_lifetime_value or total_spent
        
        # Calculate retention value
        retention_value = ltv * self.config["retention_value_multiplier"]
        
        # Determine customer segment
        is_vip = ltv >= self.config["vip_customer_ltv"]
        is_new = alert.account_age_days <= self.config["new_customer_days"]
        is_repeat = transaction_count >= 5
        
        return {
            "lifetime_value": ltv,
            "retention_value": retention_value,
            "total_spent": total_spent,
            "transaction_count": transaction_count,
            "avg_transaction_value": avg_transaction_value,
            "is_vip": is_vip,
            "is_new": is_new,
            "is_repeat": is_repeat,
            "segment": "vip" if is_vip else "repeat" if is_repeat else "new" if is_new else "regular"
        }
    
    def _analyze_cost_benefit(
        self,
        alert: RDRAlert,
        transaction: Dict,
        chargeback_risk: Dict,
        fraud_risk: Dict,
        customer_value: Dict
    ) -> Dict:
        """Perform cost-benefit analysis for refund decision"""
        
        # Cost of issuing refund
        refund_cost = alert.amount
        
        # Cost of chargeback if it occurs
        chargeback_cost = (
            alert.amount +  # Transaction amount
            self.config["chargeback_fee"] +  # Chargeback fee
            self.config["dispute_handling_cost"] +  # Handling cost
            (self.config["reputation_damage_cost"] if customer_value["is_vip"] else 0)
        )
        
        # Expected cost without refund (chargeback probability × chargeback cost)
        expected_cost_no_refund = chargeback_risk["probability"] * chargeback_cost
        
        # Expected cost with refund
        expected_cost_with_refund = refund_cost
        
        # Expected savings
        expected_savings = expected_cost_no_refund - expected_cost_with_refund
        
        # Customer retention consideration
        if customer_value["is_vip"] or customer_value["is_repeat"]:
            # Consider customer retention value
            retention_benefit = customer_value["retention_value"] * chargeback_risk["probability"] * 0.3
            expected_savings += retention_benefit
        
        # ROI calculation
        roi = (expected_savings / refund_cost) * 100 if refund_cost > 0 else 0
        
        return {
            "refund_cost": refund_cost,
            "chargeback_cost": chargeback_cost,
            "expected_cost_no_refund": expected_cost_no_refund,
            "expected_cost_with_refund": expected_cost_with_refund,
            "expected_savings": expected_savings,
            "roi_percentage": roi,
            "refund_recommended": expected_savings > 0
        }
    
    def _make_decision(
        self,
        alert: RDRAlert,
        transaction: Dict,
        chargeback_risk: Dict,
        fraud_risk: Dict,
        customer_value: Dict,
        cost_benefit: Dict
    ) -> RDRDecision:
        """Make RDR decision based on all factors"""
        
        decision_id = f"rdr_dec_{alert.alert_id}_{int(datetime.utcnow().timestamp())}"
        
        # Collect all reasoning
        reasoning = []
        risk_factors = chargeback_risk["risk_factors"] + fraud_risk["reasons"]
        recommended_actions = []
        
        # Determine decision type
        decision_type = RDRDecisionType.MANUAL_REVIEW  # Default
        refund_recommended = False
        auto_process = False
        escalate_to_human = False
        contact_customer = False
        gather_evidence = False
        
        # High fraud risk - decline automatic refund
        if fraud_risk["probability"] >= self.config["fraud_probability_high"]:
            decision_type = RDRDecisionType.DECLINE_REFUND
            reasoning.append(f"High fraud risk ({fraud_risk['probability']:.2%}) - refund declined")
            reasoning.append("Recommend gathering evidence for potential fraud case")
            gather_evidence = True
            escalate_to_human = True
            recommended_actions.extend([
                "Gather all transaction evidence",
                "Contact fraud investigation team",
                "Do not issue refund without thorough investigation"
            ])
        
        # Low-risk auto-refund scenario
        elif (
            chargeback_risk["probability"] >= self.config["chargeback_probability_high"] and
            alert.amount <= self.config["auto_refund_amount_max"] and
            cost_benefit["refund_recommended"] and
            cost_benefit["roi_percentage"] > 50 and
            fraud_risk["probability"] < 0.30
        ):
            decision_type = RDRDecisionType.AUTO_REFUND
            refund_recommended = True
            auto_process = True
            reasoning.append(f"High chargeback risk ({chargeback_risk['probability']:.2%})")
            reasoning.append(f"Positive ROI ({cost_benefit['roi_percentage']:.1f}%)")
            reasoning.append(f"Expected savings: ${cost_benefit['expected_savings']:.2f}")
            reasoning.append(f"Low fraud risk ({fraud_risk['probability']:.2%})")
            recommended_actions.extend([
                "Issue automatic refund immediately",
                "Notify customer of refund",
                "Log as chargeback prevention"
            ])
        
        # VIP customer - prioritize retention
        elif customer_value["is_vip"] and cost_benefit["refund_recommended"]:
            decision_type = RDRDecisionType.CONTACT_CUSTOMER
            refund_recommended = True
            contact_customer = True
            reasoning.append(f"VIP customer (LTV: ${customer_value['lifetime_value']:.2f})")
            reasoning.append("Prioritize customer retention")
            reasoning.append("Contact customer before processing refund")
            recommended_actions.extend([
                "Contact VIP customer within 1 hour",
                "Offer immediate resolution",
                "Consider goodwill gesture",
                "Issue refund after contact"
            ])
        
        # Medium risk - gather evidence first
        elif (
            chargeback_risk["probability"] >= self.config["chargeback_probability_medium"] and
            chargeback_risk["probability"] < self.config["chargeback_probability_high"]
        ):
            decision_type = RDRDecisionType.GATHER_EVIDENCE
            gather_evidence = True
            reasoning.append(f"Medium chargeback risk ({chargeback_risk['probability']:.2%})")
            reasoning.append("Gather evidence before deciding")
            recommended_actions.extend([
                "Collect transaction evidence (receipts, delivery confirmation)",
                "Review customer communication history",
                "Prepare dispute response package",
                "Evaluate refund option within 24 hours"
            ])
        
        # High-value transaction - manual review required
        elif alert.amount >= self.config["high_value_threshold"]:
            decision_type = RDRDecisionType.MANUAL_REVIEW
            escalate_to_human = True
            reasoning.append(f"High-value transaction (${alert.amount:.2f})")
            reasoning.append("Requires senior management approval")
            recommended_actions.extend([
                "Escalate to senior management",
                "Conduct thorough investigation",
                "Contact customer directly",
                "Review all transaction details"
            ])
        
        # New customer - gather more information
        elif customer_value["is_new"]:
            decision_type = RDRDecisionType.CONTACT_CUSTOMER
            contact_customer = True
            reasoning.append(f"New customer (account age: {alert.account_age_days} days)")
            reasoning.append("Contact customer to resolve issue")
            recommended_actions.extend([
                "Contact customer to understand issue",
                "Verify identity and transaction details",
                "Offer resolution options",
                "Build customer relationship"
            ])
        
        # Default: Manual review
        else:
            decision_type = RDRDecisionType.MANUAL_REVIEW
            escalate_to_human = True
            reasoning.append("Standard review process required")
            reasoning.append(f"Chargeback risk: {chargeback_risk['probability']:.2%}")
            reasoning.append(f"ROI: {cost_benefit['roi_percentage']:.1f}%")
            recommended_actions.extend([
                "Review transaction details",
                "Contact customer if needed",
                "Make refund decision within 24 hours"
            ])
        
        # Calculate confidence
        confidence = self._calculate_decision_confidence(
            chargeback_risk, fraud_risk, customer_value, cost_benefit
        )
        
        # Create decision
        decision = RDRDecision(
            decision_id=decision_id,
            alert_id=alert.alert_id,
            transaction_id=alert.transaction_id,
            decision_type=decision_type,
            confidence=confidence,
            reasoning=reasoning,
            risk_factors=risk_factors,
            refund_amount=alert.amount if refund_recommended else None,
            refund_recommended=refund_recommended,
            auto_process=auto_process,
            refund_cost=cost_benefit["refund_cost"],
            chargeback_cost_avoided=cost_benefit["expected_savings"] if refund_recommended else 0,
            expected_roi=cost_benefit["roi_percentage"],
            recommended_actions=recommended_actions,
            escalate_to_human=escalate_to_human,
            contact_customer=contact_customer,
            gather_evidence=gather_evidence
        )
        
        return decision
    
    def _calculate_decision_confidence(
        self,
        chargeback_risk: Dict,
        fraud_risk: Dict,
        customer_value: Dict,
        cost_benefit: Dict
    ) -> float:
        """Calculate confidence in the decision"""
        
        # Base confidence from chargeback probability
        confidence = chargeback_risk["probability"]
        
        # Adjust for fraud risk (lower confidence if high fraud)
        if fraud_risk["probability"] > 0.5:
            confidence *= 0.8
        
        # Adjust for ROI (higher confidence if good ROI)
        if cost_benefit["roi_percentage"] > 100:
            confidence = min(1.0, confidence * 1.1)
        elif cost_benefit["roi_percentage"] < 0:
            confidence *= 0.7
        
        # Adjust for customer value
        if customer_value["is_vip"]:
            confidence = min(1.0, confidence * 1.05)
        
        return min(1.0, max(0.0, confidence))
    
    def _classify_risk_level(self, probability: float) -> str:
        """Classify risk level from probability"""
        if probability >= 0.80:
            return "critical"
        elif probability >= 0.60:
            return "high"
        elif probability >= 0.35:
            return "medium"
        else:
            return "low"
    
    def _create_manual_review_decision(self, alert: RDRAlert, reason: str) -> RDRDecision:
        """Create a manual review decision"""
        return RDRDecision(
            decision_id=f"rdr_dec_{alert.alert_id}_{int(datetime.utcnow().timestamp())}",
            alert_id=alert.alert_id,
            transaction_id=alert.transaction_id,
            decision_type=RDRDecisionType.MANUAL_REVIEW,
            confidence=0.5,
            reasoning=[reason, "Manual review required"],
            risk_factors=[],
            refund_recommended=False,
            auto_process=False,
            escalate_to_human=True,
            recommended_actions=["Conduct manual review", "Investigate transaction details"]
        )
    
    def _save_decision(self, decision: RDRDecision):
        """Save decision to database"""
        try:
            self.db["rdr_decisions"].insert_one(decision.model_dump())
            logger.info(f"RDR decision saved: {decision.decision_id}")
        except Exception as e:
            logger.error(f"Error saving RDR decision: {e}")
    
    def get_metrics(self, start_date: datetime, end_date: datetime) -> Dict:
        """Get RDR metrics for a time period"""
        
        # Get all alerts in period
        alerts = list(self.db["rdr_alerts"].find({
            "created_at": {"$gte": start_date, "$lte": end_date}
        }))
        
        # Get all decisions in period
        decisions = list(self.db["rdr_decisions"].find({
            "created_at": {"$gte": start_date, "$lte": end_date}
        }))
        
        # Get completed refunds in period
        refunds = list(self.db["rdr_refunds"].find({
            "created_at": {"$gte": start_date, "$lte": end_date},
            "status": "completed"
        }))
        
        # Calculate metrics
        total_alerts = len(alerts)
        auto_refunds = len([d for d in decisions if d.get("decision_type") == "auto_refund"])
        manual_reviews = len([d for d in decisions if d.get("decision_type") == "manual_review"])
        
        total_refund_amount = sum(r.get("amount", 0) for r in refunds)
        total_cost_avoided = sum(d.get("chargeback_cost_avoided", 0) for d in decisions)
        
        # Success rate (refunds that prevented chargebacks)
        prevented_chargebacks = len([r for r in refunds if r.get("prevented_chargeback") == True])
        success_rate = prevented_chargebacks / len(refunds) if refunds else 0
        
        # Alerts by type
        alerts_by_type = {}
        for alert in alerts:
            alert_type = alert.get("alert_type", "unknown")
            alerts_by_type[alert_type] = alerts_by_type.get(alert_type, 0) + 1
        
        # Decisions by type
        decisions_by_type = {}
        for decision in decisions:
            dec_type = decision.get("decision_type", "unknown")
            decisions_by_type[dec_type] = decisions_by_type.get(dec_type, 0) + 1
        
        return {
            "total_alerts": total_alerts,
            "alerts_today": len([a for a in alerts if datetime.fromisoformat(a.get("created_at")).date() == datetime.utcnow().date()]) if alerts else 0,
            "auto_refunds_issued": auto_refunds,
            "manual_reviews": manual_reviews,
            "chargebacks_prevented": prevented_chargebacks,
            "total_refund_amount": total_refund_amount,
            "total_cost_avoided": total_cost_avoided,
            "success_rate": success_rate,
            "alerts_by_type": alerts_by_type,
            "decisions_by_type": decisions_by_type,
            "period_start": start_date,
            "period_end": end_date
        }

