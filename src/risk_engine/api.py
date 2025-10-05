"""
API wrapper for real-time risk scoring engine
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from fastapi import HTTPException
from risk_engine.engine import risk_engine
from risk_engine.models import TransactionData, RiskAssessment, RiskLevel, DecisionAction
from risk_engine.cache import cache_service

logger = logging.getLogger(__name__)

class RiskScoringAPI:
    """API wrapper for easy integration and transaction scoring"""
    
    def __init__(self):
        self.engine = risk_engine
        self.cache = cache_service
    
    def score_transaction(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Score a transaction for risk"""
        try:
            # Convert dict to TransactionData
            transaction = self._dict_to_transaction_data(transaction_data)
            
            # Assess risk
            assessment = self.engine.assess_risk(transaction)
            
            # Convert to API response format
            return self._assessment_to_api_response(assessment)
            
        except Exception as e:
            logger.error(f"Error scoring transaction: {e}")
            raise HTTPException(status_code=500, detail=f"Risk scoring failed: {str(e)}")
    
    def score_transaction_batch(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score multiple transactions for risk"""
        try:
            results = []
            
            for transaction_data in transactions:
                try:
                    result = self.score_transaction(transaction_data)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error scoring transaction {transaction_data.get('transaction_id', 'unknown')}: {e}")
                    results.append({
                        "transaction_id": transaction_data.get('transaction_id', 'unknown'),
                        "error": str(e),
                        "overall_risk_score": 0.5,
                        "risk_level": "medium",
                        "decision_action": "review"
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error scoring transaction batch: {e}")
            raise HTTPException(status_code=500, detail=f"Batch risk scoring failed: {str(e)}")
    
    def get_risk_assessment(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Get cached risk assessment for a transaction"""
        try:
            # Check cache first
            cached_assessment = self.cache.get_risk_assessment(transaction_id)
            if cached_assessment:
                return self._assessment_to_api_response(RiskAssessment(**cached_assessment))
            
            # Check database
            from database.connection import db
            assessment_doc = db["risk_assessments"].find_one({"transaction_id": transaction_id})
            
            if assessment_doc:
                return self._db_assessment_to_api_response(assessment_doc)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting risk assessment: {e}")
            return None
    
    def get_user_risk_profile(self, user_id: str) -> Dict[str, Any]:
        """Get risk profile for a user"""
        try:
            # Get cached velocity data
            velocity_data = self.cache.get_user_velocity(user_id)
            
            # Get recent risk assessments
            from database.connection import db
            recent_assessments = list(db["risk_assessments"].find({
                "transaction_id": {"$regex": f".*{user_id}.*"}
            }).sort("assessment_time", -1).limit(10))
            
            # Calculate risk metrics
            risk_scores = [a["overall_risk_score"] for a in recent_assessments]
            avg_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0.0
            
            high_risk_count = sum(1 for score in risk_scores if score > 0.7)
            
            return {
                "user_id": user_id,
                "average_risk_score": avg_risk_score,
                "high_risk_transactions": high_risk_count,
                "total_assessments": len(recent_assessments),
                "velocity_data": velocity_data,
                "risk_trend": self._calculate_risk_trend(risk_scores),
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting user risk profile: {e}")
            return {
                "user_id": user_id,
                "error": str(e),
                "average_risk_score": 0.5,
                "high_risk_transactions": 0,
                "total_assessments": 0
            }
    
    def update_risk_thresholds(self, thresholds: Dict[str, float]) -> Dict[str, Any]:
        """Update risk scoring thresholds"""
        try:
            # Update thresholds in the engine
            if "low_threshold" in thresholds:
                self.engine.thresholds.low_threshold = thresholds["low_threshold"]
            if "medium_threshold" in thresholds:
                self.engine.thresholds.medium_threshold = thresholds["medium_threshold"]
            if "high_threshold" in thresholds:
                self.engine.thresholds.high_threshold = thresholds["high_threshold"]
            if "critical_threshold" in thresholds:
                self.engine.thresholds.critical_threshold = thresholds["critical_threshold"]
            if "approve_threshold" in thresholds:
                self.engine.thresholds.approve_threshold = thresholds["approve_threshold"]
            if "review_threshold" in thresholds:
                self.engine.thresholds.review_threshold = thresholds["review_threshold"]
            if "decline_threshold" in thresholds:
                self.engine.thresholds.decline_threshold = thresholds["decline_threshold"]
            
            return {
                "status": "success",
                "message": "Risk thresholds updated successfully",
                "thresholds": {
                    "low_threshold": self.engine.thresholds.low_threshold,
                    "medium_threshold": self.engine.thresholds.medium_threshold,
                    "high_threshold": self.engine.thresholds.high_threshold,
                    "critical_threshold": self.engine.thresholds.critical_threshold,
                    "approve_threshold": self.engine.thresholds.approve_threshold,
                    "review_threshold": self.engine.thresholds.review_threshold,
                    "decline_threshold": self.engine.thresholds.decline_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"Error updating risk thresholds: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to update thresholds: {str(e)}")
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get risk scoring metrics and statistics"""
        try:
            from database.connection import db
            
            # Get assessment statistics
            total_assessments = db["risk_assessments"].count_documents({})
            
            # Risk level distribution
            risk_levels = db["risk_assessments"].aggregate([
                {"$group": {"_id": "$risk_level", "count": {"$sum": 1}}}
            ])
            risk_level_distribution = {item["_id"]: item["count"] for item in risk_levels}
            
            # Decision action distribution
            decision_actions = db["risk_assessments"].aggregate([
                {"$group": {"_id": "$decision_action", "count": {"$sum": 1}}}
            ])
            decision_action_distribution = {item["_id"]: item["count"] for item in decision_actions}
            
            # Average risk scores by time
            recent_assessments = list(db["risk_assessments"].find({
                "assessment_time": {"$gte": datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)}
            }))
            
            avg_risk_score_today = sum(a["overall_risk_score"] for a in recent_assessments) / len(recent_assessments) if recent_assessments else 0.0
            
            # Cache statistics
            cache_stats = self.cache.get_cache_stats()
            
            return {
                "total_assessments": total_assessments,
                "risk_level_distribution": risk_level_distribution,
                "decision_action_distribution": decision_action_distribution,
                "average_risk_score_today": avg_risk_score_today,
                "assessments_today": len(recent_assessments),
                "cache_stats": cache_stats,
                "thresholds": {
                    "low_threshold": self.engine.thresholds.low_threshold,
                    "medium_threshold": self.engine.thresholds.medium_threshold,
                    "high_threshold": self.engine.thresholds.high_threshold,
                    "critical_threshold": self.engine.thresholds.critical_threshold
                },
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting risk metrics: {e}")
            return {
                "error": str(e),
                "total_assessments": 0,
                "risk_level_distribution": {},
                "decision_action_distribution": {},
                "average_risk_score_today": 0.0,
                "assessments_today": 0
            }
    
    def _dict_to_transaction_data(self, data: Dict[str, Any]) -> TransactionData:
        """Convert dictionary to TransactionData object"""
        return TransactionData(
            transaction_id=data.get("transaction_id", ""),
            user_id=data.get("user_id"),
            email=data.get("email"),
            amount=float(data.get("amount", 0)),
            currency=data.get("currency", "USD"),
            payment_method=data.get("payment_method", "card"),
            card_brand=data.get("card_brand"),
            card_country=data.get("card_country"),
            ip_address=data.get("ip_address"),
            device_fingerprint=data.get("device_fingerprint"),
            user_agent=data.get("user_agent"),
            billing_country=data.get("billing_country"),
            shipping_country=data.get("shipping_country"),
            merchant_id=data.get("merchant_id"),
            product_category=data.get("product_category"),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat())),
            metadata=data.get("metadata", {})
        )
    
    def _assessment_to_api_response(self, assessment: RiskAssessment) -> Dict[str, Any]:
        """Convert RiskAssessment to API response format"""
        return {
            "transaction_id": assessment.transaction_id,
            "overall_risk_score": assessment.overall_risk_score,
            "risk_level": assessment.risk_level.value,
            "decision_action": assessment.decision_action.value,
            "confidence": assessment.confidence,
            "reasoning": assessment.reasoning,
            "ml_prediction": {
                "fraud_probability": assessment.ml_prediction.fraud_probability if assessment.ml_prediction else None,
                "confidence": assessment.ml_prediction.confidence if assessment.ml_prediction else None,
                "model_version": assessment.ml_prediction.model_version if assessment.ml_prediction else None
            } if assessment.ml_prediction else None,
            "rule_evaluation": {
                "rules_triggered": assessment.rule_evaluation.rules_triggered if assessment.rule_evaluation else [],
                "total_rule_score": assessment.rule_evaluation.total_rule_score if assessment.rule_evaluation else 0.0
            } if assessment.rule_evaluation else None,
            "velocity_features": assessment.velocity_features.__dict__ if assessment.velocity_features else None,
            "ip_features": assessment.ip_features.__dict__ if assessment.ip_features else None,
            "device_features": assessment.device_features.__dict__ if assessment.device_features else None,
            "geo_features": assessment.geo_features.__dict__ if assessment.geo_features else None,
            "assessment_time": assessment.assessment_time.isoformat(),
            "metadata": assessment.metadata
        }
    
    def _db_assessment_to_api_response(self, assessment_doc: Dict[str, Any]) -> Dict[str, Any]:
        """Convert database assessment document to API response format"""
        return {
            "transaction_id": assessment_doc["transaction_id"],
            "overall_risk_score": assessment_doc["overall_risk_score"],
            "risk_level": assessment_doc["risk_level"],
            "decision_action": assessment_doc["decision_action"],
            "confidence": assessment_doc.get("confidence", 0.0),
            "reasoning": assessment_doc.get("reasoning", []),
            "ml_prediction": assessment_doc.get("ml_prediction"),
            "rule_evaluation": assessment_doc.get("rule_evaluation"),
            "assessment_time": assessment_doc["assessment_time"].isoformat() if isinstance(assessment_doc["assessment_time"], datetime) else assessment_doc["assessment_time"]
        }
    
    def _calculate_risk_trend(self, risk_scores: List[float]) -> str:
        """Calculate risk trend from recent scores"""
        if len(risk_scores) < 2:
            return "insufficient_data"
        
        recent_scores = risk_scores[:5]  # Last 5 assessments
        older_scores = risk_scores[5:10] if len(risk_scores) > 5 else []
        
        if not older_scores:
            return "insufficient_data"
        
        recent_avg = sum(recent_scores) / len(recent_scores)
        older_avg = sum(older_scores) / len(older_scores)
        
        if recent_avg > older_avg + 0.1:
            return "increasing"
        elif recent_avg < older_avg - 0.1:
            return "decreasing"
        else:
            return "stable"

# Global API instance
risk_scoring_api = RiskScoringAPI()
