"""
Real-time risk scoring engine with ML and rule-based assessments
"""
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from database.connection import db
from risk_engine.models import (
    TransactionData, RiskAssessment, MLPrediction, RuleEvaluation,
    VelocityFeatures, IPReputationFeatures, DeviceFeatures, GeographicFeatures,
    RiskLevel, DecisionAction, RiskThresholds
)
from risk_engine.features import feature_extractor
from risk_engine.cache import cache_service
from predictions.models import model_manager

logger = logging.getLogger(__name__)

class RealTimeRiskEngine:
    """Real-time risk scoring engine combining ML and rule-based assessments"""
    
    def __init__(self):
        self.thresholds = RiskThresholds()
        self.risk_rules = self._load_risk_rules()
        self.ml_model = None
        self._load_ml_model()
    
    def _load_ml_model(self):
        """Load ML model for risk prediction"""
        try:
            # Try to load fraud detection model as base for risk scoring
            self.ml_model = model_manager.get_model('fraud_detection_pipeline')
            if self.ml_model:
                logger.info("Loaded ML model for risk scoring")
            else:
                logger.warning("No ML model available for risk scoring")
        except Exception as e:
            logger.error(f"Error loading ML model: {e}")
    
    def _load_risk_rules(self) -> List[Dict[str, Any]]:
        """Load risk assessment rules"""
        return [
            {
                "name": "high_velocity_transactions",
                "description": "Too many transactions in short time",
                "condition": lambda features: features.velocity_features.transactions_last_hour > 10,
                "score": 0.7,
                "weight": 1.0
            },
            {
                "name": "high_amount_velocity",
                "description": "High transaction amounts in short time",
                "condition": lambda features: features.velocity_features.amount_last_hour > 5000,
                "score": 0.6,
                "weight": 1.0
            },
            {
                "name": "proxy_ip",
                "description": "Transaction from proxy IP",
                "condition": lambda features: features.ip_features.is_proxy,
                "score": 0.4,
                "weight": 0.8
            },
            {
                "name": "vpn_ip",
                "description": "Transaction from VPN IP",
                "condition": lambda features: features.ip_features.is_vpn,
                "score": 0.5,
                "weight": 0.8
            },
            {
                "name": "tor_ip",
                "description": "Transaction from TOR IP",
                "condition": lambda features: features.ip_features.is_tor,
                "score": 0.9,
                "weight": 1.0
            },
            {
                "name": "datacenter_ip",
                "description": "Transaction from datacenter IP",
                "condition": lambda features: features.ip_features.is_datacenter,
                "score": 0.3,
                "weight": 0.6
            },
            {
                "name": "country_mismatch",
                "description": "IP country doesn't match billing country",
                "condition": lambda features: features.ip_features.country_mismatch,
                "score": 0.4,
                "weight": 0.7
            },
            {
                "name": "bot_user_agent",
                "description": "Bot-like user agent",
                "condition": lambda features: features.device_features.is_bot,
                "score": 0.8,
                "weight": 1.0
            },
            {
                "name": "suspicious_browser",
                "description": "Suspicious browser pattern",
                "condition": lambda features: features.device_features.browser_risk > 0.7,
                "score": 0.6,
                "weight": 0.8
            },
            {
                "name": "high_risk_country",
                "description": "Transaction from high-risk country",
                "condition": lambda features: features.geo_features.country_risk_score > 0.7,
                "score": 0.5,
                "weight": 0.8
            },
            {
                "name": "unusual_transaction_amount",
                "description": "Transaction amount significantly higher than average",
                "condition": lambda features: features.velocity_features.avg_transaction_amount > 0 and 
                                            features.transaction_data.amount > features.velocity_features.avg_transaction_amount * 5,
                "score": 0.4,
                "weight": 0.6
            },
            {
                "name": "multiple_merchants_same_day",
                "description": "Multiple merchants in same day",
                "condition": lambda features: features.velocity_features.unique_merchants_last_day > 5,
                "score": 0.3,
                "weight": 0.5
            }
        ]
    
    def assess_risk(self, transaction_data: TransactionData) -> RiskAssessment:
        """Assess risk for a transaction"""
        try:
            # Check cache first
            cached_assessment = cache_service.get_risk_assessment(transaction_data.transaction_id)
            if cached_assessment:
                return RiskAssessment(**cached_assessment)
            
            # Extract features
            velocity_features = feature_extractor.extract_velocity_features(
                transaction_data.user_id or transaction_data.email, 
                transaction_data.__dict__
            )
            
            ip_features = feature_extractor.extract_ip_reputation_features(
                transaction_data.ip_address, 
                transaction_data.__dict__
            )
            
            device_features = feature_extractor.extract_device_features(
                transaction_data.device_fingerprint, 
                transaction_data.user_agent
            )
            
            geo_features = feature_extractor.extract_geographic_features(
                transaction_data.__dict__
            )
            
            # Get ML prediction
            ml_prediction = self._get_ml_prediction(transaction_data, velocity_features, ip_features, device_features, geo_features)
            
            # Evaluate rules
            rule_evaluation = self._evaluate_rules(transaction_data, velocity_features, ip_features, device_features, geo_features)
            
            # Combine scores
            overall_risk_score = self._combine_scores(ml_prediction, rule_evaluation)
            
            # Determine risk level and action
            risk_level = self._determine_risk_level(overall_risk_score)
            decision_action = self._determine_decision_action(overall_risk_score)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(ml_prediction, rule_evaluation, overall_risk_score)
            
            # Create assessment
            assessment = RiskAssessment(
                transaction_id=transaction_data.transaction_id,
                overall_risk_score=overall_risk_score,
                risk_level=risk_level,
                decision_action=decision_action,
                ml_prediction=ml_prediction,
                rule_evaluation=rule_evaluation,
                velocity_features=velocity_features,
                ip_features=ip_features,
                device_features=device_features,
                geo_features=geo_features,
                reasoning=reasoning,
                confidence=ml_prediction.confidence if ml_prediction else 0.5,
                assessment_time=datetime.utcnow()
            )
            
            # Cache the assessment
            cache_service.cache_risk_assessment(transaction_data.transaction_id, assessment.__dict__)
            
            # Store in database
            self._store_assessment(assessment)
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing risk: {e}")
            return self._create_fallback_assessment(transaction_data, str(e))
    
    def _get_ml_prediction(self, transaction_data: TransactionData, velocity_features: VelocityFeatures, 
                          ip_features: IPReputationFeatures, device_features: DeviceFeatures, 
                          geo_features: GeographicFeatures) -> Optional[MLPrediction]:
        """Get ML model prediction"""
        try:
            if not self.ml_model:
                return None
            
            # Prepare features for ML model
            features = self._prepare_ml_features(transaction_data, velocity_features, ip_features, device_features, geo_features)
            
            # Get prediction
            if hasattr(self.ml_model, 'predict_proba'):
                probabilities = self.ml_model.predict_proba([features])[0]
                fraud_probability = probabilities[1] if len(probabilities) > 1 else probabilities[0]
            else:
                prediction = self.ml_model.predict([features])[0]
                fraud_probability = float(prediction)
            
            # Calculate confidence based on feature quality
            confidence = self._calculate_ml_confidence(features)
            
            return MLPrediction(
                fraud_probability=fraud_probability,
                confidence=confidence,
                model_version="1.0",
                features_used=list(features.keys()),
                prediction_time=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error getting ML prediction: {e}")
            return None
    
    def _evaluate_rules(self, transaction_data: TransactionData, velocity_features: VelocityFeatures, 
                       ip_features: IPReputationFeatures, device_features: DeviceFeatures, 
                       geo_features: GeographicFeatures) -> RuleEvaluation:
        """Evaluate risk rules"""
        try:
            # Create feature container for rule evaluation
            class FeatureContainer:
                def __init__(self, transaction_data, velocity_features, ip_features, device_features, geo_features):
                    self.transaction_data = transaction_data
                    self.velocity_features = velocity_features
                    self.ip_features = ip_features
                    self.device_features = device_features
                    self.geo_features = geo_features
            
            features = FeatureContainer(transaction_data, velocity_features, ip_features, device_features, geo_features)
            
            triggered_rules = []
            rule_scores = {}
            total_score = 0.0
            total_weight = 0.0
            
            for rule in self.risk_rules:
                try:
                    if rule["condition"](features):
                        triggered_rules.append(rule["name"])
                        rule_scores[rule["name"]] = rule["score"]
                        total_score += rule["score"] * rule["weight"]
                        total_weight += rule["weight"]
                except Exception as e:
                    logger.warning(f"Error evaluating rule {rule['name']}: {e}")
                    continue
            
            # Normalize score
            if total_weight > 0:
                normalized_score = total_score / total_weight
            else:
                normalized_score = 0.0
            
            return RuleEvaluation(
                rules_triggered=triggered_rules,
                rule_scores=rule_scores,
                total_rule_score=normalized_score,
                evaluation_time=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error evaluating rules: {e}")
            return RuleEvaluation()
    
    def _combine_scores(self, ml_prediction: Optional[MLPrediction], rule_evaluation: RuleEvaluation) -> float:
        """Combine ML and rule scores"""
        try:
            ml_score = ml_prediction.fraud_probability if ml_prediction else 0.0
            rule_score = rule_evaluation.total_rule_score
            
            # Weighted combination (60% ML, 40% rules)
            ml_weight = 0.6
            rule_weight = 0.4
            
            combined_score = (ml_score * ml_weight) + (rule_score * rule_weight)
            
            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, combined_score))
            
        except Exception as e:
            logger.error(f"Error combining scores: {e}")
            return 0.5  # Default medium risk
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from score"""
        if risk_score >= self.thresholds.critical_threshold:
            return RiskLevel.CRITICAL
        elif risk_score >= self.thresholds.high_threshold:
            return RiskLevel.HIGH
        elif risk_score >= self.thresholds.medium_threshold:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _determine_decision_action(self, risk_score: float) -> DecisionAction:
        """Determine decision action from score"""
        if risk_score >= self.thresholds.decline_threshold:
            return DecisionAction.DECLINE
        elif risk_score >= self.thresholds.review_threshold:
            return DecisionAction.REVIEW
        elif risk_score >= self.thresholds.approve_threshold:
            return DecisionAction.CHALLENGE
        else:
            return DecisionAction.APPROVE
    
    def _generate_reasoning(self, ml_prediction: Optional[MLPrediction], rule_evaluation: RuleEvaluation, 
                           overall_score: float) -> List[str]:
        """Generate human-readable reasoning"""
        reasoning = []
        
        if ml_prediction:
            reasoning.append(f"ML model predicted {ml_prediction.fraud_probability:.2%} fraud probability")
        
        if rule_evaluation.rules_triggered:
            reasoning.append(f"Triggered {len(rule_evaluation.rules_triggered)} risk rules")
            for rule in rule_evaluation.rules_triggered[:3]:  # Show top 3 rules
                reasoning.append(f"- {rule}")
        
        if overall_score > 0.8:
            reasoning.append("High risk indicators detected")
        elif overall_score > 0.5:
            reasoning.append("Medium risk indicators detected")
        else:
            reasoning.append("Low risk indicators")
        
        return reasoning
    
    def _prepare_ml_features(self, transaction_data: TransactionData, velocity_features: VelocityFeatures, 
                            ip_features: IPReputationFeatures, device_features: DeviceFeatures, 
                            geo_features: GeographicFeatures) -> Dict[str, float]:
        """Prepare features for ML model"""
        return {
            'amount': transaction_data.amount,
            'transactions_last_hour': velocity_features.transactions_last_hour,
            'transactions_last_day': velocity_features.transactions_last_day,
            'amount_last_hour': velocity_features.amount_last_hour,
            'amount_last_day': velocity_features.amount_last_day,
            'unique_merchants_last_day': velocity_features.unique_merchants_last_day,
            'avg_transaction_amount': velocity_features.avg_transaction_amount,
            'max_transaction_amount': velocity_features.max_transaction_amount,
            'ip_risk_score': ip_features.risk_score,
            'is_proxy': float(ip_features.is_proxy),
            'is_vpn': float(ip_features.is_vpn),
            'is_tor': float(ip_features.is_tor),
            'country_mismatch': float(ip_features.country_mismatch),
            'is_mobile': float(device_features.is_mobile),
            'is_bot': float(device_features.is_bot),
            'browser_risk': device_features.browser_risk,
            'os_risk': device_features.os_risk,
            'country_risk_score': geo_features.country_risk_score,
            'timezone_mismatch': float(geo_features.timezone_mismatch)
        }
    
    def _calculate_ml_confidence(self, features: Dict[str, float]) -> float:
        """Calculate ML prediction confidence"""
        try:
            # Simple confidence calculation based on feature completeness
            non_zero_features = sum(1 for v in features.values() if v != 0)
            total_features = len(features)
            
            if total_features == 0:
                return 0.0
            
            completeness = non_zero_features / total_features
            return min(1.0, completeness + 0.3)  # Base confidence of 0.3
            
        except Exception as e:
            logger.error(f"Error calculating ML confidence: {e}")
            return 0.5
    
    def _store_assessment(self, assessment: RiskAssessment):
        """Store risk assessment in database"""
        try:
            db["risk_assessments"].insert_one({
                "transaction_id": assessment.transaction_id,
                "overall_risk_score": assessment.overall_risk_score,
                "risk_level": assessment.risk_level.value,
                "decision_action": assessment.decision_action.value,
                "ml_prediction": assessment.ml_prediction.__dict__ if assessment.ml_prediction else None,
                "rule_evaluation": assessment.rule_evaluation.__dict__ if assessment.rule_evaluation else None,
                "reasoning": assessment.reasoning,
                "confidence": assessment.confidence,
                "assessment_time": assessment.assessment_time,
                "created_at": datetime.utcnow()
            })
            
        except Exception as e:
            logger.error(f"Error storing risk assessment: {e}")
    
    def _create_fallback_assessment(self, transaction_data: TransactionData, error: str) -> RiskAssessment:
        """Create fallback assessment when main assessment fails"""
        return RiskAssessment(
            transaction_id=transaction_data.transaction_id,
            overall_risk_score=0.5,
            risk_level=RiskLevel.MEDIUM,
            decision_action=DecisionAction.REVIEW,
            reasoning=[f"Assessment error: {error}", "Using fallback assessment"],
            confidence=0.1,
            assessment_time=datetime.utcnow()
        )

# Global risk engine instance
risk_engine = RealTimeRiskEngine()
