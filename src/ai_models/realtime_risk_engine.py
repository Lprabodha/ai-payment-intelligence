"""
Real-time risk scoring engine that integrates ML models with rule-based systems
"""
import json
import asyncio
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import redis
from dataclasses import dataclass
import logging

@dataclass
class TransactionData:
    """Transaction data structure"""
    transaction_id: str
    email: str
    amount: float
    currency: str
    card_country: str
    billing_country: str
    ip_address: str
    fingerprint: str
    gateway: str
    timestamp: datetime
    additional_fields: Dict[str, Any] = None

@dataclass
class RiskAssessment:
    """Risk assessment result"""
    transaction_id: str
    risk_score: float
    risk_level: str
    decision: str
    confidence: float
    explanations: List[str]
    model_scores: Dict[str, float]
    rule_scores: Dict[str, float]
    processing_time_ms: float

class RealTimeRiskEngine:
    """
    Real-time risk scoring engine for fraud detection and chargeback prevention
    """
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.logger = logging.getLogger(__name__)
        
        # Risk thresholds
        self.thresholds = {
            'block': 0.85,
            'review': 0.65,
            'monitor': 0.45,
            'approve': 0.30
        }
        
        # Feature cache TTL (seconds)
        self.cache_ttl = {
            'user_velocity': 3600,      # 1 hour
            'ip_reputation': 86400,     # 24 hours
            'device_fingerprint': 3600, # 1 hour
            'geo_velocity': 1800        # 30 minutes
        }
        
    async def assess_transaction_risk(self, transaction: TransactionData) -> RiskAssessment:
        """Main entry point for real-time risk assessment"""
        
        start_time = datetime.now()
        
        try:
            # 1. Extract real-time features
            features = await self._extract_realtime_features(transaction)
            
            # 2. Get ML model predictions
            ml_scores = await self._get_ml_predictions(features)
            
            # 3. Evaluate rules
            rule_scores = await self._evaluate_rules(transaction, features)
            
            # 4. Combine scores
            final_score, explanations = self._combine_scores(ml_scores, rule_scores, features)
            
            # 5. Make decision
            decision, risk_level, confidence = self._make_decision(final_score, explanations)
            
            # 6. Update caches and logs
            await self._update_caches(transaction, features)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return RiskAssessment(
                transaction_id=transaction.transaction_id,
                risk_score=final_score,
                risk_level=risk_level,
                decision=decision,
                confidence=confidence,
                explanations=explanations,
                model_scores=ml_scores,
                rule_scores=rule_scores,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Risk assessment failed: {e}")
            # Return conservative assessment on error
            return RiskAssessment(
                transaction_id=transaction.transaction_id,
                risk_score=0.5,
                risk_level="MEDIUM",
                decision="REVIEW",
                confidence=0.0,
                explanations=[f"Assessment error: {str(e)}"],
                model_scores={},
                rule_scores={},
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
    
    async def _extract_realtime_features(self, transaction: TransactionData) -> Dict[str, float]:
        """Extract real-time features for risk assessment"""
        
        features = {}
        
        # Basic transaction features
        features['amount'] = transaction.amount
        features['amount_log'] = float(np.log1p(transaction.amount))
        features['hour'] = transaction.timestamp.hour
        features['is_weekend'] = float(transaction.timestamp.weekday() >= 5)
        features['country_mismatch'] = float(transaction.card_country != transaction.billing_country)
        
        # User velocity features
        user_velocity = await self._get_user_velocity(transaction.email)
        features.update(user_velocity)
        
        # IP reputation features
        ip_features = await self._get_ip_features(transaction.ip_address)
        features.update(ip_features)
        
        # Device fingerprint features
        device_features = await self._get_device_features(transaction.fingerprint)
        features.update(device_features)
        
        # Geographic velocity features
        geo_features = await self._get_geo_velocity_features(transaction)
        features.update(geo_features)
        
        # Gateway features
        gateway_features = await self._get_gateway_features(transaction.gateway, transaction.email)
        features.update(gateway_features)
        
        return features
    
    async def _get_user_velocity(self, email: str) -> Dict[str, float]:
        """Get user transaction velocity features"""
        
        cache_key = f"user_velocity:{email}"
        cached_data = self.redis_client.get(cache_key)
        
        if cached_data:
            return json.loads(cached_data)
        
        # Calculate velocity features
        now = datetime.now()
        windows = [300, 600, 1800, 3600, 86400]  # 5m, 10m, 30m, 1h, 24h
        
        velocity_features = {}
        
        for window in windows:
            window_start = now - timedelta(seconds=window)
            
            # Count transactions in window
            tx_count_key = f"tx_count:{email}:{window}"
            tx_count = len(self.redis_client.zrangebyscore(
                f"user_transactions:{email}",
                window_start.timestamp(),
                now.timestamp()
            ))
            
            velocity_features[f'tx_count_{window}s'] = float(tx_count)
            
            # Amount velocity
            amount_key = f"amount_sum:{email}:{window}"
            cached_amount = self.redis_client.get(amount_key)
            amount_sum = float(cached_amount) if cached_amount else 0.0
            velocity_features[f'amount_sum_{window}s'] = amount_sum
        
        # Cache results
        self.redis_client.setex(cache_key, self.cache_ttl['user_velocity'], 
                               json.dumps(velocity_features))
        
        return velocity_features
    
    async def _get_ip_features(self, ip_address: str) -> Dict[str, float]:
        """Get IP-based risk features"""
        
        cache_key = f"ip_features:{ip_address}"
        cached_data = self.redis_client.get(cache_key)
        
        if cached_data:
            return json.loads(cached_data)
        
        ip_features = {
            'ip_reputation_score': 0.5,  # Default neutral score
            'ip_user_count_24h': 0.0,
            'ip_transaction_count_24h': 0.0,
            'is_tor_exit_node': 0.0,
            'is_vpn_proxy': 0.0,
            'geo_risk_score': 0.5
        }
        
        # Get IP reputation from external service (placeholder)
        ip_features['ip_reputation_score'] = await self._query_ip_reputation(ip_address)
        
        # Count unique users from this IP in last 24h
        now = datetime.now()
        day_ago = now - timedelta(hours=24)
        
        ip_users = self.redis_client.zrangebyscore(
            f"ip_users:{ip_address}",
            day_ago.timestamp(),
            now.timestamp()
        )
        ip_features['ip_user_count_24h'] = float(len(set(ip_users)))
        
        # Count transactions from this IP in last 24h
        ip_transactions = self.redis_client.zrangebyscore(
            f"ip_transactions:{ip_address}",
            day_ago.timestamp(),
            now.timestamp()
        )
        ip_features['ip_transaction_count_24h'] = float(len(ip_transactions))
        
        # Cache results
        self.redis_client.setex(cache_key, self.cache_ttl['ip_reputation'], 
                               json.dumps(ip_features))
        
        return ip_features
    
    async def _get_device_features(self, fingerprint: str) -> Dict[str, float]:
        """Get device fingerprint-based features"""
        
        cache_key = f"device_features:{fingerprint}"
        cached_data = self.redis_client.get(cache_key)
        
        if cached_data:
            return json.loads(cached_data)
        
        device_features = {
            'device_user_count_24h': 0.0,
            'device_transaction_count_24h': 0.0,
            'device_first_seen_hours': 0.0,
            'device_reputation_score': 0.5
        }
        
        now = datetime.now()
        day_ago = now - timedelta(hours=24)
        
        # Count unique users for this device in last 24h
        device_users = self.redis_client.zrangebyscore(
            f"device_users:{fingerprint}",
            day_ago.timestamp(),
            now.timestamp()
        )
        device_features['device_user_count_24h'] = float(len(set(device_users)))
        
        # Count transactions for this device in last 24h
        device_transactions = self.redis_client.zrangebyscore(
            f"device_transactions:{fingerprint}",
            day_ago.timestamp(),
            now.timestamp()
        )
        device_features['device_transaction_count_24h'] = float(len(device_transactions))
        
        # Device age (hours since first seen)
        first_seen = self.redis_client.get(f"device_first_seen:{fingerprint}")
        if first_seen:
            first_seen_dt = datetime.fromtimestamp(float(first_seen))
            device_features['device_first_seen_hours'] = (now - first_seen_dt).total_seconds() / 3600
        else:
            # First time seeing this device
            self.redis_client.set(f"device_first_seen:{fingerprint}", now.timestamp())
            device_features['device_first_seen_hours'] = 0.0
        
        # Cache results
        self.redis_client.setex(cache_key, self.cache_ttl['device_fingerprint'], 
                               json.dumps(device_features))
        
        return device_features
    
    async def _get_geo_velocity_features(self, transaction: TransactionData) -> Dict[str, float]:
        """Calculate geographic velocity features"""
        
        cache_key = f"geo_velocity:{transaction.email}"
        cached_data = self.redis_client.get(cache_key)
        
        geo_features = {
            'impossible_geo_velocity': 0.0,
            'country_switch_frequency': 0.0,
            'new_country_flag': 0.0
        }
        
        # Get last transaction location for this user
        last_location = self.redis_client.get(f"last_location:{transaction.email}")
        
        if last_location:
            last_data = json.loads(last_location)
            last_country = last_data['country']
            last_timestamp = datetime.fromisoformat(last_data['timestamp'])
            
            # Check for impossible geographic velocity
            if last_country != transaction.card_country:
                time_diff_hours = (transaction.timestamp - last_timestamp).total_seconds() / 3600
                if time_diff_hours < 2:  # Less than 2 hours between countries
                    geo_features['impossible_geo_velocity'] = 1.0
                
                geo_features['new_country_flag'] = 1.0
        
        # Update location cache
        self.redis_client.setex(
            f"last_location:{transaction.email}",
            self.cache_ttl['geo_velocity'],
            json.dumps({
                'country': transaction.card_country,
                'timestamp': transaction.timestamp.isoformat()
            })
        )
        
        return geo_features
    
    async def _get_gateway_features(self, gateway: str, email: str) -> Dict[str, float]:
        """Get gateway-specific features"""
        
        gateway_features = {
            'gateway_user_success_rate': 0.5,
            'gateway_global_success_rate': 0.5,
            'new_gateway_for_user': 0.0
        }
        
        # Check if this is a new gateway for the user
        user_gateways = self.redis_client.smembers(f"user_gateways:{email}")
        if gateway not in user_gateways:
            gateway_features['new_gateway_for_user'] = 1.0
            self.redis_client.sadd(f"user_gateways:{email}", gateway)
        
        # Get gateway success rates (would be calculated from historical data)
        gateway_features['gateway_global_success_rate'] = await self._get_gateway_success_rate(gateway)
        
        return gateway_features
    
    async def _get_ml_predictions(self, features: Dict[str, float]) -> Dict[str, float]:
        """Get predictions from ML models"""
        
        # This would call your trained ensemble models
        # For now, returning simulated scores
        
        ml_scores = {
            'fraud_model_score': min(0.9, sum(features.values()) / len(features) / 100),
            'chargeback_model_score': min(0.9, features.get('amount_log', 0) / 10),
            'ensemble_score': 0.0
        }
        
        # Ensemble score as weighted average
        ml_scores['ensemble_score'] = (
            0.6 * ml_scores['fraud_model_score'] + 
            0.4 * ml_scores['chargeback_model_score']
        )
        
        return ml_scores
    
    async def _evaluate_rules(self, transaction: TransactionData, 
                             features: Dict[str, float]) -> Dict[str, float]:
        """Evaluate rule-based risk factors"""
        
        rule_scores = {}
        
        # High velocity rules
        if features.get('tx_count_600s', 0) > 5:  # 5+ transactions in 10 minutes
            rule_scores['high_velocity_burst'] = 0.8
        
        if features.get('tx_count_3600s', 0) > 15:  # 15+ transactions in 1 hour
            rule_scores['high_velocity_sustained'] = 0.7
        
        # Amount-based rules
        if transaction.amount > 1000:
            rule_scores['high_amount_transaction'] = 0.6
        
        if features.get('amount_log', 0) > 8:  # Very high amount
            rule_scores['extreme_amount'] = 0.8
        
        # Geographic rules
        if features.get('impossible_geo_velocity', 0) > 0:
            rule_scores['impossible_geography'] = 0.9
        
        if features.get('country_mismatch', 0) > 0 and transaction.amount > 500:
            rule_scores['country_mismatch_high_amount'] = 0.7
        
        # Device/IP rules
        if features.get('device_user_count_24h', 0) > 10:
            rule_scores['device_sharing_high'] = 0.6
        
        if features.get('ip_user_count_24h', 0) > 20:
            rule_scores['ip_sharing_extreme'] = 0.8
        
        # Time-based rules
        if features.get('hour', 12) in [2, 3, 4, 5] and transaction.amount > 300:
            rule_scores['off_hours_high_amount'] = 0.5
        
        # New user/device rules
        if features.get('device_first_seen_hours', 24) < 1 and transaction.amount > 200:
            rule_scores['new_device_high_amount'] = 0.6
        
        return rule_scores
    
    def _combine_scores(self, ml_scores: Dict[str, float], 
                       rule_scores: Dict[str, float], 
                       features: Dict[str, float]) -> Tuple[float, List[str]]:
        """Combine ML and rule scores into final risk score"""
        
        # Base ML score (70% weight)
        ml_component = ml_scores.get('ensemble_score', 0.5) * 0.7
        
        # Rule component (30% weight)
        rule_component = max(rule_scores.values()) if rule_scores else 0.0
        rule_component *= 0.3
        
        # Final score
        final_score = min(1.0, ml_component + rule_component)
        
        # Generate explanations
        explanations = []
        
        # ML explanations
        if ml_scores.get('fraud_model_score', 0) > 0.5:
            explanations.append(f"Fraud model score: {ml_scores['fraud_model_score']:.3f}")
        
        if ml_scores.get('chargeback_model_score', 0) > 0.5:
            explanations.append(f"Chargeback model score: {ml_scores['chargeback_model_score']:.3f}")
        
        # Rule explanations
        for rule_name, score in rule_scores.items():
            if score > 0.3:
                explanations.append(f"Rule triggered: {rule_name} (score: {score:.3f})")
        
        # Feature-based explanations
        if features.get('tx_count_600s', 0) > 3:
            explanations.append(f"High transaction velocity: {features['tx_count_600s']} transactions in 10 minutes")
        
        if features.get('amount_log', 0) > 7:
            explanations.append(f"Large transaction amount: ${features.get('amount', 0):,.2f}")
        
        return final_score, explanations
    
    def _make_decision(self, final_score: float, 
                      explanations: List[str]) -> Tuple[str, str, float]:
        """Make final decision based on risk score"""
        
        confidence = min(1.0, abs(final_score - 0.5) * 2)  # Higher confidence for extreme scores
        
        if final_score >= self.thresholds['block']:
            return "BLOCK", "VERY HIGH", confidence
        elif final_score >= self.thresholds['review']:
            return "REVIEW", "HIGH", confidence
        elif final_score >= self.thresholds['monitor']:
            return "MONITOR", "MEDIUM", confidence
        elif final_score >= self.thresholds['approve']:
            return "APPROVE", "LOW", confidence
        else:
            return "APPROVE", "VERY LOW", confidence
    
    async def _update_caches(self, transaction: TransactionData, features: Dict[str, float]):
        """Update Redis caches with transaction data"""
        
        now = transaction.timestamp
        
        # Update user transaction history
        self.redis_client.zadd(
            f"user_transactions:{transaction.email}",
            {transaction.transaction_id: now.timestamp()}
        )
        
        # Update IP tracking
        self.redis_client.zadd(
            f"ip_users:{transaction.ip_address}",
            {transaction.email: now.timestamp()}
        )
        self.redis_client.zadd(
            f"ip_transactions:{transaction.ip_address}",
            {transaction.transaction_id: now.timestamp()}
        )
        
        # Update device tracking
        self.redis_client.zadd(
            f"device_users:{transaction.fingerprint}",
            {transaction.email: now.timestamp()}
        )
        self.redis_client.zadd(
            f"device_transactions:{transaction.fingerprint}",
            {transaction.transaction_id: now.timestamp()}
        )
        
        # Update amount tracking for velocity calculation
        for window in [300, 600, 1800, 3600, 86400]:
            amount_key = f"amount_sum:{transaction.email}:{window}"
            current_sum = float(self.redis_client.get(amount_key) or 0)
            self.redis_client.setex(
                amount_key, 
                window,
                str(current_sum + transaction.amount)
            )
    
    async def _query_ip_reputation(self, ip_address: str) -> float:
        """Query external IP reputation service"""
        # Placeholder for actual IP reputation API call
        # Would integrate with services like MaxMind, IPQualityScore, etc.
        return 0.5
    
    async def _get_gateway_success_rate(self, gateway: str) -> float:
        """Get historical success rate for gateway"""
        # Placeholder for actual gateway analytics
        gateway_rates = {
            'stripe': 0.95,
            'paypal': 0.92,
            'square': 0.88,
            'unknown': 0.70
        }
        return gateway_rates.get(gateway, 0.80)


# Usage example and integration class
class RiskScoringAPI:
    """
    API wrapper for the real-time risk scoring engine
    """
    
    def __init__(self):
        self.risk_engine = RealTimeRiskEngine()
    
    async def score_transaction(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main API endpoint for transaction risk scoring
        
        Args:
            transaction_data: Dictionary containing transaction details
            
        Returns:
            Dictionary containing risk assessment results
        """
        
        # Parse transaction data
        transaction = TransactionData(
            transaction_id=transaction_data['transaction_id'],
            email=transaction_data['email'],
            amount=float(transaction_data['amount']),
            currency=transaction_data.get('currency', 'USD'),
            card_country=transaction_data['card_country'],
            billing_country=transaction_data['billing_country'],
            ip_address=transaction_data['ip_address'],
            fingerprint=transaction_data['fingerprint'],
            gateway=transaction_data['gateway'],
            timestamp=datetime.fromisoformat(transaction_data.get('timestamp', datetime.now().isoformat())),
            additional_fields=transaction_data.get('additional_fields', {})
        )
        
        # Get risk assessment
        assessment = await self.risk_engine.assess_transaction_risk(transaction)
        
        # Convert to API response format
        response = {
            'transaction_id': assessment.transaction_id,
            'risk_score': assessment.risk_score,
            'risk_level': assessment.risk_level,
            'decision': assessment.decision,
            'confidence': assessment.confidence,
            'explanations': assessment.explanations,
            'model_scores': assessment.model_scores,
            'rule_scores': assessment.rule_scores,
            'processing_time_ms': assessment.processing_time_ms,
            'timestamp': datetime.now().isoformat()
        }
        
        return response
