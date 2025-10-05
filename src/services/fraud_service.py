"""
Fraud Detection Service
Handles fraud detection logic and model management
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pymongo import MongoClient

from config.settings import settings, model_config, feature_config
from utils.logger import get_logger

logger = get_logger(__name__)


class FraudDetectionService:
    """Service for fraud detection operations"""
    
    def __init__(self, db: MongoClient):
        self.db = db
        self.model_path = settings.model_path
        self.fraud_model = None
        self.fraud_features = []
        self.model_type = None
        
        # Load model and configuration
        self._load_model()
    
    def _load_model(self):
        """Load fraud detection model and configuration"""
        try:
            # Try to load enhanced model first
            if settings.use_enhanced_models:
                enhanced_model_path = os.path.join(
                    self.model_path, 
                    model_config.FRAUD_MODELS['enhanced']['ensemble']
                )
                
                if os.path.exists(enhanced_model_path):
                    self.fraud_model = joblib.load(enhanced_model_path)
                    self.model_type = 'enhanced_ensemble'
                    logger.info("Loaded enhanced fraud detection ensemble")
                    
                    # Load feature list from metadata
                    metadata_path = os.path.join(self.model_path, "enhanced_fraud_metadata.json")
                    if os.path.exists(metadata_path):
                        import json
                        with open(metadata_path) as f:
                            metadata = json.load(f)
                            self.fraud_features = metadata.get('features_used', [])
                    return
            
            # Fall back to legacy model
            legacy_pipeline_path = os.path.join(
                self.model_path, 
                model_config.FRAUD_MODELS['legacy']['pipeline']
            )
            
            if os.path.exists(legacy_pipeline_path):
                self.fraud_model = joblib.load(legacy_pipeline_path)
                self.model_type = 'legacy_pipeline'
                logger.info("Loaded legacy fraud detection pipeline")
                
                # Load feature list from metadata
                metadata_path = os.path.join(self.model_path, "fraud_detection_metadata.json")
                if os.path.exists(metadata_path):
                    import json
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                        self.fraud_features = metadata.get('features_used', [])
                return
            
            logger.warning("No fraud detection models found")
            
        except Exception as e:
            logger.error(f"Error loading fraud model: {e}")
            raise
    
    def predict_fraud(self, transaction_data: Dict) -> Dict:
        """
        Predict fraud for a transaction
        
        Args:
            transaction_data: Transaction data dictionary
            
        Returns:
            Dictionary containing fraud prediction results
        """
        try:
            if not self.fraud_model:
                raise ValueError("Fraud model not loaded")
            
            # Create feature vector
            features = self._create_feature_vector(transaction_data)
            
            # Make prediction
            if self.model_type == 'enhanced_ensemble':
                return self._predict_with_enhanced_model(features, transaction_data)
            else:
                return self._predict_with_legacy_model(features, transaction_data)
                
        except Exception as e:
            logger.error(f"Fraud prediction error: {e}")
            raise
    
    def _create_feature_vector(self, transaction_data: Dict) -> List[float]:
        """Create feature vector for fraud detection"""
        
        # Extract basic transaction info
        amount = float(transaction_data.get('amount', 0))
        email = transaction_data.get('email', 'unknown@example.com')
        card_country = transaction_data.get('card_country', 'UNK')
        billing_country = transaction_data.get('billing_address_country', 'UNK')
        risk_score = float(transaction_data.get('risk_score', 0))
        ip_address = transaction_data.get('ip_address', '0.0.0.0')
        fingerprint = transaction_data.get('fingerprint', 'unknown')
        hour = transaction_data.get('hour', datetime.utcnow().hour)
        
        # Get customer history
        now = datetime.utcnow()
        customer_txns = list(self.db["transactions"].find({"email": email}).sort("created_at", -1).limit(100))
        
        # Initialize feature dictionary
        features = {}
        
        # Basic features
        features['amount_log'] = np.log1p(amount)
        features['hour'] = hour
        features['is_weekend'] = int(now.weekday() >= 5)
        features['risk_score'] = risk_score
        features['country_mismatch'] = int(card_country != billing_country)
        features['email_domain_risk'] = self._get_email_domain_risk(email)
        
        # IP and fingerprint reuse
        features['ip_address_reuse_count'] = 0
        features['fingerprint_reuse_count'] = 0
        features['device_ip_pair_reuse_count'] = 0
        
        if customer_txns:
            ip_count = len([t for t in customer_txns if t.get('ip_address') == ip_address])
            fp_count = len([t for t in customer_txns if t.get('fingerprint') == fingerprint])
            pair_count = len([t for t in customer_txns if t.get('ip_address') == ip_address and t.get('fingerprint') == fingerprint])
            
            features['ip_address_reuse_count'] = ip_count
            features['fingerprint_reuse_count'] = fp_count
            features['device_ip_pair_reuse_count'] = pair_count
        
        # Customer behavior features
        features['email_transaction_count'] = len(customer_txns)
        features['customer_refund_ratio'] = 0.0
        features['time_between_transactions'] = 999999.0
        features['transaction_amount_diff'] = 0.0
        features['past_chargebacks'] = 0
        features['unusual_amount_flag'] = 0
        features['shared_card_email_count'] = 0
        features['shared_ip_email_count'] = 0
        features['previous_risk_scores_avg'] = 0.0
        features['number_of_risky_transactions'] = 0
        
        if customer_txns:
            # Calculate historical features
            amounts = [t.get('amount', 0) for t in customer_txns]
            risk_scores = [t.get('risk_score', 0) for t in customer_txns]
            refunded_count = sum(1 for t in customer_txns if t.get('refunded', False))
            disputed_count = sum(1 for t in customer_txns if t.get('disputed', False))
            
            features['customer_refund_ratio'] = refunded_count / len(customer_txns)
            features['past_chargebacks'] = disputed_count
            features['previous_risk_scores_avg'] = np.mean(risk_scores) if risk_scores else 0
            features['number_of_risky_transactions'] = sum(1 for r in risk_scores if r > 70)
            
            if amounts:
                avg_amount = np.mean(amounts)
                features['transaction_amount_diff'] = abs(amount - avg_amount)
                
                # Check for unusual amount
                q99 = np.percentile(amounts, 99)
                features['unusual_amount_flag'] = int(amount > q99)
        
        # Shared device/IP patterns
        all_ip_txns = list(self.db["transactions"].find({"ip_address": ip_address}))
        all_fp_txns = list(self.db["transactions"].find({"fingerprint": fingerprint}))
        
        features['shared_ip_email_count'] = len(set(t.get('email') for t in all_ip_txns))
        features['shared_card_email_count'] = len(set(t.get('email') for t in all_fp_txns))
        
        # Convert to list in correct order
        if self.fraud_features:
            return [features.get(feature, 0) for feature in self.fraud_features]
        else:
            # Fallback to default order
            default_features = [
                'amount_log', 'hour', 'is_weekend', 'ip_address_reuse_count', 
                'fingerprint_reuse_count', 'device_ip_pair_reuse_count', 'risk_score',
                'email_transaction_count', 'customer_refund_ratio', 'country_mismatch',
                'ip_country_mismatch', 'time_between_transactions', 'email_domain_risk',
                'transaction_amount_diff', 'past_chargebacks', 'unusual_amount_flag',
                'shared_card_email_count', 'shared_ip_email_count', 'previous_risk_scores_avg',
                'number_of_risky_transactions'
            ]
            return [features.get(feature, 0) for feature in default_features]
    
    def _predict_with_enhanced_model(self, features: List[float], transaction_data: Dict) -> Dict:
        """Make prediction using enhanced ensemble model"""
        
        # Make prediction
        proba = self.fraud_model.predict_proba([features])[0][1]
        pred = self.fraud_model.predict([features])[0]
        
        # Generate reasons
        reasons = self._generate_fraud_reasons(features, transaction_data)
        
        # Apply Stripe Radar overrides
        risk_score = transaction_data.get('risk_score', 0)
        override_applied = False
        
        if risk_score >= settings.radar_high_override:
            pred = 1
            proba = max(proba, 0.90)
            reasons.append("Stripe risk score ≥ 65 (auto-flagged high risk)")
            override_applied = True
        elif risk_score >= settings.radar_medium_hint:
            proba = max(proba, 0.50)
            reasons.append("Stripe risk score ≥ 55 (confidence boosted)")
        
        return {
            "fraud_detected": bool(pred),
            "confidence": round(proba, 4),
            "reasons": reasons,
            "override_applied": override_applied,
            "features_used": features,
            "model_type": "enhanced_ensemble"
        }
    
    def _predict_with_legacy_model(self, features: List[float], transaction_data: Dict) -> Dict:
        """Make prediction using legacy model"""
        
        # Convert features to DataFrame
        X = pd.DataFrame([features], columns=self.fraud_features if self.fraud_features else range(len(features)))
        X = X.replace([np.inf, -np.inf], 0).fillna(0)
        
        # Make prediction
        if hasattr(self.fraud_model, 'predict_proba'):
            proba = float(self.fraud_model.predict_proba(X)[0][1])
            pred = int(self.fraud_model.predict(X)[0])
        else:
            # Handle pipeline with scaler
            proba = float(self.fraud_model.predict_proba(X)[0][1])
            pred = int(self.fraud_model.predict(X)[0])
        
        # Generate reasons
        reasons = self._generate_fraud_reasons(features, transaction_data)
        
        # Apply Stripe Radar overrides
        risk_score = transaction_data.get('risk_score', 0)
        override_applied = False
        
        if risk_score >= settings.radar_high_override:
            pred = 1
            proba = max(proba, 0.90)
            reasons.append("Stripe risk score ≥ 65 (auto-flagged high risk)")
            override_applied = True
        elif risk_score >= settings.radar_medium_hint:
            proba = max(proba, 0.50)
            reasons.append("Stripe risk score ≥ 55 (confidence boosted)")
        
        return {
            "fraud_detected": bool(pred),
            "confidence": round(proba, 4),
            "reasons": reasons,
            "override_applied": override_applied,
            "features_used": {self.fraud_features[i]: float(features[i]) for i in range(len(features))},
            "model_type": "legacy"
        }
    
    def _generate_fraud_reasons(self, features: List[float], transaction_data: Dict) -> List[str]:
        """Generate human-readable reasons for fraud detection"""
        reasons = []
        
        # Feature-based reasoning
        feature_names = self.fraud_features if self.fraud_features else [
            'amount_log', 'hour', 'is_weekend', 'ip_address_reuse_count', 
            'fingerprint_reuse_count', 'device_ip_pair_reuse_count', 'risk_score',
            'email_transaction_count', 'customer_refund_ratio', 'country_mismatch',
            'ip_country_mismatch', 'time_between_transactions', 'email_domain_risk',
            'transaction_amount_diff', 'past_chargebacks', 'unusual_amount_flag',
            'shared_card_email_count', 'shared_ip_email_count', 'previous_risk_scores_avg',
            'number_of_risky_transactions'
        ]
        
        feature_values = dict(zip(feature_names, features))
        
        # Check various risk indicators
        if feature_values.get('unusual_amount_flag', 0) == 1:
            reasons.append("Transaction amount is unusually high for this customer")
        
        if feature_values.get('country_mismatch', 0) == 1:
            reasons.append("Card country and billing country do not match")
        
        if feature_values.get('ip_address_reuse_count', 0) > 5:
            reasons.append("IP address has been used in many previous transactions")
        
        if feature_values.get('fingerprint_reuse_count', 0) > 3:
            reasons.append("Device fingerprint has been reused multiple times")
        
        if feature_values.get('customer_refund_ratio', 0.0) > 0.3:
            reasons.append("Customer has a high historical refund ratio")
        
        if feature_values.get('past_chargebacks', 0) > 0:
            reasons.append("Previous chargebacks found for this customer")
        
        if feature_values.get('shared_ip_email_count', 0) > 2:
            reasons.append("IP address is associated with multiple emails")
        
        if feature_values.get('shared_card_email_count', 0) > 2:
            reasons.append("Device fingerprint is associated with multiple emails")
        
        if feature_values.get('previous_risk_scores_avg', 0.0) > 70:
            reasons.append("Customer has high average risk scores")
        
        return reasons
    
    def _get_email_domain_risk(self, email: str) -> int:
        """Calculate email domain risk score"""
        if '@' not in email:
            return 1
        
        domain = email.split('@')[-1].lower()
        
        if domain in feature_config.COMMON_EMAIL_DOMAINS:
            return 0
        elif any(hint in domain for hint in feature_config.DISPOSABLE_EMAIL_HINTS):
            return 2
        else:
            return 1
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            "model_type": self.model_type,
            "features_count": len(self.fraud_features),
            "model_path": self.model_path,
            "features": self.fraud_features[:10] if self.fraud_features else []  # Show first 10 features
        }
