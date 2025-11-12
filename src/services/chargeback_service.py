"""
Chargeback Prediction Service
Handles chargeback prediction logic and model management
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pymongo import MongoClient

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class ChargebackPredictionService:
    """Service for chargeback prediction operations"""
    
    def __init__(self, db: MongoClient):
        self.db = db
        self.model_path = settings.MODEL_PATH
        self.chargeback_model = None
        self.chargeback_features = []
        self.model_type = None
        
        # Load model and configuration
        self._load_model()
    
    def _load_model(self):
        """Load chargeback prediction model and configuration"""
        try:
            # Load chargeback prediction pipeline
            pipeline_path = os.path.join(self.model_path, "chargeback_pipeline.pkl")
            
            if os.path.exists(pipeline_path):
                self.chargeback_model = joblib.load(pipeline_path)
                self.model_type = 'legacy_pipeline'
                logger.info("Loaded chargeback prediction pipeline")
                
                # Load feature list from metadata
                metadata_path = os.path.join(self.model_path, "chargeback_metadata.json")
                if os.path.exists(metadata_path):
                    import json
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                        self.chargeback_features = metadata.get('features_used', [])
                return
            
            logger.warning("No chargeback prediction models found")
            
        except Exception as e:
            logger.error(f"Error loading chargeback model: {e}")
            raise
    
    def predict_chargeback(self, transaction_data: Dict) -> Dict:
        """
        Predict chargeback for a transaction
        
        Args:
            transaction_data: Transaction data dictionary
            
        Returns:
            Dictionary containing chargeback prediction results
        """
        try:
            if not self.chargeback_model:
                raise ValueError("Chargeback model not loaded")
            
            # Create feature vector
            features = self._create_feature_vector(transaction_data)
            
            # Make prediction
            if self.model_type == 'enhanced_ensemble':
                return self._predict_with_enhanced_model(features, transaction_data)
            else:
                return self._predict_with_legacy_model(features, transaction_data)
                
        except Exception as e:
            logger.error(f"Chargeback prediction error: {e}")
            raise
    
    def _create_feature_vector(self, transaction_data: Dict) -> List[float]:
        """Create feature vector for chargeback prediction"""
        
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
        features['ip_address_reuse_before'] = 0
        features['fingerprint_reuse_before'] = 0
        features['device_ip_pair_reuse_before'] = 0
        
        if customer_txns:
            ip_count = len([t for t in customer_txns if t.get('ip_address') == ip_address])
            fp_count = len([t for t in customer_txns if t.get('fingerprint') == fingerprint])
            pair_count = len([t for t in customer_txns if t.get('ip_address') == ip_address and t.get('fingerprint') == fingerprint])
            
            features['ip_address_reuse_before'] = ip_count
            features['fingerprint_reuse_before'] = fp_count
            features['device_ip_pair_reuse_before'] = pair_count
        
        # Customer behavior features
        features['email_transaction_count'] = len(customer_txns)
        features['customer_refund_ratio_past'] = 0.0
        features['time_between_transactions'] = 999999.0
        features['transaction_amount_diff'] = 0.0
        features['past_chargebacks'] = 0
        features['past_avg_amount'] = 0.0
        
        if customer_txns:
            # Calculate historical features
            amounts = [t.get('amount', 0) for t in customer_txns]
            refunded_count = sum(1 for t in customer_txns if t.get('refunded', False))
            disputed_count = sum(1 for t in customer_txns if t.get('disputed', False))
            
            features['customer_refund_ratio_past'] = refunded_count / len(customer_txns)
            features['past_chargebacks'] = disputed_count
            
            if amounts:
                avg_amount = np.mean(amounts)
                features['past_avg_amount'] = avg_amount
                features['transaction_amount_diff'] = abs(amount - avg_amount)
        
        # Convert to list in correct order
        if self.chargeback_features:
            return [features.get(feature, 0) for feature in self.chargeback_features]
        else:
            # Fallback to default order
            default_features = [
                'amount_log', 'hour', 'is_weekend', 'ip_address_reuse_before',
                'fingerprint_reuse_before', 'device_ip_pair_reuse_before', 'risk_score',
                'email_transaction_count', 'customer_refund_ratio_past', 'country_mismatch',
                'ip_country_mismatch', 'time_between_transactions', 'email_domain_risk',
                'transaction_amount_diff', 'past_chargebacks', 'past_avg_amount'
            ]
            return [features.get(feature, 0) for feature in default_features]
    
    def _predict_with_enhanced_model(self, features: List[float], transaction_data: Dict) -> Dict:
        """Make prediction using enhanced ensemble model"""
        
        # Make prediction
        proba = self.chargeback_model.predict_proba([features])[0][1]
        pred = self.chargeback_model.predict([features])[0]
        
        # Generate reasons
        reasons = self._generate_chargeback_reasons(features, transaction_data)
        
        return {
            "chargeback_predicted": bool(pred),
            "confidence_score": round(proba, 4),
            "chargeback_reason": ", ".join(reasons) if reasons else "No strong indicators",
            "model_type": "enhanced_ensemble"
        }
    
    def _predict_with_legacy_model(self, features: List[float], transaction_data: Dict) -> Dict:
        """Make prediction using legacy model"""
        
        # Convert features to DataFrame
        X = pd.DataFrame([features], columns=self.chargeback_features if self.chargeback_features else range(len(features)))
        X = X.replace([np.inf, -np.inf], 0).fillna(0)
        
        # Make prediction
        if hasattr(self.chargeback_model, 'predict_proba'):
            proba = float(self.chargeback_model.predict_proba(X)[0][1])
            pred = int(self.chargeback_model.predict(X)[0])
        else:
            # Handle pipeline with scaler
            proba = float(self.chargeback_model.predict_proba(X)[0][1])
            pred = int(self.chargeback_model.predict(X)[0])
        
        # Generate reasons
        reasons = self._generate_chargeback_reasons(features, transaction_data)
        
        return {
            "chargeback_predicted": bool(pred),
            "confidence_score": round(proba, 4),
            "chargeback_reason": ", ".join(reasons) if reasons else "No strong indicators",
            "model_type": "legacy"
        }
    
    def _generate_chargeback_reasons(self, features: List[float], transaction_data: Dict) -> List[str]:
        """Generate human-readable reasons for chargeback prediction"""
        reasons = []
        
        # Feature-based reasoning
        feature_names = self.chargeback_features if self.chargeback_features else [
            'amount_log', 'hour', 'is_weekend', 'ip_address_reuse_before',
            'fingerprint_reuse_before', 'device_ip_pair_reuse_before', 'risk_score',
            'email_transaction_count', 'customer_refund_ratio_past', 'country_mismatch',
            'ip_country_mismatch', 'time_between_transactions', 'email_domain_risk',
            'transaction_amount_diff', 'past_chargebacks', 'past_avg_amount'
        ]
        
        feature_values = dict(zip(feature_names, features))
        
        # Check various chargeback indicators
        if feature_values.get('customer_refund_ratio_past', 0) > 0.5:
            reasons.append("Customer has a high historical refund ratio (>50%)")
        
        if feature_values.get('device_ip_pair_reuse_before', 0) > 3:
            reasons.append("Device/IP pair has been reused in multiple transactions")
        
        if feature_values.get('country_mismatch', 0) == 1:
            reasons.append("Card country and billing country do not match")
        
        if feature_values.get('email_domain_risk', 0) == 1:
            reasons.append("Email domain is uncommon or potentially risky")
        
        if feature_values.get('transaction_amount_diff', 0) > 100:
            reasons.append("Transaction amount is unusually different from customer's average")
        
        if feature_values.get('email_transaction_count', 0) > 10:
            reasons.append("Email has an unusually high number of transactions")
        
        if feature_values.get('fingerprint_reuse_before', 0) > 5:
            reasons.append("Device fingerprint has been used for many transactions")
        
        if feature_values.get('ip_address_reuse_before', 0) > 5:
            reasons.append("IP address has been used for many transactions")
        
        if feature_values.get('risk_score', 0) > 70:
            reasons.append("Stripe risk score is high (>70)")
        
        if feature_values.get('time_between_transactions', 999999) < 60:
            reasons.append("Very short time between transactions (<60s)")
        
        if feature_values.get('past_chargebacks', 0) > 0:
            reasons.append("Previous chargebacks found for this user")
        
        if feature_values.get('amount_log', 0) > 8:
            reasons.append("Transaction amount is extremely high")
        
        return reasons
    
    def _get_email_domain_risk(self, email: str) -> int:
        """Calculate email domain risk score"""
        if '@' not in email:
            return 1
        
        domain = email.split('@')[-1].lower()
        
        # Disposable email hints
        disposable_hints = ['tempmail', 'throwaway', 'guerrillamail', 'mailinator', '10minutemail']
        
        if domain in settings.COMMON_DOMAINS:
            return 0
        elif any(hint in domain for hint in disposable_hints):
            return 2
        else:
            return 1
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            "model_type": self.model_type,
            "features_count": len(self.chargeback_features),
            "model_path": self.model_path,
            "features": self.chargeback_features[:10] if self.chargeback_features else []  # Show first 10 features
        }
