"""
Advanced feature engineering for fraud detection and chargeback prevention
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import hashlib


class AdvancedFeatureEngine:
    """Enhanced feature engineering for fraud and chargeback models"""
    
    def __init__(self):
        self.known_good_emails = set()
        self.known_bad_emails = set()
        self.velocity_thresholds = {
            'high_frequency': 10,  # transactions per hour
            'burst_detection': 5   # transactions per 10 minutes
        }
    
    def add_behavioral_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced behavioral anomaly detection features"""
        df = df.copy()
        
        # 1. Transaction timing anomalies
        df['hour_risk_score'] = df['hour'].apply(self._calculate_hour_risk)
        df['weekend_risk_multiplier'] = df['is_weekend'] * 1.5 + 1.0
        
        # 2. Amount behavior anomalies
        df['amount_percentile_user'] = self._calculate_user_amount_percentile(df)
        df['amount_spike_severity'] = np.clip(df['amount_zscore_10'], 0, 5)
        df['micro_payment_flag'] = (df['amount'] < 5).astype(int)
        df['large_payment_flag'] = (df['amount'] > 1000).astype(int)
        
        # 3. Session behavior patterns
        df['rapid_succession_flag'] = self._detect_rapid_succession(df)
        df['burst_activity_score'] = self._calculate_burst_score(df)
        
        # 4. Geographic anomalies
        df['geo_velocity_impossible'] = self._detect_impossible_geo_velocity(df)
        df['high_risk_country_flag'] = self._mark_high_risk_countries(df)
        
        return df
    
    def add_network_analysis_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add network analysis and graph-based features"""
        df = df.copy()
        
        # 1. Email domain clustering
        df['email_domain_cluster'] = self._cluster_email_domains(df)
        df['domain_reputation_score'] = self._calculate_domain_reputation(df)
        
        # 2. Device fingerprint network
        df['device_sharing_score'] = self._calculate_device_sharing(df)
        df['device_velocity_score'] = self._calculate_device_velocity(df)
        
        # 3. IP address analysis
        df['ip_reputation_score'] = self._calculate_ip_reputation(df)
        df['proxy_vpn_probability'] = self._detect_proxy_vpn(df)
        df['ip_geolocation_mismatch'] = self._detect_geo_mismatch(df)
        
        # 4. Transaction graph features
        df['transaction_centrality'] = self._calculate_transaction_centrality(df)
        df['community_risk_score'] = self._calculate_community_risk(df)
        
        return df
    
    def add_advanced_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced velocity and frequency analysis"""
        df = df.copy()
        
        # 1. Multi-dimensional velocity
        windows = [300, 600, 1800, 3600, 21600, 86400]  # 5m, 10m, 30m, 1h, 6h, 24h
        
        for window in windows:
            window_name = self._seconds_to_name(window)
            
            # Transaction count velocity
            df[f'tx_velocity_{window_name}'] = self._rolling_velocity(
                df, 'email', 'created_at', window
            )
            
            # Amount velocity
            df[f'amount_velocity_{window_name}'] = self._rolling_amount_velocity(
                df, 'email', 'created_at', 'amount', window
            )
            
            # Unique card velocity
            df[f'card_velocity_{window_name}'] = self._rolling_unique_velocity(
                df, 'email', 'created_at', 'card_country', window
            )
        
        # 2. Acceleration features (change in velocity)
        df['tx_acceleration_1h'] = df['tx_velocity_1h'] - df['tx_velocity_30m']
        df['amount_acceleration_1h'] = df['amount_velocity_1h'] - df['amount_velocity_30m']
        
        return df
    
    def add_contextual_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add contextual and environmental features"""
        df = df.copy()
        
        # 1. Temporal context
        df['is_holiday'] = self._mark_holidays(df['created_at'])
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        df['day_of_month'] = df['created_at'].dt.day
        df['is_month_end'] = (df['day_of_month'] >= 28).astype(int)
        
        # 2. Payment context
        df['gateway_risk_score'] = self._calculate_gateway_risk(df)
        df['payment_method_risk'] = self._calculate_payment_method_risk(df)
        df['currency_mismatch_flag'] = self._detect_currency_mismatch(df)
        
        # 3. Customer lifecycle context
        df['customer_lifetime_value'] = self._estimate_clv(df)
        df['customer_maturity_score'] = self._calculate_customer_maturity(df)
        df['churn_risk_score'] = self._calculate_churn_risk(df)
        
        return df
    
    def add_ensemble_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add composite risk scores from multiple models"""
        df = df.copy()
        
        # 1. Behavioral risk composite
        behavioral_features = [
            'amount_spike_severity', 'rapid_succession_flag', 
            'burst_activity_score', 'device_sharing_score'
        ]
        df['behavioral_risk_composite'] = self._calculate_composite_score(
            df, behavioral_features, weights=[0.3, 0.25, 0.25, 0.2]
        )
        
        # 2. Geographic risk composite
        geo_features = [
            'country_mismatch', 'geo_velocity_impossible', 
            'high_risk_country_flag', 'ip_geolocation_mismatch'
        ]
        df['geographic_risk_composite'] = self._calculate_composite_score(
            df, geo_features, weights=[0.25, 0.35, 0.2, 0.2]
        )
        
        # 3. Network risk composite
        network_features = [
            'ip_reputation_score', 'proxy_vpn_probability',
            'domain_reputation_score', 'community_risk_score'
        ]
        df['network_risk_composite'] = self._calculate_composite_score(
            df, network_features, weights=[0.3, 0.3, 0.2, 0.2]
        )
        
        return df
    
    # Helper methods (implement based on your specific data and requirements)
    
    def _calculate_hour_risk(self, hour: int) -> float:
        """Calculate risk score based on hour of day"""
        # Higher risk during off-hours (2-6 AM)
        if 2 <= hour <= 6:
            return 2.0
        elif 22 <= hour or hour <= 1:
            return 1.5
        else:
            return 1.0
    
    def _calculate_user_amount_percentile(self, df: pd.DataFrame) -> pd.Series:
        """Calculate what percentile current amount is for user's history"""
        return df.groupby('email')['amount'].rank(pct=True)
    
    def _detect_rapid_succession(self, df: pd.DataFrame) -> pd.Series:
        """Detect transactions in rapid succession (< 30 seconds apart)"""
        time_diff = df.groupby('email')['created_at'].diff().dt.total_seconds()
        return (time_diff < 30).fillna(False).astype(int)
    
    def _calculate_burst_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate burst activity score"""
        # Implement based on transaction frequency spikes
        return df['past_tx_count_10m'] / (df['past_tx_count_1h'] + 1)
    
    def _detect_impossible_geo_velocity(self, df: pd.DataFrame) -> pd.Series:
        """Detect geographically impossible transaction velocity"""
        # Placeholder - implement based on IP geolocation data
        return pd.Series(0, index=df.index)
    
    def _mark_high_risk_countries(self, df: pd.DataFrame) -> pd.Series:
        """Mark transactions from high-risk countries"""
        high_risk_countries = {'AF', 'IQ', 'LY', 'SO', 'SY', 'YE'}  # Example
        return df['card_country'].isin(high_risk_countries).astype(int)
    
    def _cluster_email_domains(self, df: pd.DataFrame) -> pd.Series:
        """Cluster email domains by risk level"""
        # Implement domain clustering logic
        return pd.Series(0, index=df.index)
    
    def _calculate_domain_reputation(self, df: pd.DataFrame) -> pd.Series:
        """Calculate email domain reputation score"""
        # Implement based on historical fraud rates by domain
        return pd.Series(0.5, index=df.index)
    
    def _calculate_device_sharing(self, df: pd.DataFrame) -> pd.Series:
        """Calculate how much a device is shared across users"""
        device_user_counts = df.groupby('fingerprint')['email'].nunique()
        return df['fingerprint'].map(device_user_counts) / 10.0  # Normalize
    
    def _calculate_device_velocity(self, df: pd.DataFrame) -> pd.Series:
        """Calculate device usage velocity"""
        return df['fp_reuse_before'] / (df['past_tx_count'] + 1)
    
    def _calculate_ip_reputation(self, df: pd.DataFrame) -> pd.Series:
        """Calculate IP reputation score"""
        # Implement based on IP blacklists, reputation services
        return pd.Series(0.5, index=df.index)
    
    def _detect_proxy_vpn(self, df: pd.DataFrame) -> pd.Series:
        """Detect proxy/VPN usage probability"""
        # Implement based on IP analysis
        return pd.Series(0.1, index=df.index)
    
    def _detect_geo_mismatch(self, df: pd.DataFrame) -> pd.Series:
        """Detect IP geolocation vs billing address mismatch"""
        return df['country_mismatch'].astype(float)
    
    def _calculate_transaction_centrality(self, df: pd.DataFrame) -> pd.Series:
        """Calculate centrality in transaction network"""
        # Implement graph-based centrality measures
        return pd.Series(0.1, index=df.index)
    
    def _calculate_community_risk(self, df: pd.DataFrame) -> pd.Series:
        """Calculate risk score based on transaction community"""
        # Implement community detection and risk scoring
        return pd.Series(0.2, index=df.index)
    
    def _rolling_velocity(self, df: pd.DataFrame, group_col: str, 
                         time_col: str, window_seconds: int) -> pd.Series:
        """Calculate rolling velocity for any grouping"""
        # Implement rolling window velocity calculation
        return pd.Series(0, index=df.index)
    
    def _rolling_amount_velocity(self, df: pd.DataFrame, group_col: str,
                                time_col: str, amount_col: str, window_seconds: int) -> pd.Series:
        """Calculate rolling amount velocity"""
        # Implement rolling amount velocity
        return pd.Series(0.0, index=df.index)
    
    def _rolling_unique_velocity(self, df: pd.DataFrame, group_col: str,
                                time_col: str, unique_col: str, window_seconds: int) -> pd.Series:
        """Calculate rolling unique value velocity"""
        # Implement rolling unique count velocity
        return pd.Series(0, index=df.index)
    
    def _seconds_to_name(self, seconds: int) -> str:
        """Convert seconds to readable time window name"""
        if seconds < 3600:
            return f"{seconds//60}m"
        elif seconds < 86400:
            return f"{seconds//3600}h"
        else:
            return f"{seconds//86400}d"
    
    def _mark_holidays(self, dates: pd.Series) -> pd.Series:
        """Mark holiday dates"""
        # Implement holiday detection
        return pd.Series(0, index=dates.index)
    
    def _calculate_gateway_risk(self, df: pd.DataFrame) -> pd.Series:
        """Calculate gateway-specific risk scores"""
        gateway_risk_map = {
            'stripe': 0.1,
            'paypal': 0.15,
            'square': 0.2,
            'unknown': 0.5
        }
        return df['gateway'].map(gateway_risk_map).fillna(0.3)
    
    def _calculate_payment_method_risk(self, df: pd.DataFrame) -> pd.Series:
        """Calculate payment method risk"""
        # Implement based on card type, issuer, etc.
        return pd.Series(0.2, index=df.index)
    
    def _detect_currency_mismatch(self, df: pd.DataFrame) -> pd.Series:
        """Detect currency vs country mismatch"""
        # Implement currency validation
        return pd.Series(0, index=df.index)
    
    def _estimate_clv(self, df: pd.DataFrame) -> pd.Series:
        """Estimate customer lifetime value"""
        return df.groupby('email')['amount'].transform('sum')
    
    def _calculate_customer_maturity(self, df: pd.DataFrame) -> pd.Series:
        """Calculate customer account maturity score"""
        return df.get('account_age_days', 0) / 365.0
    
    def _calculate_churn_risk(self, df: pd.DataFrame) -> pd.Series:
        """Calculate customer churn risk"""
        # Implement based on transaction patterns
        return pd.Series(0.1, index=df.index)
    
    def _calculate_composite_score(self, df: pd.DataFrame, features: List[str], 
                                  weights: List[float]) -> pd.Series:
        """Calculate weighted composite score"""
        score = pd.Series(0.0, index=df.index)
        for feature, weight in zip(features, weights):
            if feature in df.columns:
                normalized_feature = (df[feature] - df[feature].min()) / (df[feature].max() - df[feature].min() + 1e-8)
                score += weight * normalized_feature
        return score
