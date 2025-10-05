"""
Feature extraction methods for risk scoring engine
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from database.connection import db
from risk_engine.cache import cache_service
from risk_engine.models import VelocityFeatures, IPReputationFeatures, DeviceFeatures, GeographicFeatures
import requests
import re
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Feature extraction for risk scoring"""
    
    def __init__(self):
        self.country_risk_scores = self._load_country_risk_scores()
        self.device_patterns = self._load_device_patterns()
        self.browser_risk_scores = self._load_browser_risk_scores()
    
    def _load_country_risk_scores(self) -> Dict[str, float]:
        """Load country risk scores"""
        return {
            'US': 0.1, 'CA': 0.1, 'GB': 0.1, 'AU': 0.1, 'DE': 0.1,
            'FR': 0.15, 'IT': 0.2, 'ES': 0.2, 'BR': 0.3, 'MX': 0.3,
            'IN': 0.4, 'CN': 0.5, 'RU': 0.6, 'NG': 0.7, 'PK': 0.8,
            'BD': 0.8, 'VN': 0.6, 'TH': 0.5, 'ID': 0.6, 'PH': 0.7
        }
    
    def _load_device_patterns(self) -> Dict[str, List[str]]:
        """Load device patterns for detection"""
        return {
            'mobile': ['Mobile', 'Android', 'iPhone', 'iPad', 'Windows Phone'],
            'bot': ['bot', 'crawler', 'spider', 'scraper', 'curl', 'wget'],
            'suspicious': ['HeadlessChrome', 'PhantomJS', 'Selenium']
        }
    
    def _load_browser_risk_scores(self) -> Dict[str, float]:
        """Load browser risk scores"""
        return {
            'Chrome': 0.1, 'Firefox': 0.1, 'Safari': 0.1, 'Edge': 0.1,
            'Opera': 0.2, 'Internet Explorer': 0.4, 'Unknown': 0.5,
            'HeadlessChrome': 0.8, 'PhantomJS': 0.9, 'Selenium': 0.9
        }
    
    def extract_velocity_features(self, user_id: str, transaction_data: Dict[str, Any]) -> VelocityFeatures:
        """Extract user velocity features"""
        try:
            # Check cache first
            cached_velocity = cache_service.get_user_velocity(user_id)
            if cached_velocity:
                return VelocityFeatures(**cached_velocity)
            
            # Get transaction history from database
            now = datetime.utcnow()
            one_hour_ago = now - timedelta(hours=1)
            one_day_ago = now - timedelta(days=1)
            one_week_ago = now - timedelta(days=7)
            
            # Query transactions
            recent_transactions = list(db["transactions"].find({
                "email": transaction_data.get("email"),
                "created_at": {"$gte": one_week_ago}
            }).sort("created_at", -1))
            
            # Calculate velocity features
            transactions_last_hour = len([t for t in recent_transactions if t["created_at"] >= one_hour_ago])
            transactions_last_day = len([t for t in recent_transactions if t["created_at"] >= one_day_ago])
            transactions_last_week = len(recent_transactions)
            
            amount_last_hour = sum(t.get("amount", 0) for t in recent_transactions if t["created_at"] >= one_hour_ago)
            amount_last_day = sum(t.get("amount", 0) for t in recent_transactions if t["created_at"] >= one_day_ago)
            amount_last_week = sum(t.get("amount", 0) for t in recent_transactions)
            
            unique_merchants = set(t.get("merchant_id") for t in recent_transactions if t.get("merchant_id"))
            unique_countries = set(t.get("billing_address_country") for t in recent_transactions if t.get("billing_address_country"))
            
            amounts = [t.get("amount", 0) for t in recent_transactions if t.get("amount", 0) > 0]
            avg_transaction_amount = np.mean(amounts) if amounts else 0
            max_transaction_amount = max(amounts) if amounts else 0
            
            velocity_features = VelocityFeatures(
                transactions_last_hour=transactions_last_hour,
                transactions_last_day=transactions_last_day,
                transactions_last_week=transactions_last_week,
                amount_last_hour=amount_last_hour,
                amount_last_day=amount_last_day,
                amount_last_week=amount_last_week,
                unique_merchants_last_day=len(unique_merchants),
                unique_countries_last_day=len(unique_countries),
                avg_transaction_amount=avg_transaction_amount,
                max_transaction_amount=max_transaction_amount
            )
            
            # Cache the results
            cache_service.cache_user_velocity(user_id, velocity_features.__dict__)
            
            return velocity_features
            
        except Exception as e:
            logger.error(f"Error extracting velocity features: {e}")
            return VelocityFeatures()
    
    def extract_ip_reputation_features(self, ip_address: str, transaction_data: Dict[str, Any]) -> IPReputationFeatures:
        """Extract IP reputation features"""
        try:
            # Check cache first
            cached_reputation = cache_service.get_ip_reputation(ip_address)
            if cached_reputation:
                return IPReputationFeatures(**cached_reputation)
            
            # Initialize features
            ip_features = IPReputationFeatures()
            
            # Basic IP validation
            if not self._is_valid_ip(ip_address):
                ip_features.risk_score = 1.0
                ip_features.reputation_score = 0.0
                return ip_features
            
            # Check for proxy/VPN/TOR indicators
            ip_features.is_proxy = self._check_proxy(ip_address)
            ip_features.is_vpn = self._check_vpn(ip_address)
            ip_features.is_tor = self._check_tor(ip_address)
            ip_features.is_datacenter = self._check_datacenter(ip_address)
            
            # Country mismatch check
            ip_country = self._get_ip_country(ip_address)
            billing_country = transaction_data.get("billing_address_country")
            ip_features.country_mismatch = (ip_country != billing_country) if billing_country else False
            
            # Calculate risk score
            risk_factors = []
            if ip_features.is_proxy:
                risk_factors.append(0.3)
            if ip_features.is_vpn:
                risk_factors.append(0.4)
            if ip_features.is_tor:
                risk_factors.append(0.8)
            if ip_features.is_datacenter:
                risk_factors.append(0.2)
            if ip_features.country_mismatch:
                risk_factors.append(0.3)
            
            ip_features.risk_score = min(sum(risk_factors), 1.0)
            ip_features.reputation_score = 1.0 - ip_features.risk_score
            
            # Cache the results
            cache_service.cache_ip_reputation(ip_address, ip_features.__dict__)
            
            return ip_features
            
        except Exception as e:
            logger.error(f"Error extracting IP reputation features: {e}")
            return IPReputationFeatures()
    
    def extract_device_features(self, device_fingerprint: str, user_agent: str) -> DeviceFeatures:
        """Extract device fingerprint features"""
        try:
            # Check cache first
            cached_device = cache_service.get_device_fingerprint(device_fingerprint)
            if cached_device:
                return DeviceFeatures(**cached_device)
            
            device_features = DeviceFeatures()
            
            if user_agent:
                user_agent_lower = user_agent.lower()
                
                # Mobile detection
                device_features.is_mobile = any(pattern.lower() in user_agent_lower 
                                              for pattern in self.device_patterns['mobile'])
                
                # Bot detection
                device_features.is_bot = any(pattern.lower() in user_agent_lower 
                                           for pattern in self.device_patterns['bot'])
                
                # Browser risk assessment
                browser_risk = 0.5  # Default
                for browser, risk in self.browser_risk_scores.items():
                    if browser.lower() in user_agent_lower:
                        browser_risk = risk
                        break
                
                device_features.browser_risk = browser_risk
                
                # OS risk assessment
                os_risk = 0.3  # Default
                if 'windows' in user_agent_lower:
                    os_risk = 0.2
                elif 'mac' in user_agent_lower or 'ios' in user_agent_lower:
                    os_risk = 0.1
                elif 'android' in user_agent_lower:
                    os_risk = 0.3
                elif 'linux' in user_agent_lower:
                    os_risk = 0.4
                
                device_features.os_risk = os_risk
                
                # Suspicious patterns
                if any(pattern.lower() in user_agent_lower 
                      for pattern in self.device_patterns['suspicious']):
                    device_features.fingerprint_risk = 0.9
                else:
                    device_features.fingerprint_risk = 0.1
            
            # Device consistency (simplified)
            device_features.device_consistency = 0.8  # Default, would need historical data
            
            # Cache the results
            cache_service.cache_device_fingerprint(device_fingerprint, device_features.__dict__)
            
            return device_features
            
        except Exception as e:
            logger.error(f"Error extracting device features: {e}")
            return DeviceFeatures()
    
    def extract_geographic_features(self, transaction_data: Dict[str, Any]) -> GeographicFeatures:
        """Extract geographic velocity features"""
        try:
            geo_features = GeographicFeatures()
            
            # Country risk score
            billing_country = transaction_data.get("billing_address_country")
            if billing_country:
                geo_features.country_risk_score = self.country_risk_scores.get(billing_country, 0.5)
            
            # Distance from home (simplified)
            geo_features.distance_from_home = 0.0  # Would need geolocation data
            
            # Timezone mismatch (simplified)
            geo_features.timezone_mismatch = False  # Would need timezone data
            
            # Location consistency (simplified)
            geo_features.location_consistency = 0.8  # Default
            
            # Velocity anomaly (simplified)
            geo_features.velocity_anomaly = False  # Would need historical location data
            
            return geo_features
            
        except Exception as e:
            logger.error(f"Error extracting geographic features: {e}")
            return GeographicFeatures()
    
    def _is_valid_ip(self, ip_address: str) -> bool:
        """Validate IP address format"""
        try:
            import ipaddress
            ipaddress.ip_address(ip_address)
            return True
        except ValueError:
            return False
    
    def _check_proxy(self, ip_address: str) -> bool:
        """Check if IP is a proxy (simplified)"""
        # In production, would use a proxy detection service
        proxy_ranges = [
            "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"  # Private ranges
        ]
        
        try:
            import ipaddress
            ip = ipaddress.ip_address(ip_address)
            for range_str in proxy_ranges:
                if ip in ipaddress.ip_network(range_str):
                    return True
            return False
        except:
            return False
    
    def _check_vpn(self, ip_address: str) -> bool:
        """Check if IP is a VPN (simplified)"""
        # In production, would use a VPN detection service
        # For now, return False
        return False
    
    def _check_tor(self, ip_address: str) -> bool:
        """Check if IP is a TOR exit node (simplified)"""
        # In production, would use a TOR exit node list
        # For now, return False
        return False
    
    def _check_datacenter(self, ip_address: str) -> bool:
        """Check if IP is from a datacenter (simplified)"""
        # In production, would use a datacenter IP list
        # For now, return False
        return False
    
    def _get_ip_country(self, ip_address: str) -> Optional[str]:
        """Get country from IP address (simplified)"""
        # In production, would use a geolocation service
        # For now, return None
        return None

# Global feature extractor instance
feature_extractor = FeatureExtractor()
