"""
Redis caching service for risk scoring engine
"""
import json
import redis
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import logging
from config.settings import settings

logger = logging.getLogger(__name__)

class RiskCacheService:
    """Redis-based caching service for risk scoring data"""
    
    def __init__(self):
        self.redis_client = None
        self.cache_prefix = "risk_engine:"
        self.default_ttl = 3600  # 1 hour
        self._connect()
    
    def _connect(self):
        """Connect to Redis"""
        try:
            # Use Redis URL from settings or default
            redis_url = getattr(settings, 'REDIS_URL')
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            
            # Test connection
            self.redis_client.ping()
            logger.info("Connected to Redis for risk scoring cache")
            
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory cache fallback.")
            self.redis_client = None
    
    def _get_key(self, key_type: str, identifier: str) -> str:
        """Generate cache key"""
        return f"{self.cache_prefix}{key_type}:{identifier}"
    
    def _serialize_data(self, data: Any) -> str:
        """Serialize data for Redis storage"""
        if isinstance(data, datetime):
            return data.isoformat()
        return json.dumps(data, default=str)
    
    def _deserialize_data(self, data: str, data_type: type = None) -> Any:
        """Deserialize data from Redis"""
        try:
            if data_type == datetime:
                return datetime.fromisoformat(data)
            return json.loads(data)
        except (json.JSONDecodeError, ValueError):
            return data
    
    def cache_user_velocity(self, user_id: str, velocity_data: Dict[str, Any], ttl: int = None) -> bool:
        """Cache user velocity data"""
        try:
            if not self.redis_client:
                return False
            
            key = self._get_key("user_velocity", user_id)
            serialized_data = self._serialize_data(velocity_data)
            
            ttl = ttl or self.default_ttl
            return self.redis_client.setex(key, ttl, serialized_data)
            
        except Exception as e:
            logger.error(f"Error caching user velocity: {e}")
            return False
    
    def get_user_velocity(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get cached user velocity data"""
        try:
            if not self.redis_client:
                return None
            
            key = self._get_key("user_velocity", user_id)
            data = self.redis_client.get(key)
            
            if data:
                return self._deserialize_data(data)
            return None
            
        except Exception as e:
            logger.error(f"Error getting user velocity: {e}")
            return None
    
    def cache_ip_reputation(self, ip_address: str, reputation_data: Dict[str, Any], ttl: int = None) -> bool:
        """Cache IP reputation data"""
        try:
            if not self.redis_client:
                return False
            
            key = self._get_key("ip_reputation", ip_address)
            serialized_data = self._serialize_data(reputation_data)
            
            ttl = ttl or (24 * 3600)  # 24 hours for IP reputation
            return self.redis_client.setex(key, ttl, serialized_data)
            
        except Exception as e:
            logger.error(f"Error caching IP reputation: {e}")
            return False
    
    def get_ip_reputation(self, ip_address: str) -> Optional[Dict[str, Any]]:
        """Get cached IP reputation data"""
        try:
            if not self.redis_client:
                return None
            
            key = self._get_key("ip_reputation", ip_address)
            data = self.redis_client.get(key)
            
            if data:
                return self._deserialize_data(data)
            return None
            
        except Exception as e:
            logger.error(f"Error getting IP reputation: {e}")
            return None
    
    def cache_device_fingerprint(self, fingerprint: str, device_data: Dict[str, Any], ttl: int = None) -> bool:
        """Cache device fingerprint data"""
        try:
            if not self.redis_client:
                return False
            
            key = self._get_key("device_fingerprint", fingerprint)
            serialized_data = self._serialize_data(device_data)
            
            ttl = ttl or (7 * 24 * 3600)  # 7 days for device data
            return self.redis_client.setex(key, ttl, serialized_data)
            
        except Exception as e:
            logger.error(f"Error caching device fingerprint: {e}")
            return False
    
    def get_device_fingerprint(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        """Get cached device fingerprint data"""
        try:
            if not self.redis_client:
                return None
            
            key = self._get_key("device_fingerprint", fingerprint)
            data = self.redis_client.get(key)
            
            if data:
                return self._deserialize_data(data)
            return None
            
        except Exception as e:
            logger.error(f"Error getting device fingerprint: {e}")
            return None
    
    def cache_transaction_history(self, user_id: str, transactions: List[Dict[str, Any]], ttl: int = None) -> bool:
        """Cache user transaction history"""
        try:
            if not self.redis_client:
                return False
            
            key = self._get_key("transaction_history", user_id)
            serialized_data = self._serialize_data(transactions)
            
            ttl = ttl or self.default_ttl
            return self.redis_client.setex(key, ttl, serialized_data)
            
        except Exception as e:
            logger.error(f"Error caching transaction history: {e}")
            return False
    
    def get_transaction_history(self, user_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached transaction history"""
        try:
            if not self.redis_client:
                return None
            
            key = self._get_key("transaction_history", user_id)
            data = self.redis_client.get(key)
            
            if data:
                return self._deserialize_data(data)
            return None
            
        except Exception as e:
            logger.error(f"Error getting transaction history: {e}")
            return None
    
    def cache_risk_assessment(self, transaction_id: str, assessment_data: Dict[str, Any], ttl: int = None) -> bool:
        """Cache risk assessment result"""
        try:
            if not self.redis_client:
                return False
            
            key = self._get_key("risk_assessment", transaction_id)
            serialized_data = self._serialize_data(assessment_data)
            
            ttl = ttl or (24 * 3600)  # 24 hours for risk assessments
            return self.redis_client.setex(key, ttl, serialized_data)
            
        except Exception as e:
            logger.error(f"Error caching risk assessment: {e}")
            return False
    
    def get_risk_assessment(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Get cached risk assessment"""
        try:
            if not self.redis_client:
                return None
            
            key = self._get_key("risk_assessment", transaction_id)
            data = self.redis_client.get(key)
            
            if data:
                return self._deserialize_data(data)
            return None
            
        except Exception as e:
            logger.error(f"Error getting risk assessment: {e}")
            return None
    
    def invalidate_user_cache(self, user_id: str) -> bool:
        """Invalidate all cache entries for a user"""
        try:
            if not self.redis_client:
                return False
            
            patterns = [
                self._get_key("user_velocity", user_id),
                self._get_key("transaction_history", user_id)
            ]
            
            for pattern in patterns:
                self.redis_client.delete(pattern)
            
            return True
            
        except Exception as e:
            logger.error(f"Error invalidating user cache: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            if not self.redis_client:
                return {"status": "disabled", "reason": "Redis not available"}
            
            info = self.redis_client.info()
            return {
                "status": "active",
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "0B"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(info)
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"status": "error", "error": str(e)}
    
    def _calculate_hit_rate(self, info: Dict[str, Any]) -> float:
        """Calculate cache hit rate"""
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses
        
        if total == 0:
            return 0.0
        
        return hits / total

# Global cache service instance
cache_service = RiskCacheService()
