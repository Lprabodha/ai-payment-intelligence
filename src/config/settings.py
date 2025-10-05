"""
Configuration settings for the AI Payment Intelligence API
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Application settings and configuration"""
    
    # Database Configuration
    MONGO_URI = os.getenv("MONGO_URI")
    DATABASE_NAME = "payment_intelligence"
    
    # Redis Configuration
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Stripe Configuration
    STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
    STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
    
    # Solidgate Configuration
    SOLIDGATE_API_KEY = os.getenv("SOLIDGATE_API_KEY")
    SOLIDGATE_API_SECRET = os.getenv("SOLIDGATE_API_SECRET")
    
    # Model Configuration
    MODEL_PATH = "/src/data/models/"
    
    # API Configuration
    API_TITLE = "AI Payment Intelligence API"
    API_VERSION = "1.0.0"
    API_DESCRIPTION = "Advanced AI-powered payment processing and fraud detection API"
    
    # Risk Score Thresholds
    RADAR_HIGH_OVERRIDE = 65   # >= 65 → force high-risk
    RADAR_MEDIUM_HINT = 55     # >= 55 → bump confidence (optional)
    
    # Common email domains for risk assessment
    COMMON_DOMAINS = {
        "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com",
        "aol.com", "protonmail.com", "zoho.com", "mail.com", "gmx.com"
    }
    
    @classmethod
    def validate_config(cls):
        """Validate that all required configuration is present"""
        required_vars = [
            ("MONGO_URI", cls.MONGO_URI),
            ("STRIPE_SECRET_KEY", cls.STRIPE_SECRET_KEY),
            ("STRIPE_WEBHOOK_SECRET", cls.STRIPE_WEBHOOK_SECRET)
        ]
        
        missing_vars = []
        for var_name, var_value in required_vars:
            if not var_value:
                missing_vars.append(var_name)
        
        if missing_vars:
            raise RuntimeError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        return True
    
    @classmethod
    def get_model_path(cls):
        """Get the model path and validate it exists"""
        if not os.path.exists(cls.MODEL_PATH):
            print(f"Warning: Model directory does not exist: {cls.MODEL_PATH}")
            return cls.MODEL_PATH
        
        # List available model files
        model_files = [f for f in os.listdir(cls.MODEL_PATH) if f.endswith('.pkl')]
        print(f"Available .pkl model files: {model_files}")
        return cls.MODEL_PATH

# Create global settings instance
settings = Settings()