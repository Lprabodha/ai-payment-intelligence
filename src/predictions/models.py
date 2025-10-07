"""
AI model loading and prediction utilities
"""
import os
import joblib
import numpy as np
from datetime import datetime, timedelta
# Removed TensorFlow import - using scikit-learn models only
from config.settings import settings

class ModelManager:
    """Manages AI model loading and predictions"""
    
    def __init__(self):
        self.models = {}
        self.model_path = settings.get_model_path()
        self._load_models()
    
    def _load_models(self):
        """Load all available AI models"""
        print("Loading AI models...")
        
        # Load fraud detection models
        self._load_fraud_models()
        
        # Load chargeback prediction models
        self._load_chargeback_models()
        
        # Load smart routing models
        self._load_routing_models()
        
        # Load subscription revenue models
        self._load_subscription_models()
        
        print("AI models loading completed")
    
    def _load_fraud_models(self):
        """Load fraud detection models"""
        try:
            fraud_pipeline = self._load_model("fraud_detection_pipeline.pkl")
            if fraud_pipeline:
                self.models['fraud_pipeline'] = fraud_pipeline
                print("Loaded fraud detection ensemble")
            else:
                # Try loading individual components
                fraud_model = self._load_model("fraud_detection_model.pkl")
                fraud_scaler = self._load_model("fraud_detection_scaler.pkl")
                
                if fraud_model and fraud_scaler:
                    self.models['fraud_model'] = fraud_model
                    self.models['fraud_scaler'] = fraud_scaler
                    print("Loaded fraud detection model (legacy)")
                else:
                    print("Fraud detection model not found")
        except Exception as e:
            print(f"Error loading fraud models: {e}")
    
    def _load_chargeback_models(self):
        """Load chargeback prediction models"""
        try:
            chargeback_pipeline = self._load_model("chargeback_pipeline.pkl")
            if chargeback_pipeline:
                self.models['chargeback_pipeline'] = chargeback_pipeline
                print("Loaded chargeback prediction ensemble")
            else:
                # Try loading individual components
                chargeback_model = self._load_model("chargeback_prediction_model.pkl")
                chargeback_scaler = self._load_model("chargeback_prediction_scaler.pkl")
                
                if chargeback_model and chargeback_scaler:
                    self.models['chargeback_model'] = chargeback_model
                    self.models['chargeback_scaler'] = chargeback_scaler
                    print("Loaded chargeback prediction model (legacy)")
                else:
                    print("Chargeback prediction model not found")
        except Exception as e:
            print(f"Error loading chargeback models: {e}")
    
    def _load_routing_models(self):
        """Load smart routing models"""
        try:
            # Try to load scikit-learn routing model
            routing_model = self._load_model("smart_routing_rf.pkl")
            if routing_model:
                self.models['routing_model'] = routing_model
                print("Loaded smart routing model")
            else:
                print("Smart routing model not found")
        except Exception as e:
            print(f"Error loading routing models: {e}")
    
    def _load_subscription_models(self):
        """Load subscription revenue models"""
        try:
            ensemble_model = self._load_model("subscription_ensemble_model.pkl")
            subscription_scaler = self._load_model("subscription_revenue_scaler.pkl")
            
            if ensemble_model and subscription_scaler:
                self.models['subscription_ensemble'] = ensemble_model
                self.models['subscription_scaler'] = subscription_scaler
                print("Loaded subscription revenue models")
            else:
                print("Subscription revenue models not found")
        except Exception as e:
            print(f"Error loading subscription models: {e}")
    
    def _load_model(self, model_name):
        """Load a specific model by name"""
        model_path = os.path.join(self.model_path, model_name)
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                print(f"Successfully loaded model: {model_name}")
                return model
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
                return None
        else:
            print(f"Model file not found: {model_path}")
            return None
    
    def get_model(self, model_name):
        """Get a loaded model by name"""
        return self.models.get(model_name)
    
    def has_model(self, model_name):
        """Check if a model is loaded"""
        return model_name in self.models

# Global model manager instance
model_manager = ModelManager()
