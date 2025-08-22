"""
Ensemble model architecture for fraud detection and chargeback prevention
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import joblib
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')


class EnsembleModelManager:
    """
    Advanced ensemble model for fraud detection and chargeback prevention
    Combines multiple ML models with rule-based systems
    """
    
    def __init__(self):
        self.models = {}
        self.meta_model = None
        self.feature_importance_weights = {}
        self.model_weights = {}
        self.rules_engine = RulesEngine()
        
    def create_base_models(self) -> Dict[str, Any]:
        """Create diverse base models for ensemble"""
        
        models = {
            'xgboost': XGBClassifier(
                n_estimators=800,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.8,
                tree_method="hist",
                random_state=42,
                n_jobs=-1
            ),
            
            'lightgbm': LGBMClassifier(
                n_estimators=800,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            
            'random_forest': RandomForestClassifier(
                n_estimators=500,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
            
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.9,
                random_state=42
            ),
            
            'logistic': LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )
        }
        
        return models
    
    def train_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        """Train ensemble of models with cross-validation"""
        
        self.models = self.create_base_models()
        model_scores = {}
        model_predictions = {}
        
        # Train each base model
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Get validation predictions
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            model_predictions[name] = y_pred_proba
            
            # Calculate performance metrics
            pr_auc = average_precision_score(y_val, y_pred_proba)
            roc_auc = roc_auc_score(y_val, y_pred_proba) if len(np.unique(y_val)) > 1 else 0
            
            model_scores[name] = {
                'pr_auc': pr_auc,
                'roc_auc': roc_auc,
                'combined_score': 0.7 * pr_auc + 0.3 * roc_auc
            }
            
            print(f"{name} - PR-AUC: {pr_auc:.4f}, ROC-AUC: {roc_auc:.4f}")
        
        # Calculate model weights based on performance
        self._calculate_model_weights(model_scores)
        
        # Train meta-learner
        self._train_meta_learner(model_predictions, y_val)
        
        return model_scores
    
    def _calculate_model_weights(self, model_scores: Dict[str, Dict[str, float]]):
        """Calculate weights for each model based on performance"""
        
        combined_scores = {name: scores['combined_score'] 
                          for name, scores in model_scores.items()}
        
        # Softmax normalization for weights
        scores_array = np.array(list(combined_scores.values()))
        exp_scores = np.exp(scores_array - np.max(scores_array))
        weights = exp_scores / np.sum(exp_scores)
        
        self.model_weights = {name: weight for name, weight in 
                             zip(combined_scores.keys(), weights)}
        
        print(f"Model weights: {self.model_weights}")
    
    def _train_meta_learner(self, model_predictions: Dict[str, np.ndarray], 
                           y_val: pd.Series):
        """Train meta-learner to combine base model predictions"""
        
        # Create meta-features from base model predictions
        meta_features = np.column_stack(list(model_predictions.values()))
        
        # Train simple logistic regression as meta-learner
        self.meta_model = LogisticRegression(random_state=42)
        self.meta_model.fit(meta_features, y_val)
        
        print("Meta-learner trained successfully")
    
    def predict_ensemble(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Make ensemble predictions with explainability"""
        
        base_predictions = {}
        
        # Get predictions from each base model
        for name, model in self.models.items():
            pred_proba = model.predict_proba(X)[:, 1]
            base_predictions[name] = pred_proba
        
        # Weighted average of base models
        weighted_avg = np.zeros(len(X))
        for name, pred in base_predictions.items():
            weighted_avg += self.model_weights[name] * pred
        
        # Meta-learner prediction
        meta_features = np.column_stack(list(base_predictions.values()))
        meta_prediction = self.meta_model.predict_proba(meta_features)[:, 1]
        
        # Rule-based adjustment
        rule_scores = self.rules_engine.evaluate_rules(X)
        
        # Final ensemble (70% ML, 30% rules)
        final_scores = 0.7 * meta_prediction + 0.3 * rule_scores
        
        # Generate explanations
        explanations = self._generate_explanations(X, base_predictions, rule_scores)
        
        return final_scores, meta_prediction, explanations
    
    def _generate_explanations(self, X: pd.DataFrame, 
                              base_predictions: Dict[str, np.ndarray],
                              rule_scores: np.ndarray) -> Dict[str, float]:
        """Generate explanations for predictions"""
        
        explanations = {}
        
        # Feature importance from best performing model
        best_model_name = max(self.model_weights.keys(), 
                             key=lambda x: self.model_weights[x])
        best_model = self.models[best_model_name]
        
        if hasattr(best_model, 'feature_importances_'):
            feature_importances = best_model.feature_importances_
            top_features = np.argsort(feature_importances)[-10:]
            
            explanations['top_features'] = {
                X.columns[i]: feature_importances[i] for i in top_features
            }
        
        # Model contribution breakdown
        explanations['model_contributions'] = {
            name: float(self.model_weights[name] * np.mean(pred))
            for name, pred in base_predictions.items()
        }
        
        explanations['rules_contribution'] = float(np.mean(rule_scores))
        
        return explanations
    
    def save_ensemble(self, filepath: str):
        """Save the entire ensemble model"""
        
        ensemble_data = {
            'models': self.models,
            'meta_model': self.meta_model,
            'model_weights': self.model_weights,
            'rules_engine': self.rules_engine
        }
        
        joblib.dump(ensemble_data, filepath)
        print(f"Ensemble saved to {filepath}")
    
    def load_ensemble(self, filepath: str):
        """Load a saved ensemble model"""
        
        ensemble_data = joblib.load(filepath)
        
        self.models = ensemble_data['models']
        self.meta_model = ensemble_data['meta_model']
        self.model_weights = ensemble_data['model_weights']
        self.rules_engine = ensemble_data['rules_engine']
        
        print(f"Ensemble loaded from {filepath}")


class RulesEngine:
    """
    Rules-based scoring engine that complements ML models
    Based on the fraud prevention rules from your attachment
    """
    
    def __init__(self):
        self.rules = self._initialize_rules()
    
    def _initialize_rules(self) -> List[Dict]:
        """Initialize rules based on your fraud prevention rules"""
        
        rules = [
            # High-frequency transaction rules
            {
                'name': 'high_frequency_transactions',
                'condition': lambda x: (x['past_tx_count_1h'] > 10),
                'score': 0.8,
                'weight': 0.15
            },
            
            # Velocity-based rules
            {
                'name': 'burst_transactions',
                'condition': lambda x: (x['past_tx_count_10m'] > 5),
                'score': 0.7,
                'weight': 0.12
            },
            
            # Geographic anomalies
            {
                'name': 'country_mismatch_high_amount',
                'condition': lambda x: (x['country_mismatch'] == 1) & (x['amount'] > 500),
                'score': 0.75,
                'weight': 0.1
            },
            
            # Email domain risks
            {
                'name': 'disposable_email_high_amount',
                'condition': lambda x: (x['email_domain_class'] == 2) & (x['amount'] > 100),
                'score': 0.6,
                'weight': 0.08
            },
            
            # Device/IP anomalies
            {
                'name': 'high_device_sharing',
                'condition': lambda x: x['fp_unique_emails_before'] > 10,
                'score': 0.65,
                'weight': 0.1
            },
            
            # Amount anomalies
            {
                'name': 'extreme_amount_spike',
                'condition': lambda x: x['amount_zscore_10'] > 4,
                'score': 0.7,
                'weight': 0.1
            },
            
            # Timing anomalies
            {
                'name': 'off_hours_high_amount',
                'condition': lambda x: ((x['hour'] < 6) | (x['hour'] > 23)) & (x['amount'] > 300),
                'score': 0.55,
                'weight': 0.08
            },
            
            # Rapid succession
            {
                'name': 'rapid_succession_transactions',
                'condition': lambda x: x['time_since_prev_tx_sec'] < 30,
                'score': 0.6,
                'weight': 0.1
            },
            
            # Historical behavior
            {
                'name': 'high_past_disputes',
                'condition': lambda x: (x['past_disputes'] > 0) & (x['past_tx_count'] < 20),
                'score': 0.8,
                'weight': 0.12
            },
            
            # New customer high risk
            {
                'name': 'new_customer_high_amount',
                'condition': lambda x: (x['past_tx_count'] == 0) & (x['amount'] > 200),
                'score': 0.5,
                'weight': 0.05
            }
        ]
        
        return rules
    
    def evaluate_rules(self, X: pd.DataFrame) -> np.ndarray:
        """Evaluate all rules and return aggregated risk scores"""
        
        scores = np.zeros(len(X))
        
        for rule in self.rules:
            try:
                # Apply rule condition
                mask = rule['condition'](X)
                
                # Add weighted score where condition is true
                scores[mask] += rule['score'] * rule['weight']
                
            except Exception as e:
                print(f"Error in rule {rule['name']}: {e}")
                continue
        
        # Normalize scores to [0, 1]
        max_possible_score = sum(rule['score'] * rule['weight'] for rule in self.rules)
        scores = np.clip(scores / max_possible_score, 0, 1)
        
        return scores
    
    def explain_rules(self, X: pd.DataFrame, threshold: float = 0.1) -> List[Dict]:
        """Explain which rules fired for given transactions"""
        
        explanations = []
        
        for rule in self.rules:
            try:
                mask = rule['condition'](X)
                if np.any(mask):
                    fired_count = np.sum(mask)
                    contribution = rule['score'] * rule['weight']
                    
                    if contribution >= threshold:
                        explanations.append({
                            'rule_name': rule['name'],
                            'fired_count': int(fired_count),
                            'contribution': contribution,
                            'description': self._get_rule_description(rule['name'])
                        })
            except Exception:
                continue
        
        return sorted(explanations, key=lambda x: x['contribution'], reverse=True)
    
    def _get_rule_description(self, rule_name: str) -> str:
        """Get human-readable description of rules"""
        
        descriptions = {
            'high_frequency_transactions': 'More than 10 transactions in the last hour',
            'burst_transactions': 'More than 5 transactions in the last 10 minutes',
            'country_mismatch_high_amount': 'Large transaction with country mismatch',
            'disposable_email_high_amount': 'High amount transaction from disposable email',
            'high_device_sharing': 'Device used by many different email addresses',
            'extreme_amount_spike': 'Transaction amount extremely higher than user average',
            'off_hours_high_amount': 'Large transaction during off-hours',
            'rapid_succession_transactions': 'Transactions less than 30 seconds apart',
            'high_past_disputes': 'User has history of disputes with few transactions',
            'new_customer_high_amount': 'First transaction with high amount'
        }
        
        return descriptions.get(rule_name, 'Custom fraud rule')


class ModelExplainer:
    """
    Provides detailed explanations for model predictions
    """
    
    def __init__(self, ensemble_manager: EnsembleModelManager):
        self.ensemble = ensemble_manager
    
    def explain_prediction(self, X: pd.DataFrame, transaction_idx: int = 0) -> Dict:
        """Provide detailed explanation for a specific prediction"""
        
        single_transaction = X.iloc[[transaction_idx]]
        
        # Get ensemble prediction
        final_score, ml_score, explanations = self.ensemble.predict_ensemble(single_transaction)
        
        # Get rule explanations
        rule_explanations = self.ensemble.rules_engine.explain_rules(single_transaction)
        
        # Feature importance for this transaction
        feature_contributions = self._calculate_feature_contributions(single_transaction)
        
        explanation = {
            'transaction_id': transaction_idx,
            'final_risk_score': float(final_score[0]),
            'ml_score': float(ml_score[0]),
            'rule_score': explanations['rules_contribution'],
            'risk_level': self._categorize_risk(final_score[0]),
            'model_contributions': explanations['model_contributions'],
            'top_features': feature_contributions,
            'fired_rules': rule_explanations,
            'recommendations': self._generate_recommendations(final_score[0], rule_explanations)
        }
        
        return explanation
    
    def _calculate_feature_contributions(self, X: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature contributions to the prediction"""
        
        # Use the best performing model for feature importance
        best_model_name = max(self.ensemble.model_weights.keys(), 
                             key=lambda x: self.ensemble.model_weights[x])
        best_model = self.ensemble.models[best_model_name]
        
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feature_values = X.iloc[0].values
            
            # Calculate contributions (importance * normalized value)
            contributions = {}
            for i, (feature, importance) in enumerate(zip(X.columns, importances)):
                normalized_value = (feature_values[i] - X[feature].min()) / (X[feature].max() - X[feature].min() + 1e-8)
                contributions[feature] = importance * normalized_value
            
            # Return top 10 contributing features
            sorted_contributions = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
            return dict(sorted_contributions[:10])
        
        return {}
    
    def _categorize_risk(self, score: float) -> str:
        """Categorize risk level based on score"""
        
        if score >= 0.8:
            return "VERY HIGH"
        elif score >= 0.6:
            return "HIGH"
        elif score >= 0.4:
            return "MEDIUM"
        elif score >= 0.2:
            return "LOW"
        else:
            return "VERY LOW"
    
    def _generate_recommendations(self, score: float, fired_rules: List[Dict]) -> List[str]:
        """Generate action recommendations based on risk score and rules"""
        
        recommendations = []
        
        if score >= 0.8:
            recommendations.append("BLOCK: Immediate transaction blocking recommended")
            recommendations.append("ALERT: Notify fraud team for manual review")
        elif score >= 0.6:
            recommendations.append("REVIEW: Manual review required before processing")
            recommendations.append("3DS: Force 3D Secure authentication")
        elif score >= 0.4:
            recommendations.append("MONITOR: Enhanced monitoring for subsequent transactions")
            recommendations.append("VERIFY: Additional verification checks recommended")
        elif score >= 0.2:
            recommendations.append("CAUTION: Standard processing with increased logging")
        else:
            recommendations.append("APPROVE: Low risk, proceed with normal processing")
        
        # Add rule-specific recommendations
        for rule in fired_rules[:3]:  # Top 3 rules
            if rule['rule_name'] == 'high_frequency_transactions':
                recommendations.append("Consider velocity limits for this user")
            elif rule['rule_name'] == 'country_mismatch_high_amount':
                recommendations.append("Verify billing address and shipping details")
            elif rule['rule_name'] == 'new_customer_high_amount':
                recommendations.append("Implement stepped transaction limits for new users")
        
        return recommendations
