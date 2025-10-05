"""
Enhanced subscription revenue forecasting model
Uses ensemble methods for accurate revenue predictions
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from xgboost import XGBRegressor
from pymongo import MongoClient
from dotenv import load_dotenv
import joblib
import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

class EnhancedSubscriptionForecaster:
    """Enhanced subscription revenue forecasting with ensemble methods"""
    
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_columns = []
        self.results = {}
    
    def load_and_preprocess_data(self):
        """Load and preprocess subscription data"""
        try:
            # Create synthetic data for training (since we don't have real subscription data)
            print("Creating synthetic subscription data for training...")
            
            np.random.seed(42)
            n_samples = 2000
            
            # Generate synthetic features
            data = {
                'account_age_days': np.random.exponential(180, n_samples),
                'renewal_count': np.random.poisson(3, n_samples),
                'average_subscription_value': np.random.normal(50, 20, n_samples),
                'high_value_customer': np.random.binomial(1, 0.3, n_samples),
                'subscription_duration_days': np.random.exponential(30, n_samples),
                'is_weekend': np.random.binomial(1, 0.3, n_samples),
                'customer_satisfaction': np.random.uniform(1, 5, n_samples),
                'payment_success_rate': np.random.beta(8, 2, n_samples),
                'churn_risk_score': np.random.beta(2, 8, n_samples),
                'trial_conversion': np.random.binomial(1, 0.7, n_samples),
                'support_tickets': np.random.poisson(2, n_samples),
                'feature_usage_score': np.random.beta(6, 4, n_samples)
            }
            
            # Create target variable (revenue) based on features
            revenue = (
                data['average_subscription_value'] * 
                (1 + data['account_age_days'] / 365 * 0.15) *  # Growth over time
                (1 + data['renewal_count'] * 0.08) *  # Loyalty bonus
                (1 + data['high_value_customer'] * 0.4) *  # High value bonus
                (1 + data['customer_satisfaction'] * 0.12) *  # Satisfaction bonus
                (1 + data['payment_success_rate'] * 0.25) *  # Payment reliability
                (1 - data['churn_risk_score'] * 0.35) *  # Churn penalty
                (1 + data['trial_conversion'] * 0.2) *  # Trial conversion bonus
                (1 - data['support_tickets'] * 0.05) *  # Support ticket penalty
                (1 + data['feature_usage_score'] * 0.15) *  # Feature usage bonus
                (1 + np.random.normal(0, 0.08, n_samples))  # Random noise
            )
            
            data['revenue'] = np.maximum(revenue, 5)  # Ensure positive revenue
            
            df = pd.DataFrame(data)
            print(f"Generated {len(df)} synthetic subscription records")
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def _engineer_features(self, df):
        """Engineer features for subscription revenue forecasting"""
        try:
            # Time-based features
            df['account_age_months'] = df['account_age_days'] / 30
            df['subscription_age_months'] = df['subscription_duration_days'] / 30
            df['renewal_frequency'] = df['renewal_count'] / np.maximum(df['account_age_months'], 1)
            
            # Customer lifecycle features
            df['is_new_customer'] = (df['account_age_days'] < 30).astype(int)
            df['is_established_customer'] = (df['account_age_days'] > 365).astype(int)
            df['is_high_engagement'] = (df['feature_usage_score'] > 0.7).astype(int)
            
            # Revenue features
            df['revenue_per_month'] = df['average_subscription_value'] / 30
            df['revenue_growth_rate'] = df['renewal_count'] * 0.1
            df['potential_upsell'] = (df['high_value_customer'] * df['customer_satisfaction'] * 0.5).astype(int)
            
            # Risk features
            df['risk_score'] = (df['churn_risk_score'] * 100).round()
            df['payment_reliability'] = (df['payment_success_rate'] * 100).round()
            df['support_burden'] = (df['support_tickets'] > 3).astype(int)
            
            # Behavioral features
            df['weekend_activity'] = df['is_weekend'] * df['feature_usage_score']
            df['satisfaction_tier'] = pd.cut(df['customer_satisfaction'], 
                                           bins=[0, 2, 3, 4, 5], 
                                           labels=[0, 1, 2, 3]).astype(int)
            
            # Remove outliers
            Q1 = df['revenue'].quantile(0.25)
            Q3 = df['revenue'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df['revenue'] >= lower_bound) & (df['revenue'] <= upper_bound)]
            
            print(f"Feature engineering completed. Dataset shape: {df.shape}")
            return df
            
        except Exception as e:
            print(f"Feature engineering error: {e}")
            return df
    
    def prepare_features(self, df):
        """Prepare features for training"""
        try:
            # Define feature columns
            self.feature_columns = [
                'account_age_days', 'renewal_count', 'average_subscription_value',
                'high_value_customer', 'subscription_duration_days', 'is_weekend',
                'customer_satisfaction', 'payment_success_rate', 'churn_risk_score',
                'account_age_months', 'subscription_age_months', 'renewal_frequency',
                'is_new_customer', 'is_established_customer', 'is_high_engagement',
                'revenue_per_month', 'revenue_growth_rate', 'potential_upsell',
                'risk_score', 'payment_reliability', 'support_burden',
                'weekend_activity', 'satisfaction_tier'
            ]
            
            # Select features and target
            X = df[self.feature_columns]
            y = df['revenue']
            
            # Handle missing values
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())
            
            print(f"Prepared features: {X.shape[1]} features, {X.shape[0]} samples")
            return X, y
            
        except Exception as e:
            print(f"Feature preparation error: {e}")
            return None, None
    
    def train_ensemble_models(self, X, y, final_features):
        """Train ensemble of models for subscription revenue forecasting"""
        try:
            # Split data with time series consideration
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Scale features
            self.scaler = RobustScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Initialize models (without LightGBM)
            self.models = {
                'xgboost': XGBRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                ),
                'random_forest': RandomForestRegressor(
                    n_estimators=300,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=42
                ),
                'elastic_net': ElasticNet(
                    alpha=0.1,
                    l1_ratio=0.7,
                    max_iter=2000,
                    random_state=42
                ),
                'ridge': Ridge(
                    alpha=1.0,
    random_state=42
)
            }
            
            # Train models
            model_scores = {}
            for name, model in self.models.items():
                print(f"Training {name}...")
                
model.fit(X_train_scaled, y_train)

                # Evaluate
y_pred = model.predict(X_test_scaled)
                mae = mean_absolute_error(y_test, y_pred)
                model_scores[name] = mae
                print(f"   {name} MAE: {mae:.2f}")
            
            # Create ensemble
            ensemble_models = [
                ('xgboost', self.models['xgboost']),
                ('random_forest', self.models['random_forest']),
                ('gradient_boosting', self.models['gradient_boosting']),
                ('ridge', self.models['ridge'])
            ]
            
            self.models['ensemble'] = VotingRegressor(ensemble_models)
            self.models['ensemble'].fit(X_train_scaled, y_train)
            
            # Evaluate ensemble
            y_pred_ensemble = self.models['ensemble'].predict(X_test_scaled)
            ensemble_mae = mean_absolute_error(y_test, y_pred_ensemble)
            ensemble_r2 = r2_score(y_test, y_pred_ensemble)
            
            print(f"Ensemble MAE: {ensemble_mae:.2f}, R²: {ensemble_r2:.3f}")
            
            self.results = {
                'model_scores': model_scores,
                'ensemble_mae': ensemble_mae,
                'ensemble_r2': ensemble_r2,
                'test_samples': len(X_test)
            }
            
            return self.results
            
        except Exception as e:
            print(f"Model training error: {e}")
            return None
    
    def _calculate_feature_importance(self):
        """Calculate feature importance across models"""
        try:
            importance_data = {}
            
            for name, model in self.models.items():
                if hasattr(model, 'feature_importances_'):
                    importance_data[name] = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importance_data[name] = np.abs(model.coef_)
            
            # Average importance across models
            if importance_data:
                avg_importance = np.mean(list(importance_data.values()), axis=0)
                feature_importance = dict(zip(self.feature_columns, avg_importance))
                return sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            return []
            
        except Exception as e:
            print(f"Feature importance calculation error: {e}")
            return []
    
    def evaluate_models(self, X, y):
        """Evaluate models with cross-validation"""
        try:
            X_scaled = self.scaler.transform(X)
            
            cv_scores = {}
            for name, model in self.models.items():
                if name != 'ensemble':  # Skip ensemble for CV
                    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_absolute_error')
                    cv_scores[name] = -scores.mean()
                    print(f"   {name} CV MAE: {-scores.mean():.2f} (±{scores.std() * 2:.2f})")
            
            return cv_scores
            
        except Exception as e:
            print(f"Model evaluation error: {e}")
            return {}
    
    def save_models_and_metadata(self, results, features):
        """Save trained models and metadata"""
        try:
            model_path = "/src/data/models"
            os.makedirs(model_path, exist_ok=True)
            
            # Save ensemble model
            ensemble_path = os.path.join(model_path, "subscription_ensemble_model.pkl")
            joblib.dump(self.models['ensemble'], ensemble_path)
            print(f"Saved ensemble model: {ensemble_path}")
            
            # Save scaler
            scaler_path = os.path.join(model_path, "subscription_revenue_scaler.pkl")
            joblib.dump(self.scaler, scaler_path)
            print(f"Saved scaler: {scaler_path}")
            
            # Save feature importance
            feature_importance = self._calculate_feature_importance()
            importance_path = os.path.join(model_path, "subscription_feature_importance.json")
            with open(importance_path, 'w') as f:
                json.dump(dict(feature_importance[:10]), f, indent=2)
            print(f"Saved feature importance: {importance_path}")
            
            # Save metadata
metadata = {
                'feature_columns': self.feature_columns,
                'model_scores': results['model_scores'],
                'ensemble_mae': results['ensemble_mae'],
                'ensemble_r2': results['ensemble_r2'],
                'test_samples': results['test_samples'],
                'feature_importance': dict(feature_importance[:10]),
                'created_at': datetime.datetime.now().isoformat()
            }
            
            metadata_path = os.path.join(model_path, "subscription_revenue_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Saved metadata: {metadata_path}")
            
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def create_visualizations(self, X, y):
        """Create visualization plots"""
        try:
            X_scaled = self.scaler.transform(X)
            y_pred = self.models['ensemble'].predict(X_scaled)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Actual vs Predicted
            axes[0, 0].scatter(y, y_pred, alpha=0.6)
            axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
            axes[0, 0].set_xlabel('Actual Revenue')
            axes[0, 0].set_ylabel('Predicted Revenue')
            axes[0, 0].set_title('Actual vs Predicted Revenue')
            
            # Residuals
            residuals = y - y_pred
            axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
            axes[0, 1].axhline(y=0, color='r', linestyle='--')
            axes[0, 1].set_xlabel('Predicted Revenue')
            axes[0, 1].set_ylabel('Residuals')
            axes[0, 1].set_title('Residuals Plot')
            
            # Feature importance
            feature_importance = self._calculate_feature_importance()
            if feature_importance:
                top_features = feature_importance[:10]
                features, importance = zip(*top_features)
                axes[1, 0].barh(range(len(features)), importance)
                axes[1, 0].set_yticks(range(len(features)))
                axes[1, 0].set_yticklabels(features)
                axes[1, 0].set_xlabel('Importance')
                axes[1, 0].set_title('Top 10 Feature Importance')
            
            # Prediction distribution
            axes[1, 1].hist(y_pred, bins=30, alpha=0.7, label='Predicted')
            axes[1, 1].hist(y, bins=30, alpha=0.7, label='Actual')
            axes[1, 1].set_xlabel('Revenue')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Revenue Distribution')
            axes[1, 1].legend()
            
            plt.tight_layout()
            
            # Save plot
            plot_path = "/src/data/models/subscription_forecast_plots.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization: {plot_path}")
            
            plt.close()
            
        except Exception as e:
            print(f"Visualization error: {e}")

def train_enhanced_subscription_forecaster():
    """Train enhanced subscription revenue forecaster"""
    print("Starting Enhanced Subscription Revenue Forecasting Training...")
    
    try:
        # Initialize forecaster
        forecaster = EnhancedSubscriptionForecaster()
        
        # Load and preprocess data
        df = forecaster.load_and_preprocess_data()
        if df is None:
            return None
        
        # Engineer features
        df = forecaster._engineer_features(df)
        
        # Prepare features
        X, y = forecaster.prepare_features(df)
        if X is None or y is None:
            return None
        
        # Train ensemble models
        results = forecaster.train_ensemble_models(X, y, X.columns)
        if results is None:
            return None
        
        # Evaluate models
        print("🔄 Evaluating models...")
        cv_scores = forecaster.evaluate_models(X, y)
        
        # Create visualizations
        print("🔄 Creating visualizations...")
        forecaster.create_visualizations(X, y)
        
        # Save everything
        forecaster.save_models_and_metadata(results, X.columns)
        
        print("🎉 Enhanced subscription forecasting training completed!")
        return forecaster.models['ensemble'], forecaster.scaler
        
    except Exception as e:
        print(f"Training error: {e}")
        return None

def train():
    """Training function for subscription revenue forecasting"""
    return train_enhanced_subscription_forecaster()

if __name__ == "__main__":
    train_enhanced_subscription_forecaster()