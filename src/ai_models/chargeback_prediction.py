import os
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_score, TimeSeriesSplit
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.ensemble import IsolationForest, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.spatial.distance import pdist, squareform
import networkx as nx


class ChargebackPredictionModel:
    """Predicts chargeback risk for payment transactions"""
    
    def __init__(self):
        self.common_domains = {
            "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com",
            "aol.com", "protonmail.com", "zoho.com", "mail.com", "gmx.com"
        }
        self.disposable_domains = {
            "mailinator", "10minutemail", "guerrillamail", "tempmail", "yopmail", 
            "trashmail", "temp-mail", "throwaway"
        }
        
    def ensure_cols(self, df: pd.DataFrame, cols, default=np.nan, astype=None):
        """Ensure columns exist in DataFrame"""
        for c in cols:
            if c not in df.columns:
                df[c] = default
            if astype is not None:
                df[c] = df[c].astype(astype)
        return df

    def rolling_count_seconds(self, times: pd.Series, window_s: int) -> pd.Series:
        """For a monotonically increasing datetime Series, return count of prior rows in sliding window."""
        times = pd.to_datetime(times)
        idx = times.index
        out = np.zeros(len(times), dtype=np.int32)
        j = 0
        for i in range(len(times)):
            t_hi = times.iloc[i]
            t_lo = t_hi - pd.Timedelta(seconds=window_s)
            while j < i and times.iloc[j] < t_lo:
                j += 1
            out[i] = i - j
        return pd.Series(out, index=idx)

    def domain_risk(self, email: str) -> int:
        """Calculate email domain risk score"""
        dom = str(email).split("@")[-1].lower() if "@" in str(email) else "unknown"
        return 0 if dom in self.common_domains else (2 if any(d in dom for d in self.disposable_domains) else 1)

    def create_temporal_features(self, df):
        """Extract time-based features from transaction data"""
        df = df.copy()
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        # Extract time components
        df['hour'] = df['created_at'].dt.hour
        df['day_of_week'] = df['created_at'].dt.dayofweek
        df['day_of_month'] = df['created_at'].dt.day
        df['month'] = df['created_at'].dt.month
        df['quarter'] = df['created_at'].dt.quarter
        df['is_weekend'] = (df['created_at'].dt.weekday >= 5).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        # Convert time features to cyclical format
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Business calendar features
        df['is_holiday_season'] = df['month'].isin([11, 12]).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 25).astype(int)
        df['is_quarter_end'] = df['month'].isin([3, 6, 9, 12]).astype(int)
        
        return df

    def create_customer_lifecycle_features(self, df):
        """Extract customer relationship and behavior patterns"""
        df = df.copy()
        df = df.sort_values(['email', 'created_at']).reset_index(drop=True)
        
        # Customer tenure
        first_transaction = df.groupby('email')['created_at'].transform('min')
        df['customer_tenure_days'] = (df['created_at'] - first_transaction).dt.days
        df['is_new_customer'] = (df['customer_tenure_days'] <= 30).astype(int)
        df['is_early_customer'] = (df['customer_tenure_days'] <= 90).astype(int)
        
        # Transaction sequence
        df['transaction_sequence'] = df.groupby('email').cumcount() + 1
        df['is_first_transaction'] = (df['transaction_sequence'] == 1).astype(int)
        
        # Customer velocity patterns
        df['transactions_per_day'] = df['transaction_sequence'] / (df['customer_tenure_days'] + 1)
        df['avg_days_between_transactions'] = df['customer_tenure_days'] / (df['transaction_sequence'] + 1)
        
        return df

    def create_velocity_features(self, df):
        """Calculate transaction frequency patterns over time windows"""
        df = df.copy()
        df = df.sort_values(['email', 'created_at']).reset_index(drop=True)
        
        # Rolling windows for different time periods
        windows = {
            '1h': 3600, '6h': 21600, '24h': 86400, '7d': 604800, '30d': 2592000
        }
        
        for window_name, window_sec in windows.items():
            df[f'transactions_{window_name}'] = 0
            df[f'amount_sum_{window_name}'] = 0.0
            df[f'amount_avg_{window_name}'] = 0.0
            df[f'amount_max_{window_name}'] = 0.0
            df[f'chargebacks_{window_name}'] = 0
            df[f'refunds_{window_name}'] = 0
            
            for email in df['email'].unique():
                email_mask = df['email'] == email
                email_data = df[email_mask].copy()
                
                if len(email_data) > 1:
                    for i, (idx, row) in enumerate(email_data.iterrows()):
                        current_time = row['created_at']
                        window_start = current_time - pd.Timedelta(seconds=window_sec)
                        
                        # Count transactions in window
                        window_mask = (email_data['created_at'] >= window_start) & \
                                    (email_data['created_at'] < current_time)
                        window_data = email_data[window_mask]
                        
                        df.loc[idx, f'transactions_{window_name}'] = len(window_data)
                        if len(window_data) > 0:
                            df.loc[idx, f'amount_sum_{window_name}'] = window_data['amount'].sum()
                            df.loc[idx, f'amount_avg_{window_name}'] = window_data['amount'].mean()
                            df.loc[idx, f'amount_max_{window_name}'] = window_data['amount'].max()
                            df.loc[idx, f'chargebacks_{window_name}'] = window_data['disputed'].sum()
                            df.loc[idx, f'refunds_{window_name}'] = window_data['refunded'].sum()
        
        return df

    def create_behavioral_features(self, df):
        """Extract behavioral indicators from transaction data"""
        df = df.copy()
        
        # Email domain analysis
        df['email_domain'] = df['email'].str.split('@').str[1].str.lower()
        df['email_domain_risk'] = df['email'].apply(self.domain_risk)
        df['email_length'] = df['email'].str.len()
        df['email_has_numbers'] = df['email'].str.contains(r'\d').astype(int)
        df['email_has_special_chars'] = df['email'].str.contains(r'[._-]').astype(int)
        
        # Amount-based features
        df['amount_log'] = np.log1p(df['amount'])
        df['amount_sqrt'] = np.sqrt(df['amount'])
        df['amount_reciprocal'] = 1 / (df['amount'] + 1)
        
        # Country and location features
        df['country_mismatch'] = (df['card_country'] != df['billing_address_country']).astype(int)
        df['is_cross_border'] = df['country_mismatch']
        
        # Device and IP features
        df['ip_is_private'] = df['ip_address'].str.startswith(('10.', '172.', '192.168.')).astype(int)
        df['fingerprint_length'] = df['fingerprint'].str.len()
        
        # Customer history features
        df = df.sort_values(['email', 'created_at']).reset_index(drop=True)
        
        # Past refunds and chargebacks
        df['past_refund_count'] = df.groupby('email')['refunded'].shift().fillna(0).groupby(df['email']).cumsum()
        df['past_chargeback_count'] = df.groupby('email')['disputed'].shift().fillna(0).groupby(df['email']).cumsum()
        df['past_transaction_count'] = df.groupby('email').cumcount()
        
        # Ratios
        df['refund_ratio'] = df['past_refund_count'] / (df['past_transaction_count'] + 1)
        df['chargeback_ratio'] = df['past_chargeback_count'] / (df['past_transaction_count'] + 1)
        
        # Amount consistency
        df['past_avg_amount'] = df.groupby('email')['amount'].shift().groupby(df['email']).expanding().mean().reset_index(level=0, drop=True).fillna(0)
        df['amount_deviation'] = (df['amount'] - df['past_avg_amount']).abs()
        df['amount_deviation_ratio'] = df['amount_deviation'] / (df['past_avg_amount'] + 1)
        
        return df

    def create_network_features(self, df):
        """Analyze device and IP sharing patterns"""
        df = df.copy()
        
        # IP sharing network
        ip_emails = df.groupby('ip_address')['email'].apply(set).to_dict()
        df['ip_email_connections'] = df['ip_address'].map(
            lambda x: len(ip_emails.get(x, set()))
        )
        
        # Email sharing network
        email_ips = df.groupby('email')['ip_address'].apply(set).to_dict()
        df['email_ip_connections'] = df['email'].map(
            lambda x: len(email_ips.get(x, set()))
        )
        
        # Device fingerprint sharing
        fp_emails = df.groupby('fingerprint')['email'].apply(set).to_dict()
        df['fp_email_connections'] = df['fingerprint'].map(
            lambda x: len(fp_emails.get(x, set()))
        )
        
        # Network risk scores
        ip_chargeback_rates = df.groupby('ip_address')['disputed'].mean().to_dict()
        fp_chargeback_rates = df.groupby('fingerprint')['disputed'].mean().to_dict()
        
        df['ip_chargeback_rate'] = df['ip_address'].map(ip_chargeback_rates).fillna(0)
        df['fp_chargeback_rate'] = df['fingerprint'].map(fp_chargeback_rates).fillna(0)
        
        return df

    def create_anomaly_features(self, df):
        """Detect statistical outliers in transaction patterns"""
        df = df.copy()
        
        # Amount anomalies
        df['amount_zscore'] = np.abs(stats.zscore(df['amount']))
        df['amount_iqr'] = self._iqr_outlier_score(df['amount'])
        
        # Time anomalies
        df['hour_anomaly'] = np.abs(df['hour'] - df['hour'].median())
        
        # Country anomalies (frequency-based)
        country_counts = df['card_country'].value_counts()
        df['country_frequency'] = df['card_country'].map(country_counts)
        df['country_anomaly'] = 1 / (df['country_frequency'] + 1)
        
        # Customer behavior anomalies
        df['refund_ratio_anomaly'] = self._iqr_outlier_score(df['refund_ratio'])
        df['chargeback_ratio_anomaly'] = self._iqr_outlier_score(df['chargeback_ratio'])
        
        return df

    def _iqr_outlier_score(self, series):
        """Calculate IQR-based outlier score"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        return np.abs(series - series.median()) / (IQR + 1e-6)

    def prepare_data(self, df):
        """Clean and prepare transaction data for model training"""
        print("Processing transaction data for chargeback prediction...")
        
        # Basic cleaning
        df = df.dropna(subset=['created_at']).copy()
        df['created_at'] = pd.to_datetime(df['created_at'])
        df = df.sort_values('created_at').reset_index(drop=True)
        
        # Fill missing values
        df['email'] = df['email'].fillna('unknown@example.com')
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)
        df['risk_score'] = pd.to_numeric(df['risk_score'], errors='coerce').fillna(0.0)
        df['disputed'] = pd.to_numeric(df['disputed'], errors='coerce').fillna(0).astype(int)
        df['refunded'] = pd.to_numeric(df['refunded'], errors='coerce').fillna(0).astype(int)
        df['card_country'] = df['card_country'].fillna('UNK')
        df['billing_address_country'] = df['billing_address_country'].fillna('UNK')
        df['ip_address'] = df['ip_address'].fillna('0.0.0.0')
        df['fingerprint'] = df['fingerprint'].fillna('unknown')
        
        # Create all feature sets
        df = self.create_temporal_features(df)
        df = self.create_customer_lifecycle_features(df)
        df = self.create_velocity_features(df)
        df = self.create_behavioral_features(df)
        df = self.create_network_features(df)
        df = self.create_anomaly_features(df)
        
        print(f"Feature engineering complete. Created {len(df.columns)} features.")
        return df

    def create_ensemble_models(self):
        """Initialize multiple ML models for ensemble prediction"""
        
        # Base models with different characteristics
        models = {
            'xgboost': XGBClassifier(
                n_estimators=1200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="hist",
                random_state=42,
                n_jobs=-1
            ),
            'lightgbm': LGBMClassifier(
                n_estimators=1200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=600,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
            'logistic_regression': LogisticRegression(
                C=0.1,
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
        }
        
        return models

    def train_ensemble(self, X_train, y_train, X_val, y_val):
        """Train multiple models and combine their predictions"""
        
        models = self.create_ensemble_models()
        trained_models = {}
        
        # Different sampling strategies for different models
        sampling_strategies = {
            'xgboost': SMOTE(random_state=42),
            'lightgbm': BorderlineSMOTE(random_state=42),
            'random_forest': EditedNearestNeighbours(),
            'logistic_regression': SMOTE(random_state=42)
        }
        
        for name, model in models.items():
            print(f"🔄 Training {name}...")
            
            # Create pipeline with sampling and scaling
            pipeline = ImbPipeline([
                ('sampling', sampling_strategies[name]),
                ('scaler', RobustScaler()),
                ('model', model)
            ])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Validate
            y_val_pred = pipeline.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, y_val_pred)
            val_pr_auc = average_precision_score(y_val, y_val_pred)
            
            print(f"✅ {name} - Val AUC: {val_auc:.4f}, Val PR-AUC: {val_pr_auc:.4f}")
            
            trained_models[name] = pipeline
        
        # Create voting classifier
        voting_models = [(name, model) for name, model in trained_models.items()]
        ensemble = VotingClassifier(
            estimators=voting_models,
            voting='soft'
        )
        
        # Train ensemble on full training data
        ensemble.fit(X_train, y_train)
        
        # Final validation
        y_val_ensemble = ensemble.predict_proba(X_val)[:, 1]
        ensemble_auc = roc_auc_score(y_val, y_val_ensemble)
        ensemble_pr_auc = average_precision_score(y_val, y_val_ensemble)
        
        print(f"🎯 Ensemble - Val AUC: {ensemble_auc:.4f}, Val PR-AUC: {ensemble_pr_auc:.4f}")
        
        return ensemble, trained_models

    def train(self):
        """Main training function"""
        
        # Load data
        load_dotenv()
        MONGO_URI = os.getenv("MONGO_URI")
        if not MONGO_URI:
            raise ValueError("Missing MONGO_URI in environment.")
        
        client = MongoClient(MONGO_URI)
        db = client["payment_intelligence"]
        
        print("📊 Loading transaction data...")
        df = pd.DataFrame(db["transactions"].find())
        
        if df.empty:
            raise ValueError("No transactions found.")
        
        # Prepare data
        df = self.prepare_data(df)
        
        # Temporal split (more realistic for chargeback prediction)
        split_date = df['created_at'].quantile(0.8)
        train_mask = df['created_at'] <= split_date
        test_mask = df['created_at'] > split_date
        
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()
        
        # Further split training data
        val_split_date = train_df['created_at'].quantile(0.9)
        train_mask_val = train_df['created_at'] <= val_split_date
        val_mask = train_df['created_at'] > val_split_date
        
        X_train = train_df[train_mask_val].drop(['disputed', 'created_at'], axis=1)
        y_train = train_df[train_mask_val]['disputed']
        X_val = train_df[val_mask].drop(['disputed', 'created_at'], axis=1)
        y_val = train_df[val_mask]['disputed']
        X_test = test_df.drop(['disputed', 'created_at'], axis=1)
        y_test = test_df['disputed']
        
        print(f"📈 Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        print(f"🎯 Chargeback rate - Train: {y_train.mean():.3f}, Val: {y_val.mean():.3f}, Test: {y_test.mean():.3f}")
        
        # Convert to numeric and fill NaN
        X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
        X_val = X_val.apply(pd.to_numeric, errors='coerce').fillna(0)
        X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Train ensemble
        ensemble, individual_models = self.train_ensemble(X_train, y_train, X_val, y_val)
        
        # Final evaluation
        print("\n🎯 Final Model Evaluation:")
        y_test_pred = ensemble.predict_proba(X_test)[:, 1]
        
        # Find optimal threshold
        precision, recall, thresholds = precision_recall_curve(y_test, y_test_pred)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
        optimal_threshold = thresholds[np.argmax(f1_scores)]
        
        y_test_pred_binary = (y_test_pred >= optimal_threshold).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_test_pred_binary),
            'precision': precision_score(y_test, y_test_pred_binary, zero_division=0),
            'recall': recall_score(y_test, y_test_pred_binary, zero_division=0),
            'f1': f1_score(y_test, y_test_pred_binary, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_test_pred),
            'pr_auc': average_precision_score(y_test, y_test_pred),
            'optimal_threshold': optimal_threshold
        }
        
        print("\n📊 Test Set Performance:")
        for metric, value in metrics.items():
            print(f"{metric:>15}: {value:.4f}")
        
        # Feature importance analysis
        self._analyze_feature_importance(ensemble, X_train.columns, X_train)
        
        # Save models and metadata
        self._save_models(ensemble, individual_models, list(X_train.columns), metrics)
        
        return ensemble, metrics

    def _analyze_feature_importance(self, ensemble, features, X_train):
        """Analyze and visualize feature importance"""
        
        # Get feature importance from ensemble models
        importances = {}
        for name, model in ensemble.named_estimators_.items():
            if hasattr(model.named_steps['model'], 'feature_importances_'):
                importances[name] = model.named_steps['model'].feature_importances_
            elif hasattr(model.named_steps['model'], 'coef_'):
                importances[name] = np.abs(model.named_steps['model'].coef_[0])
        
        # Average importance across models
        if importances:
            avg_importance = np.mean(list(importances.values()), axis=0)
            feature_importance = pd.DataFrame({
                'feature': features,
                'importance': avg_importance
            }).sort_values('importance', ascending=False)
            
            # Plot top features
            plt.figure(figsize=(12, 8))
            top_features = feature_importance.head(20)
            sns.barplot(data=top_features, y='feature', x='importance')
            plt.title('Top 20 Feature Importances (Ensemble Average)')
            plt.tight_layout()
            
            os.makedirs("/src/data/models", exist_ok=True)
            plt.savefig("/src/data/models/chargeback_prediction_feature_importance.png", dpi=300, bbox_inches='tight')
            plt.show()

    def _save_models(self, ensemble, individual_models, features, metrics):
        """Save trained models and metadata"""
        
        os.makedirs("/src/data/models", exist_ok=True)
        
        # Save ensemble model
        joblib.dump(ensemble, "/src/data/models/chargeback_prediction_pipeline.pkl")
        
        # Save individual models
        for name, model in individual_models.items():
            joblib.dump(model, f"/src/data/models/chargeback_prediction_{name}.pkl")
        
        # Save metadata
        metadata = {
            "model_version": "2.0.0",
            "model_type": "ensemble_chargeback_prediction",
            "created_at": datetime.now().isoformat(),
            "features_used": features,
            "metrics": metrics,
            "models_in_ensemble": list(individual_models.keys()),
            "ensemble_method": "voting_classifier_soft"
        }
        
        with open("/src/data/models/chargeback_prediction_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
        
        print("✅ Chargeback prediction models saved successfully!")


def train():
    """Train chargeback prediction model"""
    predictor = ChargebackPredictionModel()
    return predictor.train()


if __name__ == "__main__":
    train()