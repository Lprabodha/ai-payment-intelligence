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
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, PolynomialFeatures
from sklearn.ensemble import IsolationForest, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel, RFECV
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from scipy import stats
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import optuna
from optuna.samplers import TPESampler


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

    def create_interaction_features(self, df):
        """Create interaction features for better chargeback pattern detection"""
        df = df.copy()
        
        # Amount × Time interactions
        df['amount_hour_interaction'] = df['amount'] * df['hour']
        df['amount_weekend_interaction'] = df['amount'] * df['is_weekend']
        df['amount_month_interaction'] = df['amount'] * df['month']
        
        # Customer lifecycle × Amount
        df['tenure_amount_ratio'] = df['customer_tenure_days'] * df['amount_log']
        df['first_txn_amount_risk'] = df['is_first_transaction'] * df['amount']
        
        # Velocity × Amount interactions
        df['velocity_amount_ratio'] = df['transactions_24h'] * df['amount_log']
        df['chargeback_amount_ratio'] = df['chargebacks_30d'] * df['amount']
        df['refund_amount_ratio'] = df['refunds_30d'] * df['amount']
        
        # Risk × Behavioral interactions
        df['risk_country_mismatch'] = df['country_mismatch'] * df['email_domain_risk']
        df['risk_new_customer'] = df['is_new_customer'] * df['email_domain_risk']
        
        # Network × Amount interactions
        df['ip_connections_amount'] = df['ip_email_connections'] * df['amount_log']
        df['fp_connections_amount'] = df['fp_email_connections'] * df['amount_log']
        
        # Anomaly combinations
        df['combined_anomaly_score'] = (
            df['amount_zscore'] + df['hour_anomaly'] + df['country_anomaly'] + 
            df['refund_ratio_anomaly'] + df['chargeback_ratio_anomaly']
        ) / 5
        
        # Historical pattern interactions
        df['deviation_per_transaction'] = df['amount_deviation_ratio'] * df['past_transaction_count']
        df['chargeback_refund_correlation'] = df['past_chargeback_count'] * df['past_refund_count']
        
        return df

    def create_graph_features(self, df):
        """Create graph-based features using network analysis"""
        df = df.copy()
        
        # Build email-IP-fingerprint tripartite graph
        G = nx.Graph()
        
        for idx, row in df.iterrows():
            email = row['email']
            ip = row['ip_address']
            fp = row['fingerprint']
            
            # Add edges with weights based on chargeback history
            weight = 1 + row.get('disputed', 0) * 5  # Higher weight for disputed transactions
            G.add_edge(f"email_{email}", f"ip_{ip}", weight=weight)
            G.add_edge(f"email_{email}", f"fp_{fp}", weight=weight)
            G.add_edge(f"ip_{ip}", f"fp_{fp}", weight=weight)
        
        # Calculate centrality measures
        try:
            degree_centrality = nx.degree_centrality(G)
            betweenness = nx.betweenness_centrality(G, k=min(100, len(G.nodes())))
            
            df['email_degree_centrality'] = df['email'].apply(
                lambda x: degree_centrality.get(f"email_{x}", 0)
            )
            df['ip_degree_centrality'] = df['ip_address'].apply(
                lambda x: degree_centrality.get(f"ip_{x}", 0)
            )
            df['fp_degree_centrality'] = df['fingerprint'].apply(
                lambda x: degree_centrality.get(f"fp_{x}", 0)
            )
            df['email_betweenness'] = df['email'].apply(
                lambda x: betweenness.get(f"email_{x}", 0)
            )
            
            # Community detection and clustering
            df['network_clustering_coef'] = 0.0
            for idx, row in df.iterrows():
                email_node = f"email_{row['email']}"
                if email_node in G:
                    neighbors = list(G.neighbors(email_node))
                    if len(neighbors) > 1:
                        subgraph = G.subgraph(neighbors)
                        df.loc[idx, 'network_clustering_coef'] = nx.density(subgraph)
        except:
            # If graph analysis fails, set defaults
            df['email_degree_centrality'] = 0.0
            df['ip_degree_centrality'] = 0.0
            df['fp_degree_centrality'] = 0.0
            df['email_betweenness'] = 0.0
            df['network_clustering_coef'] = 0.0
        
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
        print("  Creating temporal features...")
        df = self.create_temporal_features(df)
        print("  Creating customer lifecycle features...")
        df = self.create_customer_lifecycle_features(df)
        print("  Creating velocity features...")
        df = self.create_velocity_features(df)
        print("  Creating behavioral features...")
        df = self.create_behavioral_features(df)
        print("  Creating network features...")
        df = self.create_network_features(df)
        print("  Creating anomaly features...")
        df = self.create_anomaly_features(df)
        print("  Creating interaction features...")
        df = self.create_interaction_features(df)
        print("  Creating graph-based features...")
        df = self.create_graph_features(df)
        
        print(f"[SUCCESS] Feature engineering complete. Created {len(df.columns)} features.")
        return df

    def create_ensemble_models(self, use_optimized=False, optimized_params=None):
        """Initialize multiple ML models for ensemble prediction"""
        
        if optimized_params is None:
            optimized_params = {}
        
        # Base models with tuned or default parameters
        models = {
            'xgboost': XGBClassifier(
                **optimized_params.get('xgboost', {
                    'n_estimators': 1400,
                    'max_depth': 7,
                    'learning_rate': 0.035,
                    'subsample': 0.85,
                    'colsample_bytree': 0.85,
                    'min_child_weight': 3,
                    'gamma': 0.1,
                    'tree_method': 'hist',
                    'random_state': 42,
                    'n_jobs': -1
                })
            ),
            'lightgbm': LGBMClassifier(
                **optimized_params.get('lightgbm', {
                    'n_estimators': 1400,
                    'max_depth': 7,
                    'learning_rate': 0.035,
                    'subsample': 0.85,
                    'colsample_bytree': 0.85,
                    'num_leaves': 45,
                    'min_child_samples': 25,
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbose': -1
                })
            ),
            'catboost': CatBoostClassifier(
                **optimized_params.get('catboost', {
                    'iterations': 1400,
                    'depth': 7,
                    'learning_rate': 0.035,
                    'l2_leaf_reg': 3,
                    'border_count': 128,
                    'random_seed': 42,
                    'verbose': False
                })
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=700,
                max_depth=13,
                min_samples_split=9,
                min_samples_leaf=4,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=700,
                max_depth=6,
                learning_rate=0.045,
                subsample=0.85,
                random_state=42
            )
        }
        
        return models

    def train_ensemble(self, X_train, y_train, X_val, y_val, use_stacking=True):
        """Train multiple models and combine their predictions using stacking ensemble"""
        
        models = self.create_ensemble_models()
        trained_models = {}
        
        # Different sampling strategies for different models
        sampling_strategies = {
            'xgboost': SMOTETomek(random_state=42),
            'lightgbm': ADASYN(random_state=42),
            'catboost': SMOTE(random_state=42),
            'random_forest': BorderlineSMOTE(random_state=42),
            'gradient_boosting': SMOTE(random_state=42)
        }
        
        # Train base models
        base_estimators = []
        
        for name, model in models.items():
            print(f"Training {name}...")
            
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
            
            print(f"[COMPLETE] {name} - Val AUC: {val_auc:.4f}, Val PR-AUC: {val_pr_auc:.4f}")
            
            trained_models[name] = pipeline
            base_estimators.append((name, pipeline))
        
        if use_stacking:
            # Create stacking classifier with logistic regression meta-learner
            print("\nTraining stacking ensemble...")
            
            # Meta-learner: Logistic Regression with L2 regularization
            meta_learner = LogisticRegression(
                C=0.5,
                max_iter=2000,
                random_state=42,
                n_jobs=-1,
                solver='saga'
            )
            
            # Create stacking ensemble
            ensemble = StackingClassifier(
                estimators=base_estimators,
                final_estimator=meta_learner,
                cv=5,
                stack_method='predict_proba',
                n_jobs=-1,
                verbose=0
            )
            
            # Train stacking ensemble
            ensemble.fit(X_train, y_train)
            
            # Final validation
            y_val_ensemble = ensemble.predict_proba(X_val)[:, 1]
            ensemble_auc = roc_auc_score(y_val, y_val_ensemble)
            ensemble_pr_auc = average_precision_score(y_val, y_val_ensemble)
            
            print(f"[ENSEMBLE] Stacking Ensemble - Val AUC: {ensemble_auc:.4f}, Val PR-AUC: {ensemble_pr_auc:.4f}")
        else:
            # Fallback to voting classifier
            print("\nTraining voting ensemble...")
            ensemble = VotingClassifier(
                estimators=base_estimators,
                voting='soft',
                n_jobs=-1
            )
            
            ensemble.fit(X_train, y_train)
            
            y_val_ensemble = ensemble.predict_proba(X_val)[:, 1]
            ensemble_auc = roc_auc_score(y_val, y_val_ensemble)
            ensemble_pr_auc = average_precision_score(y_val, y_val_ensemble)
            
            print(f"[ENSEMBLE] Voting Ensemble - Val AUC: {ensemble_auc:.4f}, Val PR-AUC: {ensemble_pr_auc:.4f}")
        
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
        """Analyze and visualize feature importance using multiple methods"""
        
        print("\n[INFO] Analyzing feature importance...")
        
        # Get feature importance from ensemble models
        importances = {}
        
        # For stacking ensemble
        if hasattr(ensemble, 'estimators_'):
            for name, model in zip([e[0] for e in ensemble.estimators], ensemble.estimators_):
                if hasattr(model.named_steps['model'], 'feature_importances_'):
                    importances[name] = model.named_steps['model'].feature_importances_
                elif hasattr(model.named_steps['model'], 'coef_'):
                    importances[name] = np.abs(model.named_steps['model'].coef_[0])
        # For voting ensemble
        elif hasattr(ensemble, 'named_estimators_'):
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
            
            # Plot traditional feature importance
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
            
            # Top 20 features
            top_features = feature_importance.head(20)
            sns.barplot(data=top_features, y='feature', x='importance', ax=ax1)
            ax1.set_title('Top 20 Feature Importances (Ensemble Average)')
            ax1.set_xlabel('Importance Score')
            
            # Individual model comparisons
            if len(importances) > 1:
                top_10_features = feature_importance.head(10)['feature'].tolist()
                importance_df = pd.DataFrame(importances, index=features)
                importance_df.loc[top_10_features].T.plot(kind='bar', ax=ax2)
                ax2.set_title('Top 10 Features by Model')
                ax2.set_xlabel('Model')
                ax2.set_ylabel('Importance')
                ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            os.makedirs("/src/data/models", exist_ok=True)
            plt.savefig("/src/data/models/chargeback_prediction_feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # SHAP explainability analysis
        try:
            print("  Calculating SHAP values...")
            
            # Sample data for SHAP (use subset for performance)
            sample_size = min(500, len(X_train))
            X_sample = X_train.sample(n=sample_size, random_state=42) if len(X_train) > sample_size else X_train
            
            # Get one of the tree-based models for SHAP
            tree_model = None
            for name in ['xgboost', 'lightgbm', 'catboost']:
                if hasattr(ensemble, 'estimators_'):
                    for est_name, est in zip([e[0] for e in ensemble.estimators], ensemble.estimators_):
                        if name in est_name:
                            tree_model = est.named_steps['model']
                            break
                elif hasattr(ensemble, 'named_estimators_') and name in ensemble.named_estimators_:
                    tree_model = ensemble.named_estimators_[name].named_steps['model']
                    break
                if tree_model:
                    break
            
            if tree_model is not None:
                # Create SHAP explainer
                explainer = shap.TreeExplainer(tree_model)
                
                # Calculate SHAP values
                shap_values = explainer.shap_values(X_sample)
                
                # Handle binary classification output
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Use positive class
                
                # Plot SHAP summary
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=features, show=False, max_display=20)
                plt.title('SHAP Feature Importance Summary (Chargeback Prediction)')
                plt.tight_layout()
                plt.savefig("/src/data/models/chargeback_prediction_shap_summary.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                # Plot SHAP bar plot
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=features, plot_type='bar', show=False, max_display=20)
                plt.title('SHAP Feature Importance (Mean Absolute SHAP Values)')
                plt.tight_layout()
                plt.savefig("/src/data/models/chargeback_prediction_shap_bar.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                print("  [SUCCESS] SHAP analysis complete!")
                
        except Exception as e:
            print(f"  [WARNING] SHAP analysis failed: {e}")
            print("  Continuing without SHAP visualizations...")

    def _save_models(self, ensemble, individual_models, features, metrics):
        """Save trained models and metadata"""
        
        os.makedirs("/src/data/models", exist_ok=True)
        
        # Determine ensemble type
        ensemble_type = "stacking_classifier" if isinstance(ensemble, StackingClassifier) else "voting_classifier_soft"
        
        # Save ensemble model
        joblib.dump(ensemble, "/src/data/models/chargeback_prediction_pipeline.pkl")
        print("  Saved ensemble model")
        
        # Save individual models
        for name, model in individual_models.items():
            joblib.dump(model, f"/src/data/models/chargeback_prediction_{name}.pkl")
        print(f"  Saved {len(individual_models)} individual models")
        
        # Save metadata
        metadata = {
            "model_version": "3.0.0",
            "model_type": "enhanced_chargeback_prediction_ensemble",
            "created_at": datetime.now().isoformat(),
            "features_used": features,
            "feature_count": len(features),
            "metrics": metrics,
            "models_in_ensemble": list(individual_models.keys()),
            "ensemble_method": ensemble_type,
            "enhancements": [
                "CatBoost integration",
                "Stacking ensemble with meta-learner",
                "Advanced feature engineering (interaction, graph-based, customer lifecycle)",
                "Multiple sampling strategies (SMOTETomek, ADASYN, BorderlineSMOTE)",
                "SHAP explainability",
                "Improved velocity, network, and chargeback-specific features"
            ],
            "training_info": {
                "use_stacking": isinstance(ensemble, StackingClassifier),
                "use_graph_features": True,
                "use_interaction_features": True,
                "use_lifecycle_features": True,
                "use_shap": True
            }
        }
        
        with open("/src/data/models/chargeback_prediction_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
        
        print("  Saved model metadata")
        print("\n[SUCCESS] Enhanced chargeback prediction models saved successfully!")


def train():
    """Train chargeback prediction model"""
    predictor = ChargebackPredictionModel()
    return predictor.train()


if __name__ == "__main__":
    train()