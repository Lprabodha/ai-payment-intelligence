import os, json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import enhanced components (fallback if not available)
try:
    from advanced_features import AdvancedFeatureEngine
    from ensemble_model import EnsembleModelManager, RulesEngine
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    print("Enhanced features not available. Using basic implementation.")
    ENHANCED_FEATURES_AVAILABLE = False


def ensure_cols(df: pd.DataFrame, cols, default=np.nan, astype=None):
    for c in cols:
        if c not in df.columns:
            df[c] = default
        if astype is not None:
            df[c] = df[c].astype(astype)
    return df

def rolling_count_seconds(times: pd.Series, window_s: int) -> pd.Series:
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

def domain_risk(email: str, common_domains: set) -> int:
    dom = str(email).split("@")[-1].lower() if "@" in str(email) else "unknown"
    return 0 if dom in common_domains else 1

def domain_class(email: str, common_domains: set, disposable_hints: set) -> int:
    dom = str(email).split("@")[-1].lower() if "@" in str(email) else "unknown"
    if any(k in dom for k in disposable_hints):
        return 2  # disposable
    if dom in common_domains:
        return 0  # free/major
    return 1      # corporate/other

# index-safe dominant country before
def dominant_country_before_indexsafe(df: pd.DataFrame) -> pd.Series:
    seen = []
    out_vals = []
    for val in df["card_country"]:
        out_vals.append(pd.Series(seen).mode().iloc[0] if len(seen) else "UNK")
        seen.append(val)
    return pd.Series(out_vals, index=df.index, dtype="object")

class EnhancedFraudDetection:
    """
    Enhanced fraud detection with advanced features and ensemble methods
    """
    
    def __init__(self, mongo_uri: str = None):
        self.mongo_uri = mongo_uri or os.getenv("MONGO_URI")
        if not self.mongo_uri:
            raise ValueError("Missing MONGO_URI in environment.")
        
        self.mongo = MongoClient(self.mongo_uri)
        self.db = self.mongo["payment_intelligence"]
        
        # Initialize enhanced components if available
        if ENHANCED_FEATURES_AVAILABLE:
            self.feature_engine = AdvancedFeatureEngine()
            self.ensemble_manager = EnsembleModelManager()
            self.rules_engine = RulesEngine()
        else:
            self.feature_engine = None
            self.ensemble_manager = None
            self.rules_engine = None
        
        # Model configuration
        self.model_config = {
            'version': '2.1.0',
            'use_enhanced_features': ENHANCED_FEATURES_AVAILABLE,
            'ensemble_enabled': ENHANCED_FEATURES_AVAILABLE,
            'real_time_features': True
        }
        
        # Enhanced feature groups
        self.feature_groups = {
            'basic': [
                'amount_log', 'hour', 'is_weekend', 'country_mismatch',
                'past_tx_count', 'past_disputes', 'past_refunds', 'risk_score'
            ],
            'velocity': [
                'past_tx_count_10m', 'past_tx_count_1h', 'past_tx_count_24h',
                'tx_acceleration_1h', 'burst_activity_score'
            ],
            'behavioral': [
                'hour_risk_score', 'amount_percentile_user', 'amount_spike_severity',
                'rapid_succession_flag', 'sec_since_last_success', 'sec_since_last_failure'
            ],
            'network': [
                'ip_reuse_before', 'fp_reuse_before', 'fp_ip_pair_reuse_before',
                'device_sharing_score', 'ip_reputation_score', 'proxy_vpn_probability'
            ],
            'contextual': [
                'email_domain_risk', 'email_domain_class', 'gateway_risk_score',
                'customer_maturity_score', 'is_business_hours'
            ]
        }
    
    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load and prepare transaction data with enhanced preprocessing"""
        
        print("Loading transaction data...")
        
        # Load transactions
        tx = pd.DataFrame(self.db["transactions"].find())
        if tx.empty:
            raise ValueError("No transactions found.")
        
        # Load customers
        cust = pd.DataFrame(self.db["customers"].find())
        
        print(f"Loaded {len(tx)} transactions and {len(cust)} customers")
        
        # Basic preprocessing
        tx = self._preprocess_transactions(tx)
        
        # Enrich with customer data
        if not cust.empty:
            tx = self._enrich_with_customer_data(tx, cust)
        
        # Apply enhanced feature engineering
        if self.feature_engine:
            print("Applying enhanced feature engineering...")
            tx = self.feature_engine.add_behavioral_anomaly_features(tx)
            tx = self.feature_engine.add_network_analysis_features(tx)
            tx = self.feature_engine.add_advanced_velocity_features(tx)
            tx = self.feature_engine.add_contextual_features(tx)
            tx = self.feature_engine.add_ensemble_risk_scores(tx)
        else:
            print("Applying basic feature engineering...")
            tx = self._apply_basic_features(tx)
        
        print(f"Feature engineering complete. Final shape: {tx.shape}")
        return tx
    
    def _preprocess_transactions(self, tx: pd.DataFrame) -> pd.DataFrame:
        """Preprocess transaction data"""
        
        # Ensure required columns
        needed_cols = [
            "email", "amount", "card_country", "billing_address_country", 
            "risk_score", "disputed", "refunded", "ip_address", "fingerprint", 
            "transaction_id", "gateway", "created_at"
        ]
        tx = ensure_cols(tx, needed_cols)
        
        # Data type conversions
        tx["created_at"] = pd.to_datetime(tx["created_at"], errors="coerce")
        tx = tx.dropna(subset=["created_at"]).sort_values("created_at").reset_index(drop=True)
        
        tx["email"] = tx["email"].astype(str).fillna("unknown@example.com")
        tx["amount"] = pd.to_numeric(tx["amount"], errors="coerce").fillna(0.0)
        tx["risk_score"] = pd.to_numeric(tx["risk_score"], errors="coerce").fillna(0.0)
        tx["disputed"] = tx["disputed"].fillna(0).astype(int)
        tx["refunded"] = tx["refunded"].fillna(0).astype(int)
        
        # String columns
        for col in ["card_country", "billing_address_country", "gateway", "ip_address", "fingerprint"]:
            tx[col] = tx[col].fillna("unknown").astype(str)
        
        return tx
    
    def _enrich_with_customer_data(self, tx: pd.DataFrame, cust: pd.DataFrame) -> pd.DataFrame:
        """Enrich transactions with customer data"""
        
        if cust.empty:
            return tx
        
        # Process customer data
        cust["created_at"] = pd.to_datetime(cust.get("created_at"), errors="coerce")
        cust["account_age_days"] = (
            (pd.Timestamp.now(tz=cust["created_at"].dt.tz) - cust["created_at"]).dt.days
        )
        
        enrich_needed = [
            "total_transactions", "total_disputes", "total_refunds", 
            "avg_transaction_amount", "high_risk_tag", "email"
        ]
        cust = ensure_cols(cust, enrich_needed, default=0)
        cust["email"] = cust["email"].fillna("unknown@example.com").astype(str)
        
        enrich_cols = [
            "email", "account_age_days", "total_transactions", "total_disputes",
            "total_refunds", "avg_transaction_amount", "high_risk_tag"
        ]
        
        # Merge with transactions
        tx = tx.merge(cust[enrich_cols].drop_duplicates("email"), on="email", how="left")
        
        # Fill missing values
        for col in ["account_age_days", "total_transactions", "total_disputes", 
                   "total_refunds", "avg_transaction_amount", "high_risk_tag"]:
            tx[col] = tx[col].fillna(0)
        
        return tx
    
    def _apply_basic_features(self, tx: pd.DataFrame) -> pd.DataFrame:
        """Apply basic feature engineering when enhanced features are not available"""
        
        tx = tx.sort_values(["email", "created_at"]).reset_index(drop=True)
        
        # Basic derived features
        tx["hour"] = tx["created_at"].dt.hour
        tx["is_weekend"] = (tx["created_at"].dt.weekday >= 5).astype(int)
        tx["amount_log"] = np.log1p(tx["amount"])
        tx["country_mismatch"] = (tx["card_country"] != tx["billing_address_country"]).astype(int)
        
        # Past transaction features
        tx["past_tx_count"] = tx.groupby("email").cumcount()
        tx["past_disputes"] = (
            tx.groupby("email")["disputed"]
            .shift(fill_value=0)
            .groupby(tx["email"]).cumsum()
        )
        tx["past_refunds"] = (
            tx.groupby("email")["refunded"]
            .shift(fill_value=0)
            .groupby(tx["email"]).cumsum()
        )
        
        # Time-based features
        tx["time_since_prev_tx_sec"] = (
            tx.groupby("email")["created_at"]
            .diff().dt.total_seconds()
            .fillna(999999)
        )
        
        # Enhanced velocity features (multiple windows)
        windows = [(600, "10m"), (3600, "1h"), (86400, "24h")]
        tx = tx.sort_values(["email", "created_at"]).reset_index(drop=True)
        email_groups = tx.groupby("email").groups
        
        for win_s, name in windows:
            counts = pd.Series(0, index=tx.index, dtype="int32")
            for _, idx in email_groups.items():
                s = tx.loc[idx, "created_at"]
                c = rolling_count_seconds(s, win_s)
                counts.loc[idx] = c.values
            tx[f"past_tx_count_{name}"] = counts
        
        # Device and IP reuse tracking
        tx = tx.sort_values(["ip_address", "created_at"]).reset_index(drop=True)
        tx["ip_reuse_before"] = tx.groupby("ip_address").cumcount()
        
        tx = tx.sort_values(["fingerprint", "created_at"]).reset_index(drop=True)
        tx["fp_reuse_before"] = tx.groupby("fingerprint").cumcount()
        
        tx = tx.sort_values(["fingerprint", "ip_address", "created_at"]).reset_index(drop=True)
        tx["fp_ip_pair_reuse_before"] = tx.groupby(["fingerprint", "ip_address"]).cumcount()
        
        # Email domain analysis
        COMMON_DOMAINS = {
            "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com",
            "aol.com", "protonmail.com", "zoho.com", "mail.com", "gmx.com"
        }
        DISPOSABLE_HINTS = {"mailinator", "10minutemail", "guerrillamail", "tempmail", "yopmail"}
        
        tx["email_domain_risk"] = tx["email"].apply(lambda e: domain_risk(e, COMMON_DOMAINS)).astype(int)
        tx["email_domain_class"] = tx["email"].apply(lambda e: domain_class(e, COMMON_DOMAINS, DISPOSABLE_HINTS)).astype(int)
        
        # Amount behavior analysis
        tx["past_avg_amount"] = (
            tx.groupby("email")["amount"]
            .shift()
            .groupby(tx["email"]).expanding().mean()
            .reset_index(level=0, drop=True)
            .fillna(0)
        )
        tx["abs_dev_from_past_mean"] = (tx["amount"] - tx["past_avg_amount"]).abs()
        
        # Back to chronological order
        tx = tx.sort_values("created_at").reset_index(drop=True)
        
        return tx
    
    def train_model(self, df: pd.DataFrame = None) -> Dict:
        """Train the fraud detection model with enhanced capabilities"""
        
        if df is None:
            df = self.load_and_prepare_data()
        
        print("Starting model training...")
        
        # Prepare target variable
        y = df["disputed"].astype(int)
        
        # Select features
        if self.ensemble_manager and ENHANCED_FEATURES_AVAILABLE:
            # Use enhanced feature selection
            all_features = []
            for group_features in self.feature_groups.values():
                all_features.extend(group_features)
            
            # Add enhanced features if available
            enhanced_features = [col for col in df.columns if any(pattern in col for pattern in [
                '_velocity_', '_risk_', '_score', '_reputation_', '_composite'
            ])]
            all_features.extend(enhanced_features)
            
        else:
            # Use basic features
            all_features = [
                "amount_log", "hour", "is_weekend", "country_mismatch",
                "past_tx_count", "past_disputes", "past_refunds", "past_avg_amount",
                "time_since_prev_tx_sec", "ip_reuse_before", "fp_reuse_before",
                "fp_ip_pair_reuse_before", "email_domain_risk", "email_domain_class",
                "abs_dev_from_past_mean", "risk_score", "past_tx_count_10m",
                "past_tx_count_1h", "past_tx_count_24h"
            ]
        
        # Filter available features
        available_features = [f for f in all_features if f in df.columns]
        X = df[available_features].apply(pd.to_numeric, errors="coerce").fillna(0)
        
        print(f"Training with {len(available_features)} features")
        
        # Temporal split
        df = df.sort_values("created_at").reset_index(drop=True)
        cut_time_test = df["created_at"].quantile(0.80)
        train_mask = df["created_at"] <= cut_time_test
        test_mask = df["created_at"] > cut_time_test
        
        X_train_full, y_train_full = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        # Validation split
        cut_time_val = df.loc[train_mask, "created_at"].quantile(0.90)
        val_mask = (df["created_at"] > cut_time_val) & train_mask
        tr_mask = (df["created_at"] <= cut_time_val) & train_mask
        
        X_train, y_train = X[tr_mask], y[tr_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        
        print(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        print(f"Fraud rates - Train: {y_train.mean():.3%}, Val: {y_val.mean():.3%}, Test: {y_test.mean():.3%}")
        
        # Train model
        if self.ensemble_manager and ENHANCED_FEATURES_AVAILABLE:
            # Use ensemble approach
            print("Training enhanced ensemble model...")
            model_scores = self.ensemble_manager.train_ensemble(X_train, y_train, X_val, y_val)
            
            # Get test predictions
            test_scores, _, test_explanations = self.ensemble_manager.predict_ensemble(X_test)
            y_test_pred = (test_scores >= 0.5).astype(int)
            
            # Save ensemble model
            ensemble_path = "/media/mydisk/Project/Test/Lahiru/Python/ai-payment/src/data/models/enhanced_fraud_ensemble.pkl"
            self.ensemble_manager.save_ensemble(ensemble_path)
            
        else:
            # Use single XGBoost model
            print("Training XGBoost model...")
            pipe = ImbPipeline(steps=[
                ("ros", RandomOverSampler(random_state=42)),
                ("scaler", StandardScaler()),
                ("xgb", XGBClassifier(
                    n_estimators=1300,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    tree_method="hist",
                    random_state=42,
                    n_jobs=-1
                ))
            ])
            
            # Cross validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="average_precision")
            print(f"CV PR-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            # Fit and predict
            pipe.fit(X_train, y_train)
            y_val_proba = pipe.predict_proba(X_val)[:, 1]
            
            # Optimize threshold
            p, r, t = precision_recall_curve(y_val, y_val_proba)
            f1s = (2 * p * r) / (p + r + 1e-9)
            best_idx = int(np.nanargmax(f1s)) if len(f1s) else 0
            best_threshold = float(t[max(best_idx-1, 0)]) if len(t) else 0.5
            
            # Test predictions
            test_scores = pipe.predict_proba(X_test)[:, 1]
            y_test_pred = (test_scores >= best_threshold).astype(int)
            
            # Save model
            joblib.dump(pipe, "/media/mydisk/Project/Test/Lahiru/Python/ai-payment/src/data/models/fraud_detection_pipeline.pkl")
            model_scores = {'xgboost': cv_scores.mean()}
        
        # Calculate test metrics
        test_metrics = {
            "accuracy": accuracy_score(y_test, y_test_pred),
            "precision": precision_score(y_test, y_test_pred, zero_division=0),
            "recall": recall_score(y_test, y_test_pred, zero_division=0),
            "f1": f1_score(y_test, y_test_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, test_scores) if len(np.unique(y_test)) > 1 else 0,
            "pr_auc": average_precision_score(y_test, test_scores),
        }
        
        print("\n=== Test Results ===")
        for metric, value in test_metrics.items():
            print(f"{metric:>10}: {value:.4f}")
        
        # Feature importance analysis
        self._analyze_feature_importance(X_train.columns, test_metrics)
        
        # Save metadata
        self._save_model_metadata(available_features, test_metrics, model_scores)
        
        return {
            'test_metrics': test_metrics,
            'model_scores': model_scores,
            'feature_columns': available_features,
            'model_config': self.model_config
        }
    
    def _analyze_feature_importance(self, feature_columns: List[str], test_metrics: Dict):
        """Analyze and visualize feature importance"""
        
        print("\n=== Feature Importance Analysis ===")
        
        try:
            if self.ensemble_manager and hasattr(self.ensemble_manager, 'models'):
                # Get importance from best ensemble model
                best_model_name = max(self.ensemble_manager.model_weights.keys(), 
                                     key=lambda x: self.ensemble_manager.model_weights[x])
                best_model = self.ensemble_manager.models[best_model_name]
            else:
                # Load saved model
                pipe = joblib.load("/media/mydisk/Project/Test/Lahiru/Python/ai-payment/src/data/models/fraud_detection_pipeline.pkl")
                best_model = pipe.named_steps["xgb"]
            
            if hasattr(best_model, 'feature_importances_'):
                importances = best_model.feature_importances_
                
                # Create importance DataFrame
                importance_df = pd.DataFrame({
                    'feature': feature_columns,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                print("Top 15 features:")
                print(importance_df.head(15).to_string(index=False))
                
                # Plot feature importance
                plt.figure(figsize=(12, 8))
                top_features = importance_df.head(20)
                plt.barh(range(len(top_features)), top_features['importance'])
                plt.yticks(range(len(top_features)), top_features['feature'])
                plt.xlabel('Importance')
                plt.title('Top 20 Feature Importances - Fraud Detection')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.savefig('/media/mydisk/Project/Test/Lahiru/Python/ai-payment/src/data/models/fraud_detection_feature_importance.png',
                           dpi=300, bbox_inches='tight')
                plt.show()
                
        except Exception as e:
            print(f"Feature importance analysis failed: {e}")
    
    def _save_model_metadata(self, feature_columns: List[str], test_metrics: Dict, model_scores: Dict):
        """Save model metadata"""
        
        metadata = {
            'model_version': self.model_config['version'],
            'created_at': datetime.now().isoformat(),
            'model_type': 'enhanced_fraud_detection' if ENHANCED_FEATURES_AVAILABLE else 'fraud_detection',
            'features_used': feature_columns,
            'test_metrics': test_metrics,
            'model_scores': model_scores,
            'model_config': self.model_config,
            'feature_groups': self.feature_groups,
            'total_features': len(feature_columns)
        }
        
        with open("/media/mydisk/Project/Test/Lahiru/Python/ai-payment/src/data/models/fraud_detection_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4, default=str)
        
        print(f"\nModel metadata saved with {len(feature_columns)} features")
    
    def predict_transaction_risk(self, transaction_data: Dict) -> Dict:
        """Predict risk for a single transaction"""
        
        # Convert to DataFrame
        df = pd.DataFrame([transaction_data])
        
        # Apply feature engineering
        if self.feature_engine:
            df = self.feature_engine.add_behavioral_anomaly_features(df)
            df = self.feature_engine.add_network_analysis_features(df)
        else:
            df = self._apply_basic_features(df)
        
        # Load model and predict
        if ENHANCED_FEATURES_AVAILABLE and os.path.exists("/media/mydisk/Project/Test/Lahiru/Python/ai-payment/src/data/models/enhanced_fraud_ensemble.pkl"):
            # Use ensemble model
            ensemble_manager = EnsembleModelManager()
            ensemble_manager.load_ensemble("/media/mydisk/Project/Test/Lahiru/Python/ai-payment/src/data/models/enhanced_fraud_ensemble.pkl")
            
            risk_scores, _, explanations = ensemble_manager.predict_ensemble(df)
            risk_score = float(risk_scores[0])
            
        else:
            # Use single model
            pipe = joblib.load("/media/mydisk/Project/Test/Lahiru/Python/ai-payment/src/data/models/fraud_detection_pipeline.pkl")
            risk_score = float(pipe.predict_proba(df)[:, 1][0])
            explanations = {}
        
        # Generate response
        response = {
            'transaction_id': transaction_data.get('transaction_id', 'unknown'),
            'risk_score': risk_score,
            'risk_level': self._categorize_risk(risk_score),
            'recommendation': self._get_recommendation(risk_score),
            'confidence': min(1.0, abs(risk_score - 0.5) * 2),
            'model_version': self.model_config['version'],
            'explanations': explanations.get('top_features', {}),
            'processing_timestamp': datetime.now().isoformat()
        }
        
        return response
    
    def _categorize_risk(self, score: float) -> str:
        """Categorize risk level"""
        if score >= 0.8:
            return "CRITICAL"
        elif score >= 0.6:
            return "HIGH"
        elif score >= 0.4:
            return "MEDIUM"
        elif score >= 0.2:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _get_recommendation(self, score: float) -> str:
        """Get recommendation based on risk score"""
        if score >= 0.8:
            return "BLOCK_TRANSACTION"
        elif score >= 0.6:
            return "MANUAL_REVIEW"
        elif score >= 0.4:
            return "ADDITIONAL_VERIFICATION"
        elif score >= 0.2:
            return "MONITOR_CLOSELY"
        else:
            return "APPROVE"


# Main training function (backward compatibility)
def train():
    """Train the fraud detection model"""
    
    print("🚀 Enhanced Fraud Detection Training")
    print("=" * 50)
    
    try:
        # Initialize enhanced fraud detection
        fraud_detector = EnhancedFraudDetection()
        
        # Train the model
        results = fraud_detector.train_model()
        
        print("\n✅ Training completed successfully!")
        print(f"Model version: {results['model_config']['version']}")
        print(f"Enhanced features: {'✓' if results['model_config']['use_enhanced_features'] else '✗'}")
        print(f"Feature count: {len(results['feature_columns'])}")
        print(f"Test PR-AUC: {results['test_metrics']['pr_auc']:.4f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise


load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("Missing MONGO_URI in environment.")

# Legacy code for backward compatibility - wrapped in the enhanced class
if __name__ == "__main__":
    train()

# Legacy code for backward compatibility - wrapped in the enhanced class
if __name__ == "__main__":
    train()
