import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, RocCurveDisplay
)
from imblearn.over_sampling import RandomOverSampler
from pymongo import MongoClient
import joblib
import os
import json
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# Connect to MongoDB and load data
client = MongoClient(MONGO_URI)
db = client['payment_intelligence']
transactions_collection = db['transactions']
transactions = list(transactions_collection.find())
df = pd.DataFrame(transactions)

# Preprocessing
df['amount_log'] = np.log1p(df['amount'])
df['created_at'] = pd.to_datetime(df['created_at'])
df['hour'] = df['created_at'].dt.hour
df['is_weekend'] = (df['created_at'].dt.weekday >= 5).astype(int)
df['time_between_transactions'] = df.groupby('email')['created_at'].diff().dt.total_seconds().fillna(999999)

df['ip_address_reuse_count'] = df.groupby('ip_address')['transaction_id'].transform('count')
df['fingerprint_reuse_count'] = df.groupby('fingerprint')['transaction_id'].transform('count')
df['device_ip_pair_reuse_count'] = df.groupby(['fingerprint', 'ip_address'])['transaction_id'].transform('count')
df['email_transaction_count'] = df.groupby('email')['transaction_id'].transform('count')
df['customer_refund_ratio'] = df.groupby('email')['refunded'].transform('mean')
df['average_transaction_amount'] = df.groupby('email')['amount'].transform('mean')
df['transaction_amount_diff'] = abs(df['amount'] - df['average_transaction_amount'])
df['country_mismatch'] = (df['card_country'] != df['billing_address_country']).astype(int)
df['ip_country_mismatch'] = (df['card_country'] != df['ip_address']).astype(int)
valid_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'icloud.com', 'aol.com', 'protonmail.com', 'zoho.com', 'mail.com', 'gmx.com']
df['email_domain_risk'] = df['email'].apply(
    lambda x: 1 if x.split('@')[-1].lower() not in valid_domains else 0
)
df['past_chargebacks'] = df.groupby('email')['disputed'].transform('sum')

# Prepare features and target
df['disputed'] = df['disputed'].fillna(0)
y = df['disputed'].astype(int)
features = [
    'amount_log', 'hour', 'is_weekend', 'ip_address_reuse_count', 'fingerprint_reuse_count',
    'device_ip_pair_reuse_count', 'risk_score', 'email_transaction_count',
    'customer_refund_ratio', 'country_mismatch', 'ip_country_mismatch', 'time_between_transactions',
    'email_domain_risk', 'transaction_amount_diff', 'past_chargebacks'
]
X = df[features].replace('unknown', 0).apply(pd.to_numeric, errors='coerce').fillna(0)

# Oversample and scale
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Split data
X_temp, X_val, y_temp, y_val = train_test_split(X_scaled, y_resampled, test_size=0.1, stratify=y_resampled, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=0.2, stratify=y_temp, random_state=42)

# Train model
model = XGBClassifier(n_estimators=800, max_depth=10, learning_rate=0.03,
                      subsample=0.95, colsample_bytree=0.95,
                      scale_pos_weight=1, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_scaled, y_resampled, cv=cv, scoring='f1')
print(f"✅ F1 CV Scores: {cv_scores}, Mean: {cv_scores.mean():.4f}")

model.fit(X_train, y_train)
print("✅ Model trained.")

# Evaluation
def print_metrics(y_true, y_pred):
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_proba)

print_metrics(y_test, y_pred)
print(f"ROC AUC: {roc_auc:.4f}")

# Plot ROC
RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title("ROC Curve")
plt.show()

# Plot feature importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=90)
plt.tight_layout()
os.makedirs("/src/data/models", exist_ok=True)
plt.savefig("/src/data/models/chargeback_prediction.png")
plt.show()

# Save model
joblib.dump(model, "/src/data/models/chargeback_prediction_model.pkl")
joblib.dump(scaler, "/src/data/models/chargeback_prediction_scaler.pkl")
print("✅ Model and scaler saved.")

# Save report and metadata
with open("/src/data/models/chargeback_prediction_report.json", "w") as f:
    json.dump(classification_report(y_test, y_pred, output_dict=True), f, indent=4)

with open("/src/data/models/metadata.json", "w") as f:
    json.dump({
        "model_version": "1.0.0",
        "created_at": pd.Timestamp.now().isoformat(),
        "features_used": features,
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc
    }, f, indent=4)

print("✅ All results and metadata saved successfully.")
