import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import joblib
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client['payment_intelligence']
transactions_collection = db['transactions']

transactions = list(transactions_collection.find())
df = pd.DataFrame(transactions)
print(df.head())

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

df['email_domain_risk'] = df['email'].apply(lambda x: 1 if x.split('@')[-1] in ['gmail.com', 'yahoo.com', 'hotmail.com'] else 0)
df['past_chargebacks'] = df.groupby('email')['disputed'].transform('sum')


y = df['disputed'].astype(int)

feature_columns = [
    'amount_log', 'hour', 'is_weekend', 'ip_address_reuse_count', 'fingerprint_reuse_count',
    'device_ip_pair_reuse_count', 'risk_score', 'email_transaction_count',
    'customer_refund_ratio', 'country_mismatch', 'ip_country_mismatch', 'time_between_transactions',
    'email_domain_risk', 'transaction_amount_diff', 'past_chargebacks'
]

X = df[feature_columns].replace('unknown', 0).apply(pd.to_numeric, errors='coerce').fillna(0)

ros = RandomOverSampler(random_state=42)
X_balanced, y_balanced = ros.fit_resample(X, y)
print(f"Balanced dataset shape: {X_balanced.shape}, Labels: {np.bincount(y_balanced)}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_balanced)

model = XGBClassifier(
    n_estimators=800,
    max_depth=10,
    learning_rate=0.03,
    subsample=0.95,
    colsample_bytree=0.95,
    scale_pos_weight=1,
    random_state=42
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_scaled, y_balanced, cv=skf, scoring='f1')
print(f"✅ Stratified F1 scores: {scores}, Mean F1 Score: {scores.mean()}")


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_balanced, test_size=0.2, stratify=y_balanced, random_state=42)
model.fit(X_train, y_train)
print("✅ Final Chargeback Prediction AI Model trained successfully.")

y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


joblib.dump(model, "chargeback_prediction_model.pkl")
joblib.dump(scaler, "chargeback_prediction_scaler.pkl")
print("✅ Chargeback Prediction Model and Scaler saved successfully.")
