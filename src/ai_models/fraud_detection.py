import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import numpy as np
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client['payment_intelligence']
transactions_collection = db['transactions']
customers_collection = db['customers']

df = pd.DataFrame(transactions_collection.find())
df_customers = pd.DataFrame(customers_collection.find())

df_customers['created_at'] = pd.to_datetime(df_customers['created_at'])
df_customers['account_age_days'] = (pd.Timestamp.now() - df_customers['created_at']).dt.days
for col in ['total_transactions', 'total_disputes', 'total_refunds', 'avg_transaction_amount', 'high_risk_tag']:
    df_customers[col] = df_customers.get(col, 0)
df_customers['email'] = df_customers['email'].fillna('unknown@example.com')

df = pd.merge(df, df_customers[['email', 'account_age_days', 'total_transactions', 'total_disputes',
                   'total_refunds', 'avg_transaction_amount', 'high_risk_tag']], on='email', how='left').fillna(0)


columns_to_fill = ['billing_address_country', 'card_brand', 'card_country', 'cvc_check',
                   'funding_type', 'network_status', 'outcome_type', 'payment_method',
                   'postal_code_check', 'risk_level', 'email', 'fingerprint', 'gateway', 'ip_address']
for col in columns_to_fill:
    df[col] = df[col].fillna('unknown').astype(str)

categorical_cols = ['billing_address_country', 'card_brand', 'card_country', 'cvc_check',
                    'funding_type', 'network_status', 'outcome_type', 'payment_method',
                    'postal_code_check', 'risk_level', 'gateway']
label_encoders = {col: LabelEncoder().fit(df[col]) for col in categorical_cols}
for col, le in label_encoders.items():
    df[col] = le.transform(df[col])

exclude_columns = ['_id', 'transaction_id', 'created_at', 'email', 'fingerprint', 'ip_address',
                   'email_domain', 'billing_address_line1', 'billing_address_line2', 'billing_address_city',
                   'billing_email', 'billing_name', 'billing_phone', 'receipt_url', 'seller_message', 'three_d_secure', 'status', 'disputed']
final_features = [col for col in df.columns if col not in exclude_columns]

df['amount_log'] = np.log1p(df['amount'])
df['hour'] = pd.to_datetime(df['created_at']).dt.hour
df['is_weekend'] = pd.to_datetime(df['created_at']).dt.weekday.apply(lambda x: 1 if x >= 5 else 0)
df['country_mismatch'] = (df['card_country'] != df['billing_address_country']).astype(int)
df['ip_address_reuse_count'] = df.groupby('ip_address')['transaction_id'].transform('count')
df['device_fingerprint_reuse_count'] = df.groupby('fingerprint')['transaction_id'].transform('count')
df['device_ip_pair_reuse_count'] = df.groupby(['fingerprint', 'ip_address'])['transaction_id'].transform('count')
df['ip_address_country_mismatch'] = (df['card_country'] != df['ip_address']).astype(int)
df['email_transaction_count'] = df.groupby('email')['transaction_id'].transform('count')
df['email_dispute_count'] = df.groupby('email')['disputed'].transform('sum')
df['email_refund_count'] = df.groupby('email')['refunded'].transform('sum')
df['chargeback_rate'] = df['email_dispute_count'] / df['email_transaction_count']
df['email_avg_amount'] = df.groupby('email')['amount'].transform('mean')
df['unusual_amount_flag'] = (df['amount'] > df['amount'].quantile(0.99)).astype(int)
df['time_between_transactions'] = df.groupby('email')['created_at'].diff().dt.total_seconds().fillna(999999)
df['first_time_transaction'] = (df['email_transaction_count'] == 1).astype(int)
df['customer_avg_amount_diff'] = abs(df['amount'] - df['email_avg_amount'])
df['customer_last_transaction_diff'] = df.groupby('email')['amount'].diff().abs().fillna(0)
df['customer_refund_ratio'] = df['email_refund_count'] / df['email_transaction_count']
df['shared_card_email_count'] = df.groupby('fingerprint')['email'].transform('nunique')
df['shared_ip_email_count'] = df.groupby('ip_address')['email'].transform('nunique')
df['email_domain_risk'] = df['email'].apply(lambda x: 1 if x.split('@')[-1] in ['gmail.com', 'yahoo.com', 'hotmail.com'] else 0)
df['previous_risk_scores_avg'] = df.groupby('email')['risk_score'].transform('mean')
df['number_of_risky_transactions'] = df.groupby('email')['risk_score'].transform(lambda x: (x > 50).sum())

X = df[final_features].replace('unknown', 0).apply(pd.to_numeric, errors='coerce').fillna(0)
y = df['disputed'].astype(int)

X_train_raw, X_test, y_train_raw, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

df_train_raw = pd.concat([X_train_raw, y_train_raw.rename('disputed')], axis=1)
df_fraud = df_train_raw[df_train_raw['disputed'] == 1]
df_nonfraud = df_train_raw[df_train_raw['disputed'] == 0]
df_fraud_upsampled = df_fraud.sample(n=len(df_nonfraud), replace=True, random_state=42)
df_balanced = pd.concat([df_nonfraud, df_fraud_upsampled]).sample(frac=1, random_state=42)

X_train = df_balanced.drop(columns=['disputed'])
y_train = df_balanced['disputed']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = XGBClassifier(n_estimators=1000, max_depth=12, learning_rate=0.02, subsample=0.95, colsample_bytree=0.95, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(model, "/src/data/models/fraud_detection_model_final.pkl")
joblib.dump(scaler, "/src/data/models/fraud_detection_scaler_final.pkl")
joblib.dump(label_encoders, "/src/data/models/fraud_detection_label_encoders_final.pkl")
print("✅ Model, Scaler, and Encoders saved successfully.")
