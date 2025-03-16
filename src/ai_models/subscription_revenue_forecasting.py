import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client['payment_intelligence']
subscriptions_collection = db['subscriptions']

df = pd.DataFrame(subscriptions_collection.find())


df['created_at'] = pd.to_datetime(df['created_at'])
df['current_period_end'] = pd.to_datetime(df['current_period_end'])
df['current_period_start'] = pd.to_datetime(df['current_period_start'])
df['billing_cycle_anchor'] = pd.to_datetime(df['billing_cycle_anchor'])

df['account_age_days'] = (pd.Timestamp.now() - df['created_at']).dt.days
df['subscription_duration_days'] = (df['current_period_end'] - df['current_period_start']).dt.days
df['is_canceled'] = df['cancel_at_period_end'].astype(int)
df['is_active'] = (df['status'] == 'active').astype(int)

df['total_subscription_periods'] = df['account_age_days'] // 30
df['subscription_churned'] = df['ended_at'].notnull().astype(int)
df['trial_used'] = df['trial_start'].notnull().astype(int)
df['renewal_count'] = df.groupby('email')['subscription_id'].transform('count')
df['multiple_subscriptions'] = (df['renewal_count'] > 1).astype(int)

df['average_subscription_value'] = df.groupby('email')['price_amount'].transform('mean')
df['high_value_customer'] = (df['average_subscription_value'] > 100).astype(int)
df['payment_frequency'] = df.groupby('email')['interval'].transform('count')
df['long_term_discount'] = (df['subscription_duration_days'] > 180).astype(int)

df['expected_next_revenue'] = df['average_subscription_value']

df['subscription_start_month'] = df['current_period_start'].dt.month
df['start_dow'] = df['current_period_start'].dt.dayofweek
df['tenure_bucket'] = pd.cut(
    df['account_age_days'],
    bins=[0, 90, 180, 365, 730, np.inf],
    labels=['0-3m', '3-6m', '6-12m', '1-2y', '2y+']
).astype(str)

categorical_cols = ['collection_method', 'currency', 'interval', 'gateway', 'tenure_bucket']
for col in categorical_cols:
    df[col] = df[col].fillna('unknown').astype(str)

label_encoders = {col: LabelEncoder().fit(df[col]) for col in categorical_cols}
for col, le in label_encoders.items():
    df[col] = le.transform(df[col])

exclude_columns = ['_id', 'subscription_id', 'email', 'default_payment_method',
                   'latest_invoice', 'metadata', 'product_id', 'plan_id', 'plan_name',
                   'created_at', 'current_period_end', 'current_period_start', 'billing_cycle_anchor',
                   'trial_end', 'trial_start', 'ended_at', 'status', 'price_amount', 'quantity']

final_features = [col for col in df.columns if col not in exclude_columns + ['expected_next_revenue']]

X = df[final_features].apply(pd.to_numeric, errors='coerce').fillna(0)
y = df['expected_next_revenue']

unique_emails = df['email'].unique()
train_emails, test_emails = train_test_split(unique_emails, test_size=0.2, random_state=42)

X_train = df[df['email'].isin(train_emails)][final_features].apply(pd.to_numeric, errors='coerce').fillna(0)
y_train = df[df['email'].isin(train_emails)]['expected_next_revenue']
X_test = df[df['email'].isin(test_emails)][final_features].apply(pd.to_numeric, errors='coerce').fillna(0)
y_test = df[df['email'].isin(test_emails)]['expected_next_revenue']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = XGBRegressor(
    n_estimators=800,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42
)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print("📊 Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("📊 Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("📊 R2 Score:", r2_score(y_test, y_pred))

import datetime
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
joblib.dump(model, "subscription_revenue_forecasting_model.pkl")
joblib.dump(scaler, "subscription_revenue_scaler}.pkl")
joblib.dump(label_encoders, "subscription_revenue_label_encoders.pkl")
print("✅ Enhanced Subscription Revenue Forecasting Model  saved successfully.")
