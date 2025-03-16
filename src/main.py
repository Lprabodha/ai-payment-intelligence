import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib

# ==========================
# Load and Inspect CSV File
# ==========================
try:
    df = pd.read_csv('/src/data/combined_transactions.csv')
    print("✅ CSV file loaded successfully!")
except Exception as e:
    print(f"❌ Error loading CSV file: {e}")
    exit()

print(f"📊 Dataset contains {df.shape[0]} rows and {df.shape[1]} columns")

# ==========================
# Data Preprocessing & Cleaning
# ==========================

# Fill missing values for necessary fields
fill_columns = {
    'Captured': 'False',
    'Decline Reason': 'unknown',
    'Customer Email': 'unknown',
    'Converted Currency': 'unknown',
    'Amount Refunded': 0,
    'Amount': 0,
    'Converted Amount': 0
}
df.fillna(fill_columns, inplace=True)

# Encode categorical fields
categorical_cols = ['Captured', 'Currency', 'Converted Currency']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le  # Save encoder for future use

# Convert datetime and extract time features
df['Created date (UTC)'] = pd.to_datetime(df['Created date (UTC)'], errors='coerce')
df['hour'] = df['Created date (UTC)'].dt.hour.fillna(0).astype(int)
df['weekday'] = df['Created date (UTC)'].dt.weekday.fillna(0).astype(int)
df['unusual_hour'] = df['hour'].apply(lambda x: 1 if x < 6 or x > 22 else 0)
df['weekend_transaction'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)

# Refund flag (can be used as label if no explicit fraud label is present)
df['refunded'] = df['Amount Refunded'].apply(lambda x: 1 if x > 0 else 0)

# ==========================
# Dynamic Label Handling
# ==========================
if 'fraud_label' in df.columns:
    y = df['fraud_label']
    print("✅ 'fraud_label' used as target.")
else:
    y = df['refunded']  # Use refunded as proxy if no labeled fraud data
    print("⚠️ Using 'refunded' as proxy target label. Consider labeling real fraud data.")

# ==========================
# Feature Engineering & Selection
# ==========================
features = [
    'Amount', 'Captured', 'Currency', 'Converted Amount', 'Converted Currency',
    'hour', 'weekday', 'unusual_hour', 'weekend_transaction'
]

X = df[features]

# ==========================
# Feature Scaling
# ==========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✅ Features scaled.")

# ==========================
# Train-Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Training set: {X_train.shape[0]} rows, Test set: {X_test.shape[0]} rows")

# ==========================
# Handle Class Imbalance with SMOTE
# ==========================
print("⚙️ Balancing dataset using SMOTE...")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"✅ After SMOTE: {X_train_bal.shape[0]} training rows (balanced)")

# ==========================
# Train XGBoost Model (With Improved Parameters)
# ==========================
model = XGBClassifier(
    n_estimators=500,  # More trees
    max_depth=8,      # More depth for complex relations
    learning_rate=0.05,  # Lower learning rate
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    eval_metric='auc',
    use_label_encoder=False
)

model.fit(X_train_bal, y_train_bal)
print("✅ Model training completed.")

# ==========================
# Evaluate Model
# ==========================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
print(f"✅ Test Accuracy: {accuracy:.4f}")
print(f"✅ ROC AUC Score: {roc_auc:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ==========================
# Cross-validation Accuracy
# ==========================
cv_scores = cross_val_score(model, X_train_bal, y_train_bal, cv=5, scoring='accuracy')
print(f"✅ Cross-validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ==========================
# Save Model, Scaler, and Encoders
# ==========================
joblib.dump(model, "fraud_detection_model.pkl")
joblib.dump(scaler, "fraud_detection_scaler.pkl")
joblib.dump(label_encoders, "fraud_detection_label_encoders.pkl")
print("✅ Model, scaler, and label encoders saved successfully.")
