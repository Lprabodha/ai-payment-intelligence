import os, json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt

# ---------------------------
# Helpers (past-only, index-safe)
# ---------------------------
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

# ---------------------------
# 0) Load data
# ---------------------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("Missing MONGO_URI in environment.")

client = MongoClient(MONGO_URI)
db = client["payment_intelligence"]
tx = pd.DataFrame(db["transactions"].find())
if tx.empty:
    raise ValueError("No transactions found.")
if "created_at" not in tx.columns:
    raise ValueError("transactions.created_at is required.")

# ---------------------------
# 1) Basic cleaning
# ---------------------------
tx["created_at"] = pd.to_datetime(tx["created_at"], errors="coerce")
tx = tx.dropna(subset=["created_at"]).sort_values("created_at").reset_index(drop=True)

needed = [
    "transaction_id","email","amount","card_country","billing_address_country",
    "risk_score","disputed","refunded","ip_address","fingerprint","gateway"
]
tx = ensure_cols(tx, needed)
tx["email"] = tx["email"].astype(str).fillna("unknown@example.com")
tx["amount"] = pd.to_numeric(tx["amount"], errors="coerce").fillna(0.0)
tx["risk_score"] = pd.to_numeric(tx["risk_score"], errors="coerce").fillna(0.0)
tx["disputed"] = pd.to_numeric(tx["disputed"], errors="coerce").fillna(0).astype(int)
tx["refunded"] = pd.to_numeric(tx["refunded"], errors="coerce").fillna(0).astype(int)
tx["card_country"] = tx["card_country"].astype(str).fillna("UNK")
tx["billing_address_country"] = tx["billing_address_country"].astype(str).fillna("UNK")
tx["ip_address"] = tx["ip_address"].astype(str).fillna("0.0.0.0")
tx["fingerprint"] = tx["fingerprint"].astype(str).fillna("unknown")
tx["gateway"] = tx["gateway"].astype(str).fillna("unknown")

# ---------------------------
# 2) Leakage-safe feature engineering (PAST ONLY)
# ---------------------------
# Sort by entity+time for cumulative “past” features
tx = tx.sort_values(["email","created_at"]).reset_index(drop=True)

# Time/amount basics
tx["amount_log"] = np.log1p(tx["amount"])
tx["hour"] = tx["created_at"].dt.hour
tx["is_weekend"] = (tx["created_at"].dt.weekday >= 5).astype(int)

# Past behavior per email
tx["past_tx_count_email"] = tx.groupby("email").cumcount()

# Time between user transactions (past-only via diff)
tx["time_between_transactions"] = (
    tx.groupby("email")["created_at"].diff().dt.total_seconds().fillna(999999.0)
)

# Rolling velocity counts (10m/1h/24h) per email
windows = [(600, "10m"), (3600, "1h"), (86400, "24h")]
email_groups = tx.groupby("email").groups
for win_s, name in windows:
    counts = pd.Series(0, index=tx.index, dtype="int32")
    for _, idx in email_groups.items():
        s = tx.loc[idx, "created_at"]
        c = rolling_count_seconds(s, win_s)  # counts of prior events in window
        counts.loc[idx] = c.values
    tx[f"past_tx_count_{name}"] = counts

# Past rolling averages (shift then expanding mean) – refunds and amounts
tx["past_refund_count"] = (
    tx.groupby("email")["refunded"].shift(fill_value=0)
      .groupby(tx["email"]).cumsum()
)
tx["past_tx_count_for_ratio"] = tx["past_tx_count_email"].replace(0, np.nan)
tx["customer_refund_ratio_past"] = (tx["past_refund_count"] / tx["past_tx_count_for_ratio"]).fillna(0.0)

tx["past_avg_amount"] = (
    tx.groupby("email")["amount"].shift()
      .groupby(tx["email"]).expanding().mean()
      .reset_index(level=0, drop=True)
      .fillna(0.0)
)
tx["transaction_amount_diff"] = (tx["amount"] - tx["past_avg_amount"]).abs()

# “Past chargebacks” (shift then cumsum)
tx["past_chargebacks"] = (
    tx.groupby("email")["disputed"].shift(fill_value=0)
      .groupby(tx["email"]).cumsum()
)

# Country mismatch (current info; no leakage)
tx["country_mismatch"] = (tx["card_country"] != tx["billing_address_country"]).astype(int)

# IP/fingerprint reuse BEFORE current tx (index-safe via cumcount after sorting within key)
tx = tx.sort_values(["ip_address","created_at"]).reset_index(drop=True)
tx["ip_address_reuse_before"] = tx.groupby("ip_address").cumcount()

tx = tx.sort_values(["fingerprint","created_at"]).reset_index(drop=True)
tx["fingerprint_reuse_before"] = tx.groupby("fingerprint").cumcount()

tx = tx.sort_values(["fingerprint","ip_address","created_at"]).reset_index(drop=True)
tx["device_ip_pair_reuse_before"] = tx.groupby(["fingerprint","ip_address"]).cumcount()

# Back to global chronological order for splitting later
tx = tx.sort_values("created_at").reset_index(drop=True)

# Email domain risk
COMMON_DOMAINS = {
    "gmail.com","yahoo.com","hotmail.com","outlook.com","icloud.com",
    "aol.com","protonmail.com","zoho.com","mail.com","gmx.com"
}
tx["email_domain_risk"] = tx["email"].apply(lambda e: domain_risk(e, COMMON_DOMAINS)).astype(int)

# OPTIONAL: If you have ip_country column, you can add:
# if "ip_country" in tx.columns:
#     tx["ip_country_mismatch"] = (tx["card_country"] != tx["ip_country"]).astype(int)

# ---------------------------
# 3) Labels & features
# ---------------------------
y = tx["disputed"].astype(int)

feature_cols = [
    "amount_log","hour","is_weekend",
    "past_tx_count_email","past_tx_count_10m","past_tx_count_1h","past_tx_count_24h",
    "time_between_transactions",
    "past_refund_count","customer_refund_ratio_past",
    "past_avg_amount","transaction_amount_diff",
    "past_chargebacks",
    "country_mismatch",
    "ip_address_reuse_before","fingerprint_reuse_before","device_ip_pair_reuse_before",
    "email_domain_risk",
    # Use risk_score only if it is a pre-authorization signal; otherwise comment it out.
    "risk_score"
]
feature_cols = [c for c in feature_cols if c in tx.columns]
X = tx[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

# ---------------------------
# 4) Temporal split: train (old), val (recent), test (future)
# ---------------------------
cut_time_test = tx["created_at"].quantile(0.80)
train_mask = tx["created_at"] <= cut_time_test
test_mask  = tx["created_at"]  > cut_time_test

X_train_full, y_train_full = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

# validation = last 10% of train period
cut_time_val = tx.loc[train_mask, "created_at"].quantile(0.90)
val_mask = (tx["created_at"] > cut_time_val) & train_mask
tr_mask  = (tx["created_at"] <= cut_time_val) & train_mask

X_train, y_train = X[tr_mask], y[tr_mask]
X_val,   y_val   = X[val_mask], y[val_mask]

print(f"Sizes -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# ---------------------------
# 5) Pipeline + CV (oversample/scale inside folds)
# ---------------------------
pipe = ImbPipeline(steps=[
    ("ros",    RandomOverSampler(random_state=42)),
    ("scaler", StandardScaler()),
    ("xgb",    XGBClassifier(
        n_estimators=900,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    ))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="average_precision")
print(f"CV PR-AUC (train-only): mean={cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Fit on train and tune threshold on validation
pipe.fit(X_train, y_train)
y_val_proba = pipe.predict_proba(X_val)[:, 1]

p, r, t = precision_recall_curve(y_val, y_val_proba)
f1s = (2 * p * r) / (p + r + 1e-9)
best_idx = int(np.nanargmax(f1s)) if len(f1s) else 0
best_threshold = float(t[max(best_idx-1, 0)]) if len(t) else 0.5
print(f"Chosen threshold (val F1): {best_threshold:.3f}")

# ---------------------------
# 6) Final evaluation on hold-out test
# ---------------------------
y_test_proba = pipe.predict_proba(X_test)[:, 1]
y_test_pred  = (y_test_proba >= best_threshold).astype(int)

metrics = {
    "accuracy":  accuracy_score(y_test, y_test_pred) if len(y_test) else float("nan"),
    "precision": precision_score(y_test, y_test_pred, zero_division=0) if len(y_test) else float("nan"),
    "recall":    recall_score(y_test, y_test_pred, zero_division=0) if len(y_test) else float("nan"),
    "f1":        f1_score(y_test, y_test_pred, zero_division=0) if len(y_test) else float("nan"),
    "roc_auc":   roc_auc_score(y_test, y_test_proba) if len(np.unique(y_test))>1 else float("nan"),
    "pr_auc":    average_precision_score(y_test, y_test_proba) if len(y_test) else float("nan"),
}
print("\n✅ HOLD-OUT TEST METRICS")
for k, v in metrics.items():
    print(f"{k:>9}: {v:.4f}")

print("\nConfusion matrix (test):")
print(confusion_matrix(y_test, y_test_pred) if len(y_test) else "N/A")

print("\nClassification report (test):")
print(classification_report(y_test, y_test_pred, zero_division=0) if len(y_test) else "N/A")

# ---------------------------
# 7) Plots (optional)
# ---------------------------
try:
    from sklearn.metrics import RocCurveDisplay
    if len(np.unique(y_test)) > 1 and len(y_test) > 0:
        RocCurveDisplay.from_predictions(y_test, y_test_proba)
        plt.title("ROC Curve (Test)")
        plt.show()
except Exception:
    pass

try:
    xgb_model = pipe.named_steps["xgb"]
    importances = xgb_model.feature_importances_
    order = np.argsort(importances)[::-1]
    plt.figure(figsize=(12, 6))
    plt.title("Feature Importances (XGBoost)")
    plt.bar(range(len(order)), importances[order])
    plt.xticks(range(len(order)), [X.columns[i] for i in order], rotation=90)
    plt.tight_layout()
    os.makedirs("/src/data/models", exist_ok=True)
    plt.savefig("/src/data/models/chargeback_feature_importance.png")
    plt.show()
except Exception:
    pass

# ---------------------------
# 8) Save artifacts & metadata
# ---------------------------
os.makedirs("/src/data/models", exist_ok=True)
joblib.dump(pipe, "/src/data/models/chargeback_pipeline.pkl")

report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0) if len(y_test) else {}
with open("/src/data/models/chargeback_report.json", "w") as f:
    json.dump(report, f, indent=4)

metadata = {
    "model_version": "1.1.0",
    "created_at": pd.Timestamp.now().isoformat(),
    "features_used": list(X.columns),
    "threshold": best_threshold,
    "metrics_test": metrics,
    "sizes": {"train": int(len(X_train)), "val": int(len(X_val)), "test": int(len(X_test))},
    "cv_pr_auc_mean_train": float(cv_scores.mean()),
    "cv_pr_auc_std_train": float(cv_scores.std()),
}
with open("/src/data/models/chargeback_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("\n✅ Pipeline, report, and metadata saved to /src/data/models")
