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


load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("Missing MONGO_URI in environment.")

mongo = MongoClient(MONGO_URI)
db = mongo["payment_intelligence"]
tx = pd.DataFrame(db["transactions"].find())
cust = pd.DataFrame(db["customers"].find())

if tx.empty:
    raise ValueError("No transactions found.")
if "created_at" not in tx.columns:
    raise ValueError("transactions.created_at is required.")


tx["created_at"] = pd.to_datetime(tx["created_at"], errors="coerce")
tx = tx.dropna(subset=["created_at"]).sort_values("created_at").reset_index(drop=True)

needed_cols = [
    "email","amount","card_country","billing_address_country","risk_score","disputed","refunded",
    "ip_address","fingerprint","transaction_id","gateway"
]
tx = ensure_cols(tx, needed_cols)
tx["email"] = tx["email"].astype(str).fillna("unknown@example.com")
tx["amount"] = pd.to_numeric(tx["amount"], errors="coerce").fillna(0.0)
tx["risk_score"] = pd.to_numeric(tx["risk_score"], errors="coerce").fillna(0.0)
tx["disputed"] = tx["disputed"].fillna(0).astype(int)
tx["refunded"] = tx["refunded"].fillna(0).astype(int)
tx["card_country"] = tx["card_country"].fillna("UNK").astype(str)
tx["billing_address_country"] = tx["billing_address_country"].fillna("UNK").astype(str)
tx["gateway"] = tx["gateway"].fillna("unknown").astype(str)
tx["ip_address"] = tx["ip_address"].fillna("0.0.0.0").astype(str)
tx["fingerprint"] = tx["fingerprint"].fillna("unknown").astype(str)

if not cust.empty:
    cust["created_at"] = pd.to_datetime(cust.get("created_at"), errors="coerce")
    cust["account_age_days"] = (
        (pd.Timestamp.now(tz=cust["created_at"].dt.tz)
         - cust["created_at"]).dt.days
    )
    enrich_needed = ["total_transactions","total_disputes","total_refunds","avg_transaction_amount","high_risk_tag","email"]
    cust = ensure_cols(cust, enrich_needed, default=0)
    cust["email"] = cust["email"].fillna("unknown@example.com").astype(str)
    enrich_cols = ["email","account_age_days","total_transactions","total_disputes",
                   "total_refunds","avg_transaction_amount","high_risk_tag"]
    tx = tx.merge(cust[enrich_cols].drop_duplicates("email"), on="email", how="left")


tx = tx.sort_values(["email","created_at"]).reset_index(drop=True)

tx["hour"] = tx["created_at"].dt.hour
tx["is_weekend"] = (tx["created_at"].dt.weekday >= 5).astype(int)
tx["amount_log"] = np.log1p(tx["amount"])

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

tx["past_avg_amount"] = (
    tx.groupby("email")["amount"]
      .shift()
      .groupby(tx["email"]).expanding().mean()
      .reset_index(level=0, drop=True)
      .fillna(0)
)

tx["time_since_prev_tx_sec"] = (
    tx.groupby("email")["created_at"]
      .diff().dt.total_seconds()
      .fillna(999999)
)

tx["country_mismatch"] = (tx["card_country"] != tx["billing_address_country"]).astype(int)

tx = tx.sort_values(["ip_address","created_at"]).reset_index(drop=True)
tx["ip_reuse_before"] = tx.groupby("ip_address").cumcount()

tx = tx.sort_values(["fingerprint","created_at"]).reset_index(drop=True)
tx["fp_reuse_before"] = tx.groupby("fingerprint").cumcount()

tx = tx.sort_values(["fingerprint","ip_address","created_at"]).reset_index(drop=True)
tx["fp_ip_pair_reuse_before"] = tx.groupby(["fingerprint","ip_address"]).cumcount()

# Back to global chronological order
tx = tx.sort_values("created_at").reset_index(drop=True)

# Email domain risk/class
COMMON_DOMAINS = {
    "gmail.com","yahoo.com","hotmail.com","outlook.com","icloud.com",
    "aol.com","protonmail.com","zoho.com","mail.com","gmx.com"
}
DISPOSABLE_HINTS = {"mailinator", "10minutemail", "guerrillamail", "tempmail", "yopmail", "trashmail"}

tx["email_domain_risk"]  = tx["email"].apply(lambda e: domain_risk(e, COMMON_DOMAINS)).astype(int)
tx["email_domain_class"] = tx["email"].apply(lambda e: domain_class(e, COMMON_DOMAINS, DISPOSABLE_HINTS)).astype(int)

# Deviation from past mean
tx["abs_dev_from_past_mean"] = (tx["amount"] - tx["past_avg_amount"]).abs()

# ---- Extra leakage-safe features ----
# Velocity: prior tx count in 10m/1h/24h per user (INDEX-SAFE, NO groupby.apply returning MultiIndex)
windows = [(600, "10m"), (3600, "1h"), (86400, "24h")]
tx = tx.sort_values(["email","created_at"]).reset_index(drop=True)
email_groups = tx.groupby("email").groups  # dict: email -> Int64Index
for win_s, name in windows:
    counts = pd.Series(0, index=tx.index, dtype="int32")
    for _, idx in email_groups.items():
        s = tx.loc[idx, "created_at"]
        c = rolling_count_seconds(s, win_s)  # indexed by idx already
        counts.loc[idx] = c.values
    tx[f"past_tx_count_{name}"] = counts

# Success proxy (replace with real "success" if available)
if "success" not in tx.columns:
    tx["success"] = ((1 - tx["refunded"]).astype(int))

# Recency since last success/failure (index-safe via per-group loop) — make float to avoid dtype warning
tx["sec_since_last_success"] = 999999.0
tx["sec_since_last_failure"] = 999999.0
for _, idx in email_groups.items():
    dfg = tx.loc[idx, ["created_at","success"]].copy()
    # success
    last_time = None
    vals = []
    for i in range(len(dfg)):
        vals.append((dfg["created_at"].iloc[i] - last_time).total_seconds() if last_time else 999999.0)
        if dfg["success"].iloc[i] == 1:
            last_time = dfg["created_at"].iloc[i]
    tx.loc[idx, "sec_since_last_success"] = vals

    # failure (proxy = 1 - success)
    dfg["failure"] = (1 - dfg["success"]).astype(int)
    last_time = None
    vals = []
    for i in range(len(dfg)):
        vals.append((dfg["created_at"].iloc[i] - last_time).total_seconds() if last_time else 999999.0)
        if dfg["failure"].iloc[i] == 1:
            last_time = dfg["created_at"].iloc[i]
    tx.loc[idx, "sec_since_last_failure"] = vals

# Amount behavior: rolling median/std + z-score vs past (reset_index to drop group level)
roll = tx.groupby("email")["amount"].shift().groupby(tx["email"]).rolling(10, min_periods=1)
tx["past_amount_median_10"] = roll.median().reset_index(level=0, drop=True).fillna(0)
tx["past_amount_std_10"]    = roll.std().reset_index(level=0, drop=True).fillna(0)
tx["amount_zscore_10"] = (tx["amount"] - tx["past_amount_median_10"]) / (tx["past_amount_std_10"] + 1e-6)


# User x Gateway
tx = tx.sort_values(["email","gateway","created_at"]).reset_index(drop=True)
ug = tx.groupby(["email","gateway"])
tx["ug_tx_count"]      = ug.cumcount()
tx["ug_success_cum"]   = ug["success"].transform(lambda s: s.shift(fill_value=0).cumsum())
tx["ug_approval_rate"] = (tx["ug_success_cum"] / tx["ug_tx_count"].replace(0, np.nan)).fillna(0)

# Gateway global
tx = tx.sort_values(["gateway","created_at"]).reset_index(drop=True)
gg = tx.groupby("gateway")
tx["g_tx_count"]      = gg.cumcount()
tx["g_success_cum"]   = gg["success"].transform(lambda s: s.shift(fill_value=0).cumsum())
tx["g_approval_rate"] = (tx["g_success_cum"] / tx["g_tx_count"].replace(0, np.nan)).fillna(0)

tx = tx.sort_values(["fingerprint","created_at"]).reset_index(drop=True)
tx["fp_unique_emails_before"] = 0
for _, idx in tx.groupby("fingerprint").groups.items():
    s = tx.loc[idx, "email"].astype(str)
    uniq_counts = []
    seen = set()
    for v in s.shift().fillna("").tolist():
        if v != "": seen.add(v)
        uniq_counts.append(len(seen))
    tx.loc[idx, "fp_unique_emails_before"] = uniq_counts

tx = tx.sort_values(["ip_address","created_at"]).reset_index(drop=True)
tx["ip_unique_emails_before"] = 0
for _, idx in tx.groupby("ip_address").groups.items():
    s = tx.loc[idx, "email"].astype(str)
    uniq_counts = []
    seen = set()
    for v in s.shift().fillna("").tolist():
        if v != "": seen.add(v)
        uniq_counts.append(len(seen))
    tx.loc[idx, "ip_unique_emails_before"] = uniq_counts

tx = tx.sort_values(["email","created_at"]).reset_index(drop=True)
tx["dominant_card_country_before"] = "UNK"
for _, idx in tx.groupby("email").groups.items():
    dfg = tx.loc[idx, ["card_country"]]
    ser = dominant_country_before_indexsafe(dfg)
    tx.loc[idx, "dominant_card_country_before"] = ser.values

tx["dominant_country_mismatch"] = (tx["card_country"] != tx["dominant_card_country_before"]).astype(int)

# Interactions
tx["country_mismatch_x_amount_spike"] = tx["country_mismatch"] * (tx["amount_zscore_10"] > 3).astype(int)
tx["vpn_flag"] = 0  # placeholder; set via IP intel if available

# Back to global chronological order for splitting
tx = tx.sort_values("created_at").reset_index(drop=True)

y = tx["disputed"].astype(int)

base_features = [
    "amount_log","hour","is_weekend",
    "past_tx_count","past_disputes","past_refunds","past_avg_amount",
    "time_since_prev_tx_sec",
    "country_mismatch",
    "ip_reuse_before","fp_reuse_before","fp_ip_pair_reuse_before",
    "email_domain_risk","email_domain_class",
    "abs_dev_from_past_mean",
    "risk_score", 
    "account_age_days","total_transactions","total_disputes","total_refunds","avg_transaction_amount","high_risk_tag"
]
extra_features = [
    "past_tx_count_10m","past_tx_count_1h","past_tx_count_24h",
    "sec_since_last_success","sec_since_last_failure",
    "past_amount_median_10","past_amount_std_10","amount_zscore_10",
    "ug_tx_count","ug_approval_rate","g_tx_count","g_approval_rate",
    "fp_unique_emails_before","ip_unique_emails_before",
    "dominant_country_mismatch",
    "country_mismatch_x_amount_spike","vpn_flag"
]
feature_cols = [c for c in (base_features + extra_features) if c in tx.columns]
X = tx[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

cut_time_test = tx["created_at"].quantile(0.80)
train_mask = tx["created_at"] <= cut_time_test
test_mask  = tx["created_at"]  > cut_time_test

X_train_full, y_train_full = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

# validation is last 10% of train period
cut_time_val = tx.loc[train_mask, "created_at"].quantile(0.90)
val_mask = (tx["created_at"] > cut_time_val) & train_mask
tr_mask  = (tx["created_at"] <= cut_time_val) & train_mask

X_train, y_train = X[tr_mask], y[tr_mask]
X_val,   y_val   = X[val_mask], y[val_mask]

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# ---------------------------
# 5) Pipeline + CV (no leakage)
# ---------------------------
pipe = ImbPipeline(steps=[
    ("ros",    RandomOverSampler(random_state=42)),
    ("scaler", StandardScaler()),
    ("xgb",    XGBClassifier(
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

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="average_precision")
print(f"CV PR-AUC (train): mean={cv_scores.mean():.4f}, std={cv_scores.std():.4f}")

# Fit on train → tune threshold on val
pipe.fit(X_train, y_train)
y_val_proba = pipe.predict_proba(X_val)[:, 1]

p, r, t = precision_recall_curve(y_val, y_val_proba)
f1s = (2 * p * r) / (p + r + 1e-9)
best_idx = int(np.nanargmax(f1s)) if len(f1s) else 0
best_threshold = float(t[max(best_idx-1, 0)]) if len(t) else 0.5
print(f"Chosen threshold (val F1): {best_threshold:.3f}")


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
    print(f"{k:>8}: {v:.4f}")

print("\nConfusion matrix (test):")
print(confusion_matrix(y_test, y_test_pred) if len(y_test) else "N/A")

print("\nClassification report (test):")
print(classification_report(y_test, y_test_pred, zero_division=0) if len(y_test) else "N/A")


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
    plt.savefig("/src/data/models/fraud_detection_feature_importance.png")
    plt.show()
except Exception:
    pass

# ---------------------------
# 8) Save artifacts & metadata
# ---------------------------
os.makedirs("/src/data/models", exist_ok=True)
joblib.dump(pipe, "/src/data/models/fraud_detection_pipeline.pkl")

report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0) if len(y_test) else {}
with open("/src/data/models/fraud_detection_report.json", "w") as f:
    json.dump(report, f, indent=4)

metadata = {
    "model_version": "1.3.0",
    "created_at": pd.Timestamp.now().isoformat(),
    "features_used": list(X.columns),
    "threshold": best_threshold,
    "metrics_test": metrics,
    "sizes": {"train": int(len(X_train)), "val": int(len(X_val)), "test": int(len(X_test))},
    "cv_pr_auc_mean_train": float(cv_scores.mean()),
    "cv_pr_auc_std_train": float(cv_scores.std()),
}
with open("/src/data/models/fraud_detection_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("\n✅ Pipeline, report, and metadata saved to /src/data/models")
