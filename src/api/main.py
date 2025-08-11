import os
import json
import joblib
import stripe
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Header, BackgroundTasks, APIRouter
from pydantic import BaseModel
from datetime import datetime
from tensorflow.keras.models import load_model
from pymongo import MongoClient
from dotenv import load_dotenv

# ---------------------------
# Load env & setup
# ---------------------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
MODEL_PATH = "/src/data/models/"

if not MONGO_URI:
    raise RuntimeError("Missing MONGO_URI")
if not STRIPE_SECRET_KEY:
    raise RuntimeError("Missing STRIPE_SECRET_KEY")
if not WEBHOOK_SECRET:
    raise RuntimeError("Missing STRIPE_WEBHOOK_SECRET")

stripe.api_key = STRIPE_SECRET_KEY
client = MongoClient(MONGO_URI)
db = client["payment_intelligence"]
app = FastAPI(title="AI Payment Intelligence API")
router = APIRouter()

# ---------------------------
# Utilities
# ---------------------------
def load_ai_model(name):
    path = os.path.join(MODEL_PATH, name)
    return joblib.load(path) if os.path.exists(path) else None


def sanitize_for_mongo(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_mongo(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_mongo(i) for i in obj]
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, datetime):
        return obj
    elif isinstance(obj, str):
        return obj
    else:
        return str(obj)


def to_native(value):
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def classify_risk_level(confidence, high=0.85, medium=0.5):
    if confidence >= high:
        return "high"
    elif confidence >= medium:
        return "medium"
    return "low"


class TransactionRequest(BaseModel):
    amount: float
    card_country: str
    billing_country: str
    email: str
    risk_score: float
    ip_address: str
    fingerprint: str
    hour: int

# ---------------------------
# Load models & metadata
# ---------------------------
# Flexible loader: supports either (a) combined pipelines or (b) separate scaler+model files

def _load_pipeline_or_legacy(prefix: str, pipeline_name: str, model_name: str, scaler_name: str):
    pipeline = load_ai_model(pipeline_name)
    model = None
    scaler = None
    if pipeline is None:
        model = load_ai_model(model_name)
        scaler = load_ai_model(scaler_name)
    return pipeline, model, scaler

# Fraud
fraud_pipeline, fraud_model, fraud_scaler = _load_pipeline_or_legacy(
    prefix="fraud", 
    pipeline_name="fraud_detection_pipeline.pkl",
    model_name="fraud_detection_model_final.pkl",
    scaler_name="fraud_detection_scaler_final.pkl",
)

# Chargeback
chargeback_pipeline, chargeback_model, chargeback_scaler = _load_pipeline_or_legacy(
    prefix="chargeback",
    pipeline_name="chargeback_pipeline.pkl",
    model_name="chargeback_prediction_model.pkl",
    scaler_name="chargeback_prediction_scaler.pkl",
)

# Subscription (keep legacy names; add optional pipeline fallback if you create one later)
subscription_pipeline = load_ai_model("subscription_revenue_pipeline.pkl")
subscription_model = subscription_pipeline if subscription_pipeline is not None else load_ai_model("subscription_revenue_model.pkl")
subscription_scaler = None if subscription_pipeline is not None else load_ai_model("subscription_revenue_scaler.pkl")

smart_routing_model = load_model(
    os.path.join(MODEL_PATH, "smart_payment_routing_model.h5"),
    compile=False
) if os.path.exists(os.path.join(MODEL_PATH, "smart_payment_routing_model.h5")) else None

fraud_features_path = os.path.join(MODEL_PATH, "fraud_detection_metadata.json")
if os.path.exists(fraud_features_path):
    with open(fraud_features_path) as f:
        fraud_features = json.load(f).get("features_used", [])
else:
    fraud_features = []

# Load chargeback feature order from metadata
chargeback_meta_path = os.path.join(MODEL_PATH, "chargeback_metadata.json")
if os.path.exists(chargeback_meta_path):
    with open(chargeback_meta_path) as f:
        chargeback_features = json.load(f).get("features_used", [])
else:
    # fallback to legacy list if metadata missing
    chargeback_features = [
        'amount_log','hour','is_weekend','ip_address_reuse_before','fingerprint_reuse_before',
        'device_ip_pair_reuse_before','risk_score','email_transaction_count',
        'customer_refund_ratio_past','country_mismatch','ip_country_mismatch','time_between_transactions',
        'email_domain_risk','transaction_amount_diff','past_chargebacks','past_avg_amount'
    ]

# Common email domains (keep consistent with training)
COMMON_DOMAINS = {
    "gmail.com","yahoo.com","hotmail.com","outlook.com","icloud.com",
    "aol.com","protonmail.com","zoho.com","mail.com","gmx.com"
}

# ---------------------------
# FastAPI endpoints
# ---------------------------
@app.get("/")
def root():
    return {"message": "🤖 AI Payment Intelligence API"}


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/webhook/stripe")
async def stripe_webhook(request: Request, background_tasks: BackgroundTasks, stripe_signature: str = Header(None)):
    try:
        payload = await request.body()
        event = stripe.Webhook.construct_event(payload, stripe_signature, WEBHOOK_SECRET)
    except Exception as e:
        return {"error": str(e)}

    event_type = event["type"]
    obj = event["data"]["object"] 

    if event_type == "customer.created":
        address = obj.get("address") or {}
        invoice_settings = obj.get("invoice_settings") or {}
        customer_data = {
            "email": obj.get("email"),
            "name": obj.get("name"),
            "phone": obj.get("phone"),
            "currency": obj.get("currency"),
            "country": address.get("country"),
            "address_line1": address.get("line1"),
            "address_line2": address.get("line2"),
            "city": address.get("city"),
            "state": address.get("state"),
            "postal_code": address.get("postal_code"),
            "created_at": datetime.utcfromtimestamp(obj.get("created")),
            "delinquent": obj.get("delinquent", False),
            "default_payment_method": invoice_settings.get("default_payment_method"),
            "balance": obj.get("balance", 0),
            "tax_info": obj.get("tax_info", {}),
            "metadata": obj.get("metadata", {}),
            "invoice_prefix": obj.get("invoice_prefix"),
            "gateway_customer_ids": {"stripe": obj.get("id")}
        }
        db["customers"].update_one({"email": customer_data["email"]}, {"$set": customer_data}, upsert=True)
        print(f"✅ Customer saved: {customer_data['email']}")

    elif event_type == "customer.subscription.created":
        sub = obj
        items = sub.get("items", {}).get("data", [])
        plan = items[0].get("plan", {}) if items else {}
        sub_data = {
            "subscription_id": sub.get("id"),
            "email": sub.get("metadata", {}).get("user_email", "unknown@example.com"),
            "gateway": "Stripe",
            "status": sub.get("status"),
            "current_period_start": datetime.utcfromtimestamp(sub.get("current_period_start")),
            "current_period_end": datetime.utcfromtimestamp(sub.get("current_period_end")),
            "plan_id": plan.get("id"),
            "plan_name": plan.get("nickname"),
            "product_id": plan.get("product"),
            "price_amount": plan.get("amount", 0) / 100.0,
            "currency": plan.get("currency"),
            "interval": plan.get("interval"),
            "created_at": datetime.utcfromtimestamp(sub.get("created")),
            "cancel_at_period_end": sub.get("cancel_at_period_end"),
            "canceled_at": datetime.utcfromtimestamp(sub.get("canceled_at")) if sub.get("canceled_at") else None,
            "ended_at": datetime.utcfromtimestamp(sub.get("ended_at")) if sub.get("ended_at") else None,
            "trial_start": datetime.utcfromtimestamp(sub.get("trial_start")) if sub.get("trial_start") else None,
            "trial_end": datetime.utcfromtimestamp(sub.get("trial_end")) if sub.get("trial_end") else None,
            "quantity": sub.get("quantity"),
            "metadata": sub.get("metadata"),
            "latest_invoice": sub.get("latest_invoice"),
            "collection_method": sub.get("collection_method"),
            "default_payment_method": sub.get("default_payment_method"),
            "billing_cycle_anchor": datetime.utcfromtimestamp(sub.get("billing_cycle_anchor")) if sub.get("billing_cycle_anchor") else None
        }
        db["subscriptions"].update_one({"subscription_id": sub_data["subscription_id"]}, {"$set": sub_data}, upsert=True)
        print(f"✅ Subscription created: {sub_data['subscription_id']}")

    elif event_type == "customer.subscription.deleted":
        sub_id = obj.get("id")
        db["subscriptions"].update_one({"subscription_id": sub_id}, {"$set": {
            "status": "canceled",
            "canceled_at": datetime.utcnow(),
            "ended_at": datetime.utcnow()
        }})
        print(f"🚫 Subscription canceled: {sub_id}")

    elif event_type == "refund.created":
        charge_id = obj.get("charge")
        db["transactions"].update_one({"transaction_id": charge_id}, {"$set": {
            "refunded": True,
            "amount_refunded": obj.get("amount", 0) / 100.0,
            "refund_created_at": datetime.utcfromtimestamp(obj.get("created"))
        }})
        print(f"💸 Refund created for transaction: {charge_id}")

    elif event_type == "charge.dispute.closed":
        charge_id = obj.get("charge")
        db["transactions"].update_one({"transaction_id": charge_id}, {"$set": {
            "disputed": True,
            "dispute_status": "closed",
            "dispute_closed_at": datetime.utcnow()
        }})
        print(f"⚠️ Dispute closed for transaction: {charge_id}")

    elif event_type == "invoice.paid":
        try:
            invoice = obj
            charge_id = invoice.get("charge")
            charge = stripe.Charge.retrieve(charge_id) if charge_id else {}

            billing = charge.get("billing_details", {})
            card = charge.get("payment_method_details", {}).get("card", {})
            outcome = charge.get("outcome", {})

            transaction = {
                "transaction_id": charge_id,
                "email": invoice.get("customer_email", "unknown@example.com"),
                "amount": invoice.get("amount_paid", 0) / 100.0,
                "currency": invoice.get("currency", "usd"),
                "gateway": "Stripe",
                "status": invoice.get("status", charge.get("status", "unknown")),

                # Payment method
                "payment_method": charge.get("payment_method_details", {}).get("type"),
                "card_brand": card.get("brand"),
                "card_country": card.get("country"),
                "fingerprint": card.get("fingerprint"),
                "funding_type": card.get("funding"),
                "three_d_secure": card.get("three_d_secure"),
                "cvc_check": card.get("checks", {}).get("cvc_check"),
                "address_line1_check": card.get("checks", {}).get("address_line1_check"),
                "postal_code_check": card.get("checks", {}).get("address_postal_code_check"),

                # Risk info
                "risk_level": outcome.get("risk_level", "unknown"),
                "risk_score": outcome.get("risk_score", 0),
                "seller_message": outcome.get("seller_message"),
                "network_status": outcome.get("network_status"),
                "outcome_type": outcome.get("type"),

                # Billing & IP
                "ip_address": card.get("country") or billing.get("address", {}).get("country", "unknown"),
                "billing_name": billing.get("name"),
                "billing_email": billing.get("email"),
                "billing_phone": billing.get("phone"),
                "billing_address_country": billing.get("address", {}).get("country"),
                "billing_address_line1": billing.get("address", {}).get("line1"),
                "billing_address_line2": billing.get("address", {}).get("line2"),
                "billing_address_postal_code": billing.get("address", {}).get("postal_code"),
                "billing_address_city": billing.get("address", {}).get("city"),
                "billing_address_state": billing.get("address", {}).get("state"),
                "refunded": charge.get("refunded", False),
                "amount_refunded": charge.get("amount_refunded", 0) / 100.0,
                "disputed": charge.get("disputed", False),
                "captured": charge.get("captured", False),
                "paid": charge.get("paid", False),
                "created_at": datetime.utcfromtimestamp(charge.get("created")) if charge.get("created") else datetime.utcnow()
            }
            
            # Save to DB
            db["transactions"].update_one(
                {"transaction_id": transaction["transaction_id"]},
                {"$set": sanitize_for_mongo(transaction)},
                upsert=True
            )

            background_tasks.add_task(process_fraud_workflow, transaction)
            background_tasks.add_task(predict_and_store_chargeback, transaction)

            return {"status": "queued", "invoice_id": invoice["id"]}

        except Exception as e:
            print("❌ invoice.paid error:", e)
            raise HTTPException(status_code=500, detail=f"invoice.paid error: {str(e)}")

        except Exception as e:
            print("❌ invoice.paid error:", e)
            return {"error": str(e)}
        
        
    elif event_type == "refund.created":
        try:
            charge_id = obj.get("charge")
            refund_amount = obj.get("amount", 0) / 100.0
            refund_reason = obj.get("reason", "not_provided")
            refund_created_at = datetime.utcfromtimestamp(obj.get("created"))

            db["transactions"].update_one(
                {"transaction_id": charge_id},
                {"$set": {
                    "refunded": True,
                    "amount_refunded": refund_amount,
                    "refund_reason": refund_reason,
                    "refund_created_at": refund_created_at
                }},
                upsert=True
            )

            refund_doc = {
                "transaction_id": charge_id,
                "refund_id": obj.get("id"),
                "amount_refunded": refund_amount,
                "reason": refund_reason,
                "status": obj.get("status", "succeeded"),
                "created_at": refund_created_at,
                "currency": obj.get("currency", "usd"),
                "metadata": obj.get("metadata", {}),
                "gateway": "Stripe"
            }

            db["refunds"].update_one(
                {"refund_id": refund_doc["refund_id"]},
                {"$set": sanitize_for_mongo(refund_doc)},
                upsert=True
            )

            print(f"💸 Refund recorded: {refund_doc['refund_id']} for transaction {charge_id}")

        except Exception as e:
            print("❌ Refund handling failed:", str(e))
            
            
    elif event_type == "charge.dispute.created":
        try:
            charge_id = obj.get("charge")
            dispute_id = obj.get("id")
            reason = obj.get("reason", "unspecified")
            amount = obj.get("amount", 0) / 100.0
            status = obj.get("status", "needs_response")
            created_at = datetime.utcfromtimestamp(obj.get("created"))

            db["transactions"].update_one(
                {"transaction_id": charge_id},
                {"$set": {
                    "disputed": True,
                    "dispute_status": status,
                    "dispute_reason": reason,
                    "dispute_id": dispute_id,
                    "dispute_amount": amount,
                    "dispute_created_at": created_at
                }},
                upsert=True
            )

            db["chargebacks"].update_one(
                {"dispute_id": dispute_id},
                {"$set": {
                    "dispute_id": dispute_id,
                    "transaction_id": charge_id,
                    "amount": amount,
                    "reason": reason,
                    "status": status,
                    "created_at": created_at,
                    "currency": obj.get("currency", "usd"),
                    "evidence_due_by": datetime.utcfromtimestamp(obj.get("evidence_due_by")) if obj.get("evidence_due_by") else None
                }},
                upsert=True
            )

            print(f"⚠️ Dispute created: {dispute_id} for transaction {charge_id}")

        except Exception as e:
            print("❌ Dispute created handling failed:", str(e))
            
            
    elif event_type == "charge.dispute.closed":
        try:
            charge_id = obj.get("charge")
            dispute_id = obj.get("id")
            outcome = obj.get("status") 

            db["transactions"].update_one(
                {"transaction_id": charge_id},
                {"$set": {
                    "dispute_status": "closed",
                    "dispute_outcome": outcome,
                    "dispute_closed_at": datetime.utcnow()
                }}
            )

            db["chargebacks"].update_one(
                {"dispute_id": dispute_id},
                {"$set": {
                    "status": "closed",
                    "outcome": outcome,
                    "closed_at": datetime.utcnow()
                }}
            )

            print(f"✅ Dispute closed: {dispute_id}, outcome: {outcome}")

        except Exception as e:
            print("❌ Dispute closed handling failed:", str(e))
            
    else:
        print(f"🔍 Unhandled event type: {event_type}")

    return {"status": 200, "message": f"Event {event_type} processed successfully"}



# ---------------------------
# Fraud Prediction
# ---------------------------
def process_fraud_workflow(transaction):
    try:
        print("🚀 Running fraud check for:", transaction.get("transaction_id"))
        model_input = TransactionRequest(**{
            "amount": transaction.get("amount", 0.0),
            "card_country": transaction.get("card_country", "UNK"),
            "billing_country": transaction.get("billing_address_country", "UNK"),
            "email": transaction.get("email", "unknown@example.com"),
            "risk_score": float(transaction.get("risk_score", 0.0) or 0.0),
            "ip_address": transaction.get("ip_address", "0.0.0.0"),
            "fingerprint": transaction.get("fingerprint", "unknown"),
            "hour": datetime.utcnow().hour,
        })
        prediction = run_fraud_prediction(model_input)
        confidence = float(prediction["confidence"]) if "confidence" in prediction else 0.0
        risk_level = classify_risk_level(confidence)

        result_doc = {
            "transaction_id": transaction.get("transaction_id"),
            "email": transaction.get("email"),
            "fraud_prediction": {
                "detected": bool(prediction.get("fraud_detected", False)),
                "confidence_score": confidence,
                "risk_level": risk_level,
                "reasons": prediction.get("reasons", []),
                "thresholds": {"high_risk": 0.85, "medium_risk": 0.5},
            },
            "features_used": sanitize_for_mongo(prediction.get("features_used", {})),
            "model_info": {"name": "fraud_detection_model_final.pkl", "version": "v1.0", "run_time": datetime.utcnow().isoformat()},
            "created_at": datetime.utcnow(),
        }
        db["fraud_results"].update_one(
            {"transaction_id": transaction.get("transaction_id")},
            {"$set": result_doc},
            upsert=True
        )
        rec = build_recommendations(transaction, prediction, None)
        db["transactions"].update_one(
            {"transaction_id": transaction.get("transaction_id")},
            {"$set": {"recommendations": rec, "updated_at": datetime.utcnow()}},
            upsert=True
        )
        print("✅ Improved fraud result saved.")
    except Exception as e:
        print("❌ Fraud processing failed:", e)


def _email_domain_risk(email: str) -> int:
    dom = str(email).split("@")[-1].lower() if "@" in str(email) else "unknown"
    return 0 if dom in COMMON_DOMAINS else 1


def run_fraud_prediction(req: TransactionRequest):
    # Accept either a full pipeline OR (model + scaler)
    if fraud_pipeline is None and fraud_model is None:
        raise HTTPException(status_code=500, detail="Fraud model not loaded")
    # If using separate model, ensure scaler exists
    if fraud_pipeline is None and fraud_scaler is None:
        raise HTTPException(status_code=500, detail="Fraud scaler not loaded for legacy model")

        raise HTTPException(status_code=500, detail="Fraud model or scaler not loaded")

    try:
        now = datetime.utcnow()
        # Pull recent history (exclude current row; we will add current separately but compute features from history only)
        recent_txn = db["transactions"].find({
            "email": req.email
        }).sort("created_at", -1).limit(200)
        rows = list(recent_txn)
        for r in rows:
            r["created_at"] = r.get("created_at", now)

        # Build history and current
        history_df = pd.DataFrame(rows)
        cur = {
            "amount": req.amount,
            "risk_score": req.risk_score,
            "card_country": req.card_country,
            "billing_address_country": req.billing_country,
            "email": req.email,
            "ip_address": req.ip_address,
            "fingerprint": req.fingerprint,
            "refunded": False,
            "disputed": False,
            "transaction_id": "latest_txn",
            "created_at": now,
        }

        # Compute features PAST-ONLY (from history_df)
        def safe_mean(s):
            return float(s.mean()) if len(s) else 0.0
        def safe_sum(s):
            return int(s.sum()) if len(s) else 0
        def safe_count(df):
            return int(len(df))

        # Per-entity slices
        h_email = history_df[history_df.get("email", "") == req.email] if not history_df.empty else pd.DataFrame()
        h_ip = history_df[history_df.get("ip_address", "") == req.ip_address] if not history_df.empty else pd.DataFrame()
        h_fp = history_df[history_df.get("fingerprint", "") == req.fingerprint] if not history_df.empty else pd.DataFrame()
        h_pair = history_df[(history_df.get("fingerprint", "") == req.fingerprint) & (history_df.get("ip_address", "") == req.ip_address)] if not history_df.empty else pd.DataFrame()

        last_time = pd.to_datetime(h_email["created_at"]).max() if not h_email.empty else None
        time_between = (now - last_time).total_seconds() if last_time is not None else 999999.0

        # Quantile for unusual amount: use per-customer if enough data else global
        if not h_email.empty and len(h_email) >= 20:
            q99 = float(pd.to_numeric(h_email.get("amount", pd.Series(dtype=float)), errors="coerce").quantile(0.99))
        else:
            q99 = float(pd.to_numeric(history_df.get("amount", pd.Series(dtype=float)), errors="coerce").quantile(0.99)) if not history_df.empty else 0.0

        avg_amount_email = safe_mean(pd.to_numeric(h_email.get("amount", pd.Series(dtype=float)), errors="coerce"))
        email_txn_count = safe_count(h_email)
        email_dispute_count = safe_sum(pd.to_numeric(h_email.get("disputed", pd.Series(dtype=int)), errors="coerce"))
        email_refund_count = safe_sum(pd.to_numeric(h_email.get("refunded", pd.Series(dtype=int)), errors="coerce"))
        chargeback_rate = (email_dispute_count / email_txn_count) if email_txn_count else 0.0
        refund_ratio = (email_refund_count / email_txn_count) if email_txn_count else 0.0

        features = {
            # Basic
            "amount_log": float(np.log1p(req.amount)),
            "hour": int(req.hour),
            "is_weekend": int(datetime.utcnow().weekday() >= 5),
            "ip_address_reuse_count": safe_count(h_ip),
            "fingerprint_reuse_count": safe_count(h_fp),
            "device_ip_pair_reuse_count": safe_count(h_pair),
            "risk_score": float(req.risk_score),
            "email_transaction_count": email_txn_count,
            "customer_refund_ratio": float(refund_ratio),
            "country_mismatch": int(req.card_country != req.billing_country),
            "ip_country_mismatch": 0,
            "time_between_transactions": float(time_between),
            "email_domain_risk": int(_email_domain_risk(req.email)),
            "transaction_amount_diff": float(abs(req.amount - avg_amount_email)),
            "past_chargebacks": int(email_dispute_count),
            "unusual_amount_flag": int(req.amount > q99 if q99 > 0 else 0),
            "shared_card_email_count": int(h_fp["email"].nunique() if not h_fp.empty else 0),
            "shared_ip_email_count": int(h_ip["email"].nunique() if not h_ip.empty else 0),
            "previous_risk_scores_avg": float(safe_mean(pd.to_numeric(h_email.get("risk_score", pd.Series(dtype=float)), errors="coerce"))),
            "number_of_risky_transactions": int((pd.to_numeric(h_email.get("risk_score", pd.Series(dtype=float)), errors="coerce") > 70).sum()) if not h_email.empty else 0,
        }

        # Align to model feature order
        X = pd.DataFrame([{k: features.get(k, 0) for k in fraud_features}], columns=fraud_features)
        X = X.replace([np.inf, -np.inf], 0).fillna(0)
        if fraud_pipeline is not None:
            proba = float(fraud_pipeline.predict_proba(X)[0][1])
            pred = int(fraud_pipeline.predict(X)[0])
        else:
            X_scaled = fraud_scaler.transform(X)
            proba = float(fraud_model.predict_proba(X_scaled)[0][1])
            pred = int(fraud_model.predict(X_scaled)[0])

        # Human-readable reasons (rule layer)
        reasons = []
        rule_checks = [
            (features.get("unusual_amount_flag", 0) == 1, "Transaction amount is in the top 1% of customer's history"),
            (features.get("country_mismatch", 0) == 1, "Card country and billing country do not match"),
            (features.get("ip_country_mismatch", 0) == 1, "Card country and IP country do not match"),
            (features.get("ip_address_reuse_count", 0) > 5, "IP address has been used in more than 5 previous transactions"),
            (features.get("fingerprint_reuse_count", 0) > 3, "Device fingerprint has been reused in more than 3 transactions"),
            (features.get("device_ip_pair_reuse_count", 0) > 3, "Same device and IP pair has been used in more than 3 transactions"),
            (features.get("email_domain_risk", 0) == 1, "Email domain is uncommon or potentially risky"),
            (features.get("customer_refund_ratio", 0.0) > 0.3, "Customer has a high historical refund ratio (>30%)"),
            (features.get("past_chargebacks", 0) > 0, "Previous chargebacks found for this customer"),
            (features.get("time_between_transactions", 999999.0) < 30, "Very short time between transactions (<30 seconds)"),
            (features.get("transaction_amount_diff", 0.0) > 100, "Transaction amount deviates significantly from customer's average"),
            (features.get("previous_risk_scores_avg", 0.0) > 70, "Customer has a high average Stripe risk score (>70)"),
            (features.get("shared_card_email_count", 0) > 2, "Device fingerprint is associated with multiple emails"),
            (features.get("shared_ip_email_count", 0) > 2, "IP address is associated with multiple emails"),
            (features.get("number_of_risky_transactions", 0) > 2, "Customer has multiple previous high-risk transactions"),
        ]
        for cond, msg in rule_checks:
            if cond:
                reasons.append(msg)

        return {
            "fraud_detected": bool(pred),
            "confidence": round(proba, 4),
            "reasons": reasons,
            "features_used": {k: float(X.iloc[0][k]) for k in fraud_features},
        }

    except Exception as e:
        print("❌ Prediction error:", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/fraud")
def predict_fraud(req: TransactionRequest):
    try:
        return run_fraud_prediction(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------
# Chargeback Prediction
# ---------------------------
def predict_and_store_chargeback(transaction):
    try:
        req = TransactionRequest(
            amount=transaction.get("amount", 0.0),
            card_country=transaction.get("card_country", "UNK"),
            billing_country=transaction.get("billing_address_country", "UNK"),
            email=transaction.get("email", "unknown@example.com"),
            risk_score=float(transaction.get("risk_score", 0.0) or 0.0),
            ip_address=transaction.get("ip_address", "0.0.0.0"),
            fingerprint=transaction.get("fingerprint", "unknown"),
            hour=datetime.utcnow().hour,
        )
        prediction = predict_chargeback(req)
        safe_prediction = {k: to_native(v) for k, v in prediction.items()}
        safe_prediction["transaction_id"] = transaction.get("transaction_id")
        safe_prediction["email"] = transaction.get("email")
        safe_prediction["created_at"] = datetime.utcnow()
        db["chargeback_predictions"].update_one({"transaction_id": transaction.get("transaction_id")}, {"$set": safe_prediction}, upsert=True)
        rec = build_recommendations(transaction, None, prediction)
        db["transactions"].update_one(
            {"transaction_id": transaction.get("transaction_id")},
            {"$set": {"recommendations": rec, "updated_at": datetime.utcnow()}},
            upsert=True
        )
        print("✅ Chargeback prediction saved.")
        update_transaction_with_chargeback(transaction.get("transaction_id"), prediction, transaction.get("email"))
        print(f"🔍 Chargeback prediction stored for {transaction.get('transaction_id')}")
    except Exception as e:
        print("❌ Chargeback prediction error:", e)


@app.get("/jobs/predict-chargebacks")
def run_chargeback_predictions_job():
    count = 0
    cursor = db["transactions"].find({"chargeback_predicted": {"$exists": False}})
    for txn in cursor:
        try:
            req = TransactionRequest(
                amount=txn.get("amount", 0.0),
                card_country=txn.get("card_country", "UNK"),
                billing_country=txn.get("billing_address_country", "UNK"),
                email=txn.get("email", "unknown@example.com"),
                risk_score=float(txn.get("risk_score", 0.0) or 0.0),
                ip_address=txn.get("ip_address", "0.0.0.0"),
                fingerprint=txn.get("fingerprint", "unknown"),
                hour=(txn.get("created_at") or datetime.utcnow()).hour,
            )
            prediction = predict_chargeback(req)
            safe_prediction = {k: to_native(v) for k, v in prediction.items()}
            safe_prediction["transaction_id"] = txn.get("transaction_id")
            safe_prediction["email"] = txn.get("email")
            safe_prediction["created_at"] = datetime.utcnow()
            db["chargeback_predictions"].update_one({"transaction_id": txn.get("transaction_id")}, {"$set": safe_prediction}, upsert=True)
            print("✅ Chargeback prediction saved.")
            update_transaction_with_chargeback(txn.get("transaction_id"), prediction, txn.get("email"))
            count += 1
        except Exception as e:
            print(f"❌ Error predicting for {txn.get('transaction_id')}: {str(e)}")
    return {"message": f"✅ Chargeback prediction run for {count} transactions"}


def predict_chargeback(req: TransactionRequest):
    # Accept either a full pipeline OR (model + scaler)
    if chargeback_pipeline is None and chargeback_model is None:
        raise HTTPException(status_code=500, detail="Chargeback model not loaded")
    if chargeback_pipeline is None and chargeback_scaler is None:
        raise HTTPException(status_code=500, detail="Chargeback scaler not loaded for legacy model")
    try:
        now = datetime.utcnow()
        recent_txn = db["transactions"].find({"email": req.email}).sort("created_at", -1).limit(200)
        rows = list(recent_txn)
        for r in rows:
            r["created_at"] = r.get("created_at", now)
        history_df = pd.DataFrame(rows)

        # Past-only aggregates
        def safe_mean(s):
            return float(pd.to_numeric(s, errors='coerce').mean()) if len(s) else 0.0
        def safe_sum(s):
            return int(pd.to_numeric(s, errors='coerce').sum()) if len(s) else 0
        def safe_count(df):
            return int(len(df))

        h_email = history_df[history_df.get("email", "") == req.email] if not history_df.empty else pd.DataFrame()
        h_ip = history_df[history_df.get("ip_address", "") == req.ip_address] if not history_df.empty else pd.DataFrame()
        h_fp = history_df[history_df.get("fingerprint", "") == req.fingerprint] if not history_df.empty else pd.DataFrame()
        h_pair = history_df[(history_df.get("fingerprint", "") == req.fingerprint) & (history_df.get("ip_address", "") == req.ip_address)] if not history_df.empty else pd.DataFrame()

        last_time = pd.to_datetime(h_email["created_at"]).max() if not h_email.empty else None
        time_between = (now - last_time).total_seconds() if last_time is not None else 999999.0
        avg_amount_email = safe_mean(h_email.get("amount", pd.Series(dtype=float)))
        email_txn_count = safe_count(h_email)
        email_dispute_count = safe_sum(h_email.get("disputed", pd.Series(dtype=int)))
        email_refund_count = safe_sum(h_email.get("refunded", pd.Series(dtype=int)))
        refund_ratio_past = (email_refund_count / email_txn_count) if email_txn_count else 0.0

        # Build feature dict with names EXPECTED by the trained model
        features = {
            'amount_log': float(np.log1p(req.amount)),
            'hour': int(datetime.utcnow().hour),
            'is_weekend': int(datetime.utcnow().weekday() >= 5),
            'ip_address_reuse_before': safe_count(h_ip),
            'fingerprint_reuse_before': safe_count(h_fp),
            'device_ip_pair_reuse_before': safe_count(h_pair),
            'risk_score': float(req.risk_score),
            'email_transaction_count': email_txn_count,
            'customer_refund_ratio_past': float(refund_ratio_past),
            'country_mismatch': int(req.card_country != req.billing_country),
            'ip_country_mismatch': 0,  # only if you enrich ip_country
            'time_between_transactions': float(time_between),
            'email_domain_risk': int(_email_domain_risk(req.email)),
            'transaction_amount_diff': float(abs(req.amount - avg_amount_email)),
            'past_chargebacks': int(email_dispute_count),
            'past_avg_amount': float(avg_amount_email),
        }

        # Align to model's feature order
        X = pd.DataFrame([{k: features.get(k, 0) for k in chargeback_features}], columns=chargeback_features)
        X = X.replace([np.inf, -np.inf], 0).fillna(0)

        if chargeback_pipeline is not None:
            confidence = float(chargeback_pipeline.predict_proba(X)[0][1])
            prediction = int(chargeback_pipeline.predict(X)[0])
        else:
            X_scaled = chargeback_scaler.transform(X)
            confidence = float(chargeback_model.predict_proba(X_scaled)[0][1])
            prediction = int(chargeback_model.predict(X_scaled)[0])

        reasons = []
        if features['customer_refund_ratio_past'] > 0.5:
            reasons.append("Customer has a high historical refund ratio (>50%)")
        if features['device_ip_pair_reuse_before'] > 3:
            reasons.append("Device/IP pair has been reused in multiple transactions")
        if features['country_mismatch'] == 1:
            reasons.append("Card country and billing country do not match")
        if features['email_domain_risk'] == 1:
            reasons.append("Email domain is uncommon or potentially risky")
        if features['transaction_amount_diff'] > max(100, features['amount_log']):
            reasons.append("Transaction amount is unusually different from customer's average")
        if features['email_transaction_count'] > 10:
            reasons.append("Email has an unusually high number of transactions")
        if features['fingerprint_reuse_before'] > 5:
            reasons.append("Device fingerprint has been used for many transactions")
        if features['ip_address_reuse_before'] > 5:
            reasons.append("IP address has been used for many transactions")
        if features['risk_score'] > 70:
            reasons.append("Stripe risk score is high (>70)")
        if features['time_between_transactions'] < 60:
            reasons.append("Very short time between transactions (<60s)")
        if features.get('past_chargebacks', features.get('past_chargeback', 0)) > 0:
            reasons.append("Previous chargebacks found for this user")
        if features['amount_log'] > 8:
            reasons.append("Transaction amount is extremely high")

        return {"chargeback_predicted": bool(prediction), "confidence_score": round(confidence, 4), "chargeback_reason": ", ".join(reasons) if reasons else "No strong indicators"}

    except Exception as e:
        print("❌ Chargeback prediction error:", e)
        raise HTTPException(status_code=500, detail=str(e))


def predict_and_store_subscription_revenue(transaction):
    try:
        req = TransactionRequest(
            amount=transaction.get("amount", 0.0),
            card_country=transaction.get("card_country", "UNK"),
            billing_country=transaction.get("billing_address_country", "UNK"),
            email=transaction.get("email", "unknown@example.com"),
            risk_score=float(transaction.get("risk_score", 0.0) or 0.0),
            ip_address=transaction.get("ip_address", "0.0.0.0"),
            fingerprint=transaction.get("fingerprint", "unknown"),
            hour=datetime.utcnow().hour,
        )
            # Your predict_subscription_revenue(...) would go here
        prediction = predict_subscription_revenue(req)
        
        # Persist as needed
        pass
    except Exception as e:
        print("❌ Revenue forecasting failed:", e)
        
def predict_subscription_revenue(req: TransactionRequest):
    if subscription_pipeline is None and subscription_model is None:
        raise HTTPException(status_code=500, detail="Subscription revenue model not loaded")
    if subscription_pipeline is None and subscription_scaler is None:
        raise HTTPException(status_code=500, detail="Subscription scaler not loaded for legacy model")
    try:
        now = datetime.utcnow()
        # Example feature engineering for subscription revenue prediction
        features = {
            'amount_log': float(np.log1p(req.amount)),
            'hour': int(now.hour),
            'is_weekend': int(now.weekday() >= 5),
            'risk_score': float(req.risk_score),
            'email_domain_risk': int(_email_domain_risk(req.email)),
            "customer_country_mismatch": int(req.card_country != req.billing_country),
            "hour_of_day": int(now.hour),
            "week_of_year": int(now.isocalendar()[1]),
            "is_night": int(now.hour < 6 or now.hour > 22),
            "email_length": int(len(req.email)),
            "risk_score_squared": float(req.risk_score ** 2),
            "email_at_gmail": int(req.email.lower().endswith("@gmail.com")),
            "ip_address_length": int(len(req.ip_address)),
            "fingerprint_length": int(len(req.fingerprint)),
        }
        subscription_meta_path = os.path.join(MODEL_PATH, "subscription_metadata.json")
        if os.path.exists(subscription_meta_path):
            with open(subscription_meta_path) as f:
                subscription_features = json.load(f).get("features_used", list(features.keys()))
        else:
            subscription_features = list(features.keys())

        X = pd.DataFrame([{k: features.get(k, 0) for k in subscription_features}], columns=subscription_features)
        X = X.replace([np.inf, -np.inf], 0).fillna(0)

        if subscription_pipeline is not None:
            prediction = float(subscription_pipeline.predict(X)[0])
        else:
            X_scaled = subscription_scaler.transform(X)
            prediction = float(subscription_model.predict(X_scaled)[0])

        return {"predicted_subscription_revenue": round(prediction, 2)}

    except Exception as e:
        print("❌ Subscription revenue prediction error:", e)
        raise HTTPException(status_code=500, detail=str(e))
    
    
# ---------------------------
# Recommendation engine
# ---------------------------
def _priority_from_risk(fraud_risk_level: str, cb_conf: float) -> str:
    # map to a single priority flag
    if fraud_risk_level == "high" or cb_conf >= 0.85:
        return "critical"
    if fraud_risk_level == "medium" or cb_conf >= 0.5:
        return "high"
    if cb_conf >= 0.2:
        return "medium"
    return "low"

def _uniq(seq):
    return list(dict.fromkeys([s for s in seq if s]))  # keep order, drop empties

def build_recommendations(transaction: dict,
                          fraud_pred: dict | None,
                          chargeback_pred: dict | None) -> dict:
    """
    Returns a recommendation payload to store on the transaction.
    """
    txid = transaction.get("transaction_id")
    amount = float(transaction.get("amount", 0.0))
    currency = transaction.get("currency", "usd")

    # Inputs
    fraud_detected = bool(fraud_pred.get("fraud_detected", False)) if fraud_pred else False
    fraud_conf = float(fraud_pred.get("confidence", 0.0)) if fraud_pred else 0.0
    fraud_level = "high" if fraud_conf >= 0.85 else "medium" if fraud_conf >= 0.5 else "low"

    cb_pred = bool(chargeback_pred.get("chargeback_predicted", False)) if chargeback_pred else False
    cb_conf = float(chargeback_pred.get("confidence_score", 0.0)) if chargeback_pred else 0.0

    # Human-facing reasons (merge & dedup)
    reasons = []
    if fraud_pred:
        reasons.extend(fraud_pred.get("reasons", []))
    if chargeback_pred and chargeback_pred.get("chargeback_reason"):
        reasons.extend([s.strip() for s in chargeback_pred["chargeback_reason"].split(",")])

    actions = []

    # 1) Payment safety controls
    if fraud_detected or fraud_level in ("medium", "high"):
        actions.append("Require 3DS / step-up authentication")
        actions.append("Hold for manual review before capture")
        actions.append("Collect and verify billing address (AVS), postal code, and CVC")
        actions.append("Delay fulfillment; require signature on delivery if physical goods")
        actions.append("Auto-cancel if additional KYC fails within 24h")

    # 2) Routing suggestions (simple heuristics; you can replace with smart_routing_model if present)
    # Examples: prefer gateway with stronger risk controls or lower dispute rate
    actions.append("Route future attempts for this email/device to a gateway with stronger risk controls")
    actions.append("Throttle repeat attempts (>3 in 10m) and add velocity checks")

    # 3) Reputation/velocity mitigations
    if transaction.get("ip_address"):
        actions.append("Temporarily block or rate-limit this IP if repeated high-risk attempts")
    if transaction.get("fingerprint"):
        actions.append("Flag device fingerprint for enhanced screening")
    if transaction.get("email"):
        actions.append("Place customer on watchlist; require verified email domain for higher amounts")

    # 4) Chargeback-specific playbook
    if cb_pred or cb_conf >= 0.5:
        actions.append("Require proof-of-identity (photo ID) or SCA on next purchase")
        actions.append("Use address + delivery confirmation evidence templates")
        actions.append("Reduce high-value single-charge exposure (split payments or escrow)")

    # 5) Tiering by amount
    if amount >= 500:
        actions.append("Enable step-up checks for amounts >= 500 " + currency.upper())
    if amount >= 2000:
        actions.append("Require manual review for amounts >= 2000 " + currency.upper())

    priority = _priority_from_risk(fraud_level, cb_conf)

    return {
        "transaction_id": txid,
        "created_at": datetime.utcnow(),
        "priority": priority,
        "summary": {
            "fraud_detected": fraud_detected,
            "fraud_confidence": round(fraud_conf, 4),
            "fraud_level": fraud_level,
            "chargeback_predicted": cb_pred,
            "chargeback_confidence": round(cb_conf, 4),
            "amount": amount,
            "currency": currency,
        },
        "reasons": _uniq(reasons),
        "recommended_actions": _uniq(actions)[:12], 
        "ttl_days": 30
    }

@app.get("/transactions/{txid}/recommendations")
def get_recommendations(txid: str):
    doc = db["transactions"].find_one({"transaction_id": txid}, {"_id": 0, "recommendations": 1})
    if not doc or "recommendations" not in doc:
        raise HTTPException(404, "No recommendations for this transaction")
    return doc["recommendations"]


# ---------------------------
# Shared helpers
# ---------------------------

def update_transaction_with_chargeback(transaction_id, prediction_result, email):
    try:
        chargeback_predicted = to_native(prediction_result.get("chargeback_predicted", False))
        confidence_score = to_native(prediction_result.get("confidence_score", 0.0))
        db["transactions"].update_one({"transaction_id": transaction_id}, {"$set": {"chargeback_predicted": chargeback_predicted, "chargeback_confidence": confidence_score, "email": email, "updated_at": datetime.utcnow()}}, upsert=True)
        print("✅ Chargeback prediction saved in `transactions`.")
    except Exception as e:
        print(f"❌ Failed to update `transactions`: {e}")


# Optional: expose router
app.include_router(router)
