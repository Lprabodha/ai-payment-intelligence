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

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
MODEL_PATH = "/src/data/models/"

stripe.api_key = STRIPE_SECRET_KEY
client = MongoClient(MONGO_URI)
db = client["payment_intelligence"]
app = FastAPI(title="AI Payment Intelligence API")
router = APIRouter()

def load_ai_model(name):
    path = os.path.join(MODEL_PATH, name)
    return joblib.load(path) if os.path.exists(path) else None

fraud_model = load_ai_model("fraud_detection_model_final.pkl")
fraud_scaler = load_ai_model("fraud_detection_scaler_final.pkl")
chargeback_model = load_ai_model("chargeback_prediction_model.pkl")
chargeback_scaler = load_ai_model("chargeback_prediction_scaler.pkl")
subscription_model = load_ai_model("subscription_revenue_model.pkl")
subscription_scaler = load_ai_model("subscription_revenue_scaler.pkl")
smart_routing_model = load_model(os.path.join(MODEL_PATH, "smart_payment_routing_model.h5")) \
    if os.path.exists(os.path.join(MODEL_PATH, "smart_payment_routing_model.h5")) else None

with open(os.path.join(MODEL_PATH, "fraud_detection_metadata.json")) as f:
    fraud_features = json.load(f)["features_used"]

def sanitize_for_mongo(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_mongo(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_mongo(i) for i in obj]
    elif isinstance(obj, (np.integer, int)): return int(obj)
    elif isinstance(obj, (np.floating, float)): return float(obj)
    elif isinstance(obj, (np.bool_, bool)): return bool(obj)
    elif isinstance(obj, datetime): return obj
    elif isinstance(obj, str): return obj
    else: return str(obj)
    
def to_native(value):
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value

def classify_risk_level(confidence, high=0.85, medium=0.5):
    if confidence >= high: return "high"
    elif confidence >= medium: return "medium"
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

@app.get("/")
def root(): return {"message": "🤖 AI Payment Intelligence API"}

@app.get("/health")
def health(): return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

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


def process_fraud_workflow(transaction):
    try:
        print("🚀 Running fraud check for:", transaction["transaction_id"])

        model_input = TransactionRequest(**{
            "amount": transaction["amount"],
            "card_country": transaction["card_country"],
            "billing_country": transaction["billing_address_country"],
            "email": transaction["email"],
            "risk_score": transaction["risk_score"],
            "ip_address": transaction["ip_address"],
            "fingerprint": transaction["fingerprint"],
            "hour": datetime.utcnow().hour
        })

        prediction = run_fraud_prediction(model_input)
        confidence = float(prediction["confidence"])
        risk_level = classify_risk_level(confidence)

        result_doc = {
            "transaction_id": transaction["transaction_id"],
            "email": transaction["email"],
            "fraud_prediction": {
                "detected": bool(prediction["fraud_detected"]),
                "confidence_score": confidence,
                "risk_level": risk_level,
                "reasons": prediction["reasons"],
                "thresholds": {"high_risk": 0.85, "medium_risk": 0.5}
            },
            "features_used": sanitize_for_mongo(prediction["features_used"]),
            "model_info": {
                "name": "fraud_detection_model_final.pkl",
                "version": "v1.0",
                "run_time": datetime.utcnow().isoformat()
            },
            "created_at": datetime.utcnow()
        }

        db["fraud_results"].insert_one(result_doc)
        print("✅ Improved fraud result saved.")

    except Exception as e:
        print("❌ Fraud processing failed:", e)


def run_fraud_prediction(req: TransactionRequest):
    try:
        txn = db["transactions"].find_one(
            {"email": req.email, "ip_address": req.ip_address, "fingerprint": req.fingerprint},
            sort=[("created_at", -1)]
        )

        df = pd.DataFrame([txn]) if txn else pd.DataFrame([{**req.dict(), "created_at": datetime.utcnow()}])

        for col in ["refunded", "disputed", "transaction_id"]:
            if col not in df.columns:
                df[col] = False

        df.loc[0, ["amount", "risk_score", "hour", "card_country", "billing_address_country",
                   "email", "ip_address", "fingerprint"]] = [
            req.amount, req.risk_score, req.hour, req.card_country,
            req.billing_country, req.email, req.ip_address, req.fingerprint
        ]
        df["created_at"] = pd.to_datetime(datetime.utcnow())

        df["amount_log"] = np.log1p(df["amount"])
        df["country_mismatch"] = (df["card_country"] != df["billing_address_country"]).astype(int)
        df["email_domain_risk"] = df["email"].apply(
            lambda x: 1 if x.split("@")[1] in ["gmail.com", "yahoo.com", "hotmail.com"] else 0
        )

        df["ip_address_reuse_count"] = df.groupby("ip_address")["fingerprint"].transform("count")
        df["device_fingerprint_reuse_count"] = df.groupby("fingerprint")["transaction_id"].transform("count")
        df["device_ip_pair_reuse_count"] = df.groupby(["fingerprint", "ip_address"])["transaction_id"].transform("count")
        df["ip_address_country_mismatch"] = (df["card_country"] != df["ip_address"]).astype(int)

        df["email_transaction_count"] = df.groupby("email")["transaction_id"].transform("count")
        df["email_dispute_count"] = df.groupby("email")["disputed"].transform("sum")
        df["email_refund_count"] = df.groupby("email")["refunded"].transform("sum")
        df["chargeback_rate"] = df["email_dispute_count"] / df["email_transaction_count"]
        df["email_avg_amount"] = df.groupby("email")["amount"].transform("mean")

        df["unusual_amount_flag"] = (df["amount"] > df["amount"].quantile(0.99)).astype(int)
        df["time_between_transactions"] = df.groupby("email")["created_at"].diff().dt.total_seconds().fillna(999999)
        df["first_time_transaction"] = (df["email_transaction_count"] == 1).astype(int)
        df["customer_avg_amount_diff"] = abs(df["amount"] - df["email_avg_amount"])
        df["customer_last_transaction_diff"] = df.groupby("email")["amount"].diff().abs().fillna(0)
        df["customer_refund_ratio"] = df["email_refund_count"] / df["email_transaction_count"]
        df["shared_card_email_count"] = df.groupby("fingerprint")["email"].transform("nunique")
        df["shared_ip_email_count"] = df.groupby("ip_address")["email"].transform("nunique")
        df["previous_risk_scores_avg"] = df.groupby("email")["risk_score"].transform("mean")
        df["number_of_risky_transactions"] = df.groupby("email")["risk_score"].transform(lambda x: (x > 50).sum())

        for col in fraud_features:
            if col not in df.columns:
                df[col] = 0

        X = df[fraud_features].replace("unknown", 0).apply(pd.to_numeric, errors="coerce").fillna(0)
        X_scaled = fraud_scaler.transform(X)
        pred = fraud_model.predict(X_scaled)[0]
        proba = fraud_model.predict_proba(X_scaled)[0][1]

        reasons = []

        rules = [
            (req.amount > 5000, "Unusually high transaction amount"),
            (df.get("unusual_amount_flag", pd.Series([0])).iloc[0] == 1, "Amount falls in top 1% of all transactions"),
            (df.get("country_mismatch", pd.Series([0])).iloc[0] == 1, "Card and billing country mismatch"),
            (df.get("ip_address_country_mismatch", pd.Series([0])).iloc[0] == 1, "Card country and IP address mismatch"),
            (df.get("ip_address_reuse_count", pd.Series([0])).iloc[0] > 5, "IP address reused across multiple transactions"),
            (df.get("device_fingerprint_reuse_count", pd.Series([0])).iloc[0] > 3, "Device fingerprint reused for many transactions"),
            (df.get("device_ip_pair_reuse_count", pd.Series([0])).iloc[0] > 3, "Same device-IP pair used repeatedly"),
            (df.get("shared_card_email_count", pd.Series([0])).iloc[0] > 2, "Same card used by multiple email accounts"),
            (df.get("shared_ip_email_count", pd.Series([0])).iloc[0] > 2, "Same IP address linked to multiple email accounts"),
            (df.get("email_domain_risk", pd.Series([0])).iloc[0] == 1, "Email uses public domain (e.g. Gmail/Yahoo)"),
            (df.get("customer_refund_ratio", pd.Series([0.0])).iloc[0] > 0.3, "High refund ratio for this customer"),
            (df.get("chargeback_rate", pd.Series([0.0])).iloc[0] > 0.2, "Frequent chargebacks by this customer"),
            (df.get("time_between_transactions", pd.Series([999999])).iloc[0] < 30, "Rapid repeat transaction (< 30s)"),
            (df.get("first_time_transaction", pd.Series([0])).iloc[0] == 1, "First-time transaction for this email"),
            (df.get("customer_avg_amount_diff", pd.Series([0.0])).iloc[0] > 100, "Transaction amount differs significantly from customer's average"),
            (df.get("customer_last_transaction_diff", pd.Series([0.0])).iloc[0] > 100, "Large difference from previous transaction amount"),
            (df.get("previous_risk_scores_avg", pd.Series([0.0])).iloc[0] > 70, "Historically high risk scores"),
            (df.get("number_of_risky_transactions", pd.Series([0])).iloc[0] > 3, "Customer has multiple past high-risk transactions"),
            (df.get("risk_score", pd.Series([0])).iloc[0] > 70, "High Stripe risk score")
        ]

        for condition, reason in rules:
            if condition:
                reasons.append(reason)

        return {
            "fraud_detected": bool(pred),
            "confidence": round(proba, 4),
            "reasons": reasons,
            "features_used": {k: float(X.iloc[0][k]) for k in fraud_features}
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

def predict_and_store_chargeback(transaction):
    try:
        req = TransactionRequest(
            amount=transaction["amount"],
            card_country=transaction["card_country"],
            billing_country=transaction["billing_address_country"],
            email=transaction["email"],
            risk_score=transaction["risk_score"],
            ip_address=transaction["ip_address"],
            fingerprint=transaction["fingerprint"],
            hour=datetime.utcnow().hour
        )

        prediction = predict_chargeback(req)
        safe_prediction = {k: to_native(v) for k, v in prediction.items()}

        safe_prediction["transaction_id"] = transaction["transaction_id"]
        safe_prediction["email"] = transaction["email"]
        safe_prediction["created_at"] = datetime.utcnow()

        db["chargeback_predictions"].update_one(
            {"transaction_id": transaction["transaction_id"]},
            {"$set": safe_prediction},
            upsert=True
        )
        print("✅ Chargeback prediction saved.")
        
        update_transaction_with_chargeback(
            transaction["transaction_id"],
            prediction,
            transaction["email"]
        )

        print(f"🔍 Chargeback prediction stored for {transaction['transaction_id']}")

    except Exception as e:
        print("❌ Chargeback prediction error:", e)


@router.post("/predict/payment_gateway")
def predict_payment_gateway(req: TransactionRequest):
    if not smart_routing_model:
        raise HTTPException(500, "Smart routing model unavailable")
    try:
        X = np.array([[req.amount, req.risk_score, req.hour]])
        predictions = smart_routing_model.predict(X, verbose=0)[0]
        idx = int(np.argmax(predictions))
        return {
            "recommended_gateway": ["Stripe", "PayPal", "Adyen"][idx],
            "confidence_scores": {
                "Stripe": round(float(predictions[0]), 4),
                "PayPal": round(float(predictions[1]), 4),
                "Adyen": round(float(predictions[2]), 4)
            }
        }
    except Exception as e:
        raise HTTPException(500, str(e))
    
@router.post("/predict/subscription_revenue")
def predict_subscription_revenue(req: TransactionRequest):
    if not subscription_model or not subscription_scaler:
        raise HTTPException(500, "Subscription model unavailable")
    try:
        X = subscription_scaler.transform([[req.amount, req.risk_score, req.hour]])
        revenue = subscription_model.predict(X)[0]
        return {
            "expected_next_revenue": round(float(revenue), 2),
            "customer_signal": {
                "high_risk_score": req.risk_score > 60,
                "transaction_hour": req.hour
            },
            "note": "Predicted using historical subscription patterns"
        }
    except Exception as e:
        raise HTTPException(500, str(e))

    
    
@app.get("/jobs/predict-chargebacks")
def run_chargeback_predictions_job():
    count = 0
    cursor = db["transactions"].find({"chargeback_predicted": {"$exists": False}})

    for txn in cursor:
        try:
            req = TransactionRequest(
                amount=txn["amount"],
                card_country=txn["card_country"],
                billing_country=txn["billing_address_country"],
                email=txn["email"],
                risk_score=txn["risk_score"],
                ip_address=txn["ip_address"],
                fingerprint=txn["fingerprint"],
                hour=txn.get("created_at", datetime.utcnow()).hour
            )

            prediction = predict_chargeback(req)
            safe_prediction = {k: to_native(v) for k, v in prediction.items()}

            safe_prediction["transaction_id"] = txn["transaction_id"]
            safe_prediction["email"] = txn["email"]
            safe_prediction["created_at"] = datetime.utcnow()

            db["chargeback_predictions"].update_one(
                {"transaction_id": txn["transaction_id"]},
                {"$set": safe_prediction},
                upsert=True
            )
            print("✅ Chargeback prediction saved.")

            update_transaction_with_chargeback(
                txn["transaction_id"],
                prediction,
                txn["email"]
            )
            count += 1
        except Exception as e:
            print(f"❌ Error predicting for {txn.get('transaction_id')}: {str(e)}")

    return {"message": f"✅ Chargeback prediction run for {count} transactions"}



def predict_chargeback(req: TransactionRequest):
    if not chargeback_model or not chargeback_scaler:
        raise HTTPException(status_code=500, detail="Chargeback model or scaler not loaded")

    try:
        now = datetime.utcnow()

        recent_txn = db["transactions"].find({
            "email": req.email,
            "ip_address": req.ip_address,
            "fingerprint": req.fingerprint
        }).sort("created_at", -1).limit(10)

        rows = list(recent_txn)
        for r in rows:
            r["created_at"] = r.get("created_at", now)

        df = pd.DataFrame(rows if rows else [{
            "amount": req.amount,
            "risk_score": req.risk_score,
            "card_country": req.card_country,
            "billing_address_country": req.billing_country,
            "email": req.email,
            "ip_address": req.ip_address,
            "fingerprint": req.fingerprint,
            "refunded": False,
            "disputed": False,
            "transaction_id": "new_txn",
            "created_at": now
        }])

        current_txn = {
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
            "created_at": now
        }
        df = pd.concat([df, pd.DataFrame([current_txn])], ignore_index=True)

        # Preprocessing
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['amount_log'] = np.log1p(df['amount'])
        df['hour'] = df['created_at'].dt.hour
        df['is_weekend'] = (df['created_at'].dt.weekday >= 5).astype(int)
        df['time_between_transactions'] = df.groupby('email')['created_at'].diff().dt.total_seconds().fillna(999999)

        df['ip_address_reuse_count'] = df.groupby('ip_address')['transaction_id'].transform('count')
        df['fingerprint_reuse_count'] = df.groupby('fingerprint')['transaction_id'].transform('count')
        df['device_ip_pair_reuse_count'] = df.groupby(['fingerprint', 'ip_address'])['transaction_id'].transform('count')
        df['email_transaction_count'] = df.groupby('email')['transaction_id'].transform('count')
        df['customer_refund_ratio'] = df.groupby('email')['refunded'].transform('mean').fillna(0)
        df['average_transaction_amount'] = df.groupby('email')['amount'].transform('mean').fillna(0)
        df['transaction_amount_diff'] = abs(df['amount'] - df['average_transaction_amount'])
        df['country_mismatch'] = (df['card_country'] != df['billing_address_country']).astype(int)
        df['ip_country_mismatch'] = (df['card_country'] != df['ip_address']).astype(int)
        df['email_domain_risk'] = df['email'].apply(
            lambda x: 1 if str(x).split('@')[-1] in ['gmail.com', 'yahoo.com', 'hotmail.com'] else 0
        )
        df['past_chargebacks'] = df.groupby('email')['disputed'].transform('sum')

        latest = df.iloc[-1:]

        feature_columns = [
            'amount_log', 'hour', 'is_weekend', 'ip_address_reuse_count', 'fingerprint_reuse_count',
            'device_ip_pair_reuse_count', 'risk_score', 'email_transaction_count',
            'customer_refund_ratio', 'country_mismatch', 'ip_country_mismatch', 'time_between_transactions',
            'email_domain_risk', 'transaction_amount_diff', 'past_chargebacks'
        ]

        X = latest[feature_columns].replace([np.inf, -np.inf], 0).fillna(0)
        X_scaled = chargeback_scaler.transform(X)
        prediction = chargeback_model.predict(X_scaled)[0]
        confidence = chargeback_model.predict_proba(X_scaled)[0][1]

        reasons = []
        if latest['customer_refund_ratio'].values[0] > 0.5:
            reasons.append("High refund ratio")
        if latest['device_ip_pair_reuse_count'].values[0] > 3:
            reasons.append("Device/IP reused")
        if latest['country_mismatch'].values[0] == 1:
            reasons.append("Card and billing country mismatch")
        if latest['ip_country_mismatch'].values[0] == 1:
            reasons.append("Card country and IP mismatch")
        if latest['email_domain_risk'].values[0] == 1:
            reasons.append("Free email domain used")
        if latest['transaction_amount_diff'].values[0] > latest['amount_log'].values[0]:
            reasons.append("Unusual transaction amount")
        if latest['email_transaction_count'].values[0] > 10:
            reasons.append("Unusual number of transactions from email")
        if latest['fingerprint_reuse_count'].values[0] > 5:
            reasons.append("Device fingerprint reused frequently")
        if latest['ip_address_reuse_count'].values[0] > 5:
            reasons.append("IP address used for multiple transactions")
        if latest['risk_score'].values[0] > 70:
            reasons.append("High fraud risk score")
        if latest['time_between_transactions'].values[0] < 60:
            reasons.append("Rapid transaction frequency")
        if latest['past_chargebacks'].values[0] > 0:
            reasons.append("Past chargebacks found for user")
        if latest['amount_log'].values[0] > 8: 
            reasons.append("Very high transaction amount")

        chargeback_reason = ", ".join(reasons) if reasons else "No strong indicators"

        return {
            "chargeback_predicted": bool(prediction),
            "confidence_score": round(confidence, 4),
            "chargeback_reason": chargeback_reason
        }

    except Exception as e:
        print("❌ Chargeback prediction error:", e)
        raise HTTPException(status_code=500, detail=str(e))
    
def predict_and_store_subscription_revenue(transaction):
    try:
        req = TransactionRequest(
            amount=transaction["amount"],
            card_country=transaction["card_country"],
            billing_country=transaction["billing_address_country"],
            email=transaction["email"],
            risk_score=transaction["risk_score"],
            ip_address=transaction["ip_address"],
            fingerprint=transaction["fingerprint"],
            hour=datetime.utcnow().hour
        )

        prediction = predict_subscription_revenue(req)
        safe_prediction = {k: to_native(v) for k, v in prediction.items()}
        safe_prediction["transaction_id"] = transaction["transaction_id"]
        safe_prediction["email"] = transaction["email"]
        safe_prediction["created_at"] = datetime.utcnow()

        db["subscription_revenue_forecasts"].update_one(
            {"transaction_id": transaction["transaction_id"]},
            {"$set": safe_prediction},
            upsert=True
        )
        print(f"📈 Subscription revenue forecast saved for {transaction['transaction_id']}")

    except Exception as e:
        print("❌ Revenue forecasting failed:", e)
    
    
def update_transaction_with_chargeback(transaction_id, prediction_result, email):
    try:
        chargeback_predicted = to_native(prediction_result.get("chargeback_predicted", False))
        confidence_score = to_native(prediction_result.get("confidence_score", 0.0))

        db["transactions"].update_one(
            {"transaction_id": transaction_id},
            {
                "$set": {
                    "chargeback_predicted": chargeback_predicted,
                    "chargeback_confidence": confidence_score,
                    "email": email,
                    "updated_at": datetime.utcnow()
                }
            },
            upsert=True
        )
        print("✅ Chargeback prediction saved in `transactions`.")
    except Exception as e:
        print(f"❌ Failed to update `transactions`: {e}")

