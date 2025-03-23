import os
import hmac
import hashlib
import base64
import requests
import json
from datetime import datetime, timedelta
from pymongo import MongoClient
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

API_KEY = os.getenv("SOLIDGATE_API_KEY")
API_SECRET = os.getenv("SOLIDGATE_API_SECRET")
MONGO_URI = os.getenv("MONGO_URI")

mongo_client = MongoClient(MONGO_URI)
db = mongo_client['payment_intelligence']
customers_collection = db['customers']
subscriptions_collection = db['subscriptions']
transactions_collection = db['transactions']

SOLIDGATE_API_BASE = "https://reports.solidgate.com/api/v1"


def generate_signature(public_key, payload_json, secret_key):
    data = public_key + payload_json + public_key
    hmac_digest = hmac.new(secret_key.encode('utf-8'), data.encode('utf-8'), hashlib.sha512).digest()
    return base64.b64encode(hmac_digest).decode('utf-8')

def solidgate_post(endpoint, data):
    payload_json = json.dumps(data, separators=(',', ':'), sort_keys=True)
    signature = generate_signature(API_KEY, payload_json, API_SECRET)

    headers = {
        "Content-Type": "application/json",
        "merchant": API_KEY,
        "signature": signature
    }

    url = f"{SOLIDGATE_API_BASE}{endpoint}"

    response = requests.post(url, data=payload_json, headers=headers)

    try:
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError:
        print("❌ Solidgate Error:", response.status_code, response.text)
        return response.json()


def parse_datetime(dt_str):
    try:
        return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    except:
        return None


def fetch_and_store_solidgate_orders():
    print("\n🔄 Fetching Solidgate orders...")
    now = datetime.utcnow()
    from_date = (now - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
    to_date = now.strftime("%Y-%m-%d %H:%M:%S")

    data = {
        "filter": "updated_at",
        "date_from": from_date,
        "date_to": to_date,
        "limit": 1000
    }

    result = solidgate_post("/card-orders", data)
    orders = result.get('orders')

    for order in orders:
        transaction_id = order.get('transactions', [{}])[0].get('id')
        if not transaction_id:
            print("⚠️ Skipping transaction with missing ID")
            continue

        if transactions_collection.find_one({"transaction_id": transaction_id}):
            print(f"ℹ️ Transaction {transaction_id} already exists. Skipping.")
            continue

        card = order.get("transactions", [{}])[0].get("card", {})

        transaction = {
            "transaction_id": transaction_id,
            "email": order.get("customer_email", "unknown@example.com"),
            "amount": float(order.get("amount", 0)) / 100,
            "currency": order.get("processing_currency", order.get("currency", "usd")).lower(),
            "gateway": "Solidgate",
            "status": order.get("status"),
            "payment_method": order.get("payment_type", "unknown"),
            "card_brand": card.get("brand"),
            "card_country": card.get("country"),
            "fingerprint": card.get("card_id"),
            "funding_type": card.get("card_type", "").lower(),
            "three_d_secure": None,
            "cvc_check": None,
            "address_line1_check": None,
            "postal_code_check": None,
            "risk_level": None,
            "risk_score": None,
            "seller_message": None,
            "network_status": None,
            "outcome_type": None,
            "ip_address": order.get("ip_address", "unknown"),
            "billing_name": card.get("card_holder"),
            "billing_email": order.get("customer_email", None),
            "billing_phone": None,
            "billing_address_country": order.get("geo_country"),
            "billing_address_line1": None,
            "billing_address_line2": None,
            "billing_address_postal_code": None,
            "billing_address_city": None,
            "billing_address_state": None,
            "refunded": False,
            "amount_refunded": 0,
            "disputed": False,
            "captured": True,
            "paid": True,
            "created_at": parse_datetime(order.get("created_at"))
        }

        try:
            result = transactions_collection.insert_one(transaction)
            print(f"✅ Inserted transaction {transaction_id} with _id: {result.inserted_id}")
        except Exception as e:
            print(f"❌ Failed to insert transaction {transaction_id}: {e}")


def fetch_and_store_solidgate_subscriptions():
    print("\n🔄 Fetching Solidgate subscriptions...")
    now = datetime.utcnow()
    from_date = (now - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
    to_date = now.strftime("%Y-%m-%d %H:%M:%S")

    data = {
        "filter": "updated_at",
        "date_from": from_date,
        "date_to": to_date,
        "limit": 1000
    }

    result = solidgate_post("/subscriptions", data)
    for sub_id, sub in result.get('subscriptions', {}).items():
        if subscriptions_collection.find_one({"subscription_id": sub_id}):
            continue

        product = sub.get("product", {})
        customer = sub.get("customer", {})
        subscription = {
            "subscription_id": sub_id,
            "email": customer.get("customer_email", "unknown@example.com"),
            "gateway": "Solidgate",
            "status": sub.get("status"),
            "current_period_start": parse_datetime(sub.get("started_at")),
            "current_period_end": parse_datetime(sub.get("expired_at")),
            "plan_id": product.get("id"),
            "plan_name": product.get("name"),
            "product_id": product.get("id"),
            "price_amount": float(product.get("amount", 0)) / 100,
            "currency": product.get("currency", "usd").lower(),
            "interval": product.get("payment_action"),
            "created_at": parse_datetime(sub.get("started_at")),
            "cancel_at_period_end": False,
            "canceled_at": parse_datetime(sub.get("cancelled_at")),
            "ended_at": parse_datetime(sub.get("expired_at")),
            "trial_start": None,
            "trial_end": None,
            "quantity": 1,
            "metadata": {},
            "latest_invoice": None,
            "collection_method": "charge_automatically",
            "default_payment_method": None,
            "billing_cycle_anchor": parse_datetime(sub.get("started_at"))
        }

        subscriptions_collection.insert_one(subscription)
        print(f"✅ Stored subscription: {sub_id}")


def fetch_and_store_solidgate_customers():
    print("\n🔄 Indexing Solidgate customers...")
    emails = set()

    for tx in transactions_collection.find({"gateway": "Solidgate"}):
        emails.add(tx.get('email'))

    for sub in subscriptions_collection.find({"gateway": "Solidgate"}):
        emails.add(sub.get('email'))

    for email in emails:
        if customers_collection.find_one({"email": email}):
            continue

        tx = transactions_collection.find_one({"email": email, "gateway": "Solidgate"})

        customer = {
            "email": email,
            "name": None,
            "phone": None,
            "currency": tx.get("currency", "usd") if tx else "usd",
            "country": None,
            "address_line1": None,
            "address_line2": None,
            "city": None,
            "state": None,
            "postal_code": None,
            "created_at": datetime.utcnow(),
            "delinquent": False,
            "default_payment_method": None,
            "balance": 0,
            "tax_info": {
                "tax_id": None,
                "type": None
            },
            "metadata": {},
            "invoice_prefix": None,
            "gateway_customer_ids": {
                "solidgate": tx.get("fingerprint") if tx else "unknown"
            }
        }

        customers_collection.insert_one(customer)
        print(f"✅ Stored customer: {email}")


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=3) as executor:
        executor.submit(fetch_and_store_solidgate_orders)
        executor.submit(fetch_and_store_solidgate_subscriptions)
        executor.submit(fetch_and_store_solidgate_customers)

    print("\n✅ All Solidgate data fetched and stored successfully!")
