import os
import hmac
import hashlib
import base64
import requests
from datetime import datetime
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

SOLIDGATE_API_BASE = "https://api.solidgate.com"


def generate_signature(payload):
    message = base64.b64encode(payload.encode())
    signature = hmac.new(API_SECRET.encode(), message, hashlib.sha512).hexdigest()
    return signature


def solidgate_post(endpoint, data={}):
    url = f"{SOLIDGATE_API_BASE}{endpoint}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": API_KEY,
        "X-Signature": generate_signature(str(data))
    }
    response = requests.post(url, json=data, headers=headers)
    response.raise_for_status()
    return response.json()


def parse_datetime(dt_str):
    try:
        return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    except:
        return None


def fetch_and_store_solidgate_orders():
    print("Fetching Solidgate orders...")
    try:
        result = solidgate_post("/reports/card-orders", {})
        for order in result.get('orders', []):
            transaction_id = order.get('order_id')
            if not transaction_id or transactions_collection.find_one({"transaction_id": transaction_id}):
                continue

            transaction = {
                "transaction_id": transaction_id,
                "email": order.get("customer_email", "unknown@example.com"),
                "amount": float(order.get("amount", 0)) / 100,
                "currency": order.get("customer_currency", order.get("currency", "usd")).lower(),
                "gateway": "Solidgate",
                "status": order.get("status"),
                "payment_method": order.get("payment_type", "unknown"),
                "card_brand": None,
                "card_country": None,
                "fingerprint": None,
                "funding_type": None,
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
                "billing_name": None,
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

            if order.get("transactions"):
                card = order["transactions"][0].get("card", {})
                transaction.update({
                    "card_brand": card.get("brand"),
                    "card_country": card.get("country"),
                    "funding_type": card.get("card_type", "").lower(),
                    "billing_name": card.get("card_holder"),
                    "fingerprint": card.get("card_id", None)
                })

            transactions_collection.insert_one(transaction)
            print(f"✅ Stored transaction: {transaction_id}")
    except Exception as e:
        print(f"❌ Error fetching orders: {e}")


def fetch_and_store_solidgate_subscriptions():
    print("Fetching Solidgate subscriptions...")
    try:
        result = solidgate_post("/reports/subscriptions", {})
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
    except Exception as e:
        print(f"❌ Error fetching subscriptions: {e}")


def fetch_and_store_solidgate_customers():
    print("Fetching Solidgate customers from emails...")
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

    print("✅ All Solidgate data fetched and stored successfully!")
