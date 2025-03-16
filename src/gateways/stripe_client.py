
import stripe
from pymongo import MongoClient
from datetime import datetime
import os

stripe.api_key = os.getenv("STRIPE_SECRET_KEY") 

MONGO_URI = os.getenv("MONGO_URI")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client['payment_intelligence']
customers_collection = db['customers']
subscriptions_collection = db['subscriptions']
transactions_collection = db['transactions']


def fetch_and_store_customers_by_email():
    print("Fetching Stripe customers...")
    customers = stripe.Customer.list(limit=100)
    
    for customer in customers.auto_paging_iter():
        email = customer['email'] or 'unknown@example.com' 
        existing_customer = customers_collection.find_one({"email": email})

        customer_record = {
            "email": email,
            "name": customer.get('name', None),
            "phone": customer.get('phone', None),
            "country": customer.get('address', {}).get('country', None),
            "created_at": datetime.utcfromtimestamp(customer['created']),
        }

        if existing_customer:
            existing_ids = existing_customer.get('gateway_customer_ids', {})
            existing_ids['stripe'] = customer['id']
            customer_record['gateway_customer_ids'] = existing_ids
        else:
            customer_record['gateway_customer_ids'] = {'stripe': customer['id']}
        
        customers_collection.update_one(
            {"email": email},
            {"$set": customer_record},
            upsert=True
        )
        print(f"✅ Stored customer {email}")

def fetch_and_store_subscriptions_by_email():
    print("Fetching Stripe subscriptions...")
    customers = customers_collection.find()

    for customer in customers:
        email = customer['email']
        stripe_customer_id = customer.get('gateway_customer_ids', {}).get('stripe', None)
        if not stripe_customer_id:
            continue

        subscriptions = stripe.Subscription.list(customer=stripe_customer_id, limit=100)

        for sub in subscriptions.auto_paging_iter():
            sub_data = {
                "subscription_id": sub['id'],
                "email": email,
                "gateway": "Stripe",
                "status": sub['status'],
                "current_period_start": datetime.utcfromtimestamp(sub['current_period_start']),
                "current_period_end": datetime.utcfromtimestamp(sub['current_period_end']),
                "plan_id": sub['plan']['id'] if sub['plan'] else None,
                "created_at": datetime.utcfromtimestamp(sub['created']),
                "cancel_at_period_end": sub['cancel_at_period_end'],
                "quantity": sub['quantity'],
            }
            subscriptions_collection.update_one(
                {"subscription_id": sub['id']},
                {"$set": sub_data},
                upsert=True
            )
            print(f"✅ Stored subscription {sub['id']} for {email}")

def fetch_and_store_transactions_by_email():
    print("Fetching Stripe transactions (charges)...")
    customers = customers_collection.find()

    for customer in customers:
        email = customer['email']
        stripe_customer_id = customer.get('gateway_customer_ids', {}).get('stripe', None)
        if not stripe_customer_id:
            continue

        charges = stripe.Charge.list(customer=stripe_customer_id, limit=100)

        for charge in charges.auto_paging_iter():
            transaction_data = {
                "transaction_id": charge['id'],
                "email": email,
                "amount": charge['amount'] / 100,
                "currency": charge['currency'],
                "gateway": "Stripe",
                "status": charge['status'],
                "payment_method": charge['payment_method_details']['type'] if charge.get('payment_method_details') else 'unknown',
                "card_brand": charge['payment_method_details']['card']['brand'] if charge.get('payment_method_details', {}).get('card') else None,
                "risk_level": charge.get('outcome', {}).get('risk_level', 'unknown'),
                "risk_score": charge.get('outcome', {}).get('risk_score', 0),
                "ip_address": charge['payment_method_details'].get('card', {}).get('country', 'unknown') if charge.get('payment_method_details', {}).get('card') else None,
                "created_at": datetime.utcfromtimestamp(charge['created']),
                "refunded": charge['refunded'],
                "disputed": charge['disputed']
            }
            transactions_collection.update_one(
                {"transaction_id": charge['id']},
                {"$set": transaction_data},
                upsert=True
            )
            print(f"✅ Stored transaction {charge['id']} for {email}")

if __name__ == '__main__':
    fetch_and_store_customers_by_email()
    fetch_and_store_subscriptions_by_email()
    fetch_and_store_transactions_by_email()
    print("✅ All Stripe data fetched and stored by email successfully!")
