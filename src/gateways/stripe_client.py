from pymongo import MongoClient
from datetime import datetime
import stripe
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup Stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
stripe.max_network_retries = 3

# Setup MongoDB
MONGO_URI = os.getenv("MONGO_URI")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client['payment_intelligence']
customers_collection = db['customers']
subscriptions_collection = db['subscriptions']
transactions_collection = db['transactions']

def safe_datetime(timestamp):
    try:
        return datetime.utcfromtimestamp(timestamp) if timestamp else None
    except Exception:
        return None

def fetch_and_store_customers_by_email():
    print("🚀 Fetching Stripe customers...")
    try:
        existing_emails = set(c['email'] for c in customers_collection.find({}, {"email": 1}))
        for customer in stripe.Customer.list(limit=100).auto_paging_iter():
            email = customer.get('email') or 'unknown@example.com'
            if email in existing_emails:
                continue

            address = customer.get('address') or {}
            tax_info = customer.get('tax_info') or {}
            customer_record = {
                "email": email,
                "name": customer.get('name'),
                "phone": customer.get('phone'),
                "currency": customer.get('currency'),
                "country": address.get('country'),
                "address_line1": address.get('line1'),
                "address_line2": address.get('line2'),
                "city": address.get('city'),
                "state": address.get('state'),
                "postal_code": address.get('postal_code'),
                "created_at": safe_datetime(customer.get('created')),
                "delinquent": customer.get('delinquent', False),
                "default_payment_method": customer.get('invoice_settings', {}).get('default_payment_method'),
                "balance": customer.get('balance', 0),
                "tax_info": {
                    "tax_id": tax_info.get('tax_id'),
                    "type": tax_info.get('type')
                },
                "metadata": customer.get('metadata', {}),
                "invoice_prefix": customer.get('invoice_prefix'),
                "gateway_customer_ids": {'stripe': customer['id']}
            }

            customers_collection.insert_one(customer_record)
            print(f"✅ Stored customer: {email}")
    except Exception as e:
        print(f"❌ Error fetching customers: {e}")
        
def fetch_and_store_subscriptions_globally():
    print("🚀 Fetching all Stripe subscriptions (global)...")
    try:
        # Build a mapping of Stripe customer ID -> email from your DB
        customer_map = {
            c['gateway_customer_ids']['stripe']: c['email']
            for c in customers_collection.find(
                {"gateway_customer_ids.stripe": {"$exists": True}},
                {"gateway_customer_ids.stripe": 1, "email": 1}
            )
        }

        # Cache existing subscription IDs
        existing_sub_ids = set(
            s['subscription_id'] for s in subscriptions_collection.find({}, {"subscription_id": 1})
        )

        new_subscriptions = []
        for sub in stripe.Subscription.list(limit=100).auto_paging_iter():
            sub_id = sub.get('id')
            if sub_id in existing_sub_ids:
                continue

            customer_id = sub.get('customer')
            email = customer_map.get(customer_id, "unknown@example.com")

            items = sub.get('items', {}).get('data', [])
            plan = items[0].get('plan', {}) if items else {}

            new_subscriptions.append({
                "subscription_id": sub_id,
                "email": email,
                "gateway": "Stripe",
                "status": sub.get('status'),
                "current_period_start": safe_datetime(sub.get('current_period_start')),
                "current_period_end": safe_datetime(sub.get('current_period_end')),
                "plan_id": plan.get('id'),
                "plan_name": plan.get('nickname'),
                "product_id": plan.get('product'),
                "price_amount": plan.get('amount', 0) / 100,
                "currency": plan.get('currency'),
                "interval": plan.get('interval'),
                "created_at": safe_datetime(sub.get('created')),
                "cancel_at_period_end": sub.get('cancel_at_period_end', False),
                "canceled_at": safe_datetime(sub.get('canceled_at')),
                "ended_at": safe_datetime(sub.get('ended_at')),
                "trial_start": safe_datetime(sub.get('trial_start')),
                "trial_end": safe_datetime(sub.get('trial_end')),
                "quantity": sub.get('quantity'),
                "metadata": sub.get('metadata', {}),
                "latest_invoice": sub.get('latest_invoice'),
                "collection_method": sub.get('collection_method', 'charge_automatically'),
                "default_payment_method": sub.get('default_payment_method'),
                "billing_cycle_anchor": safe_datetime(sub.get('billing_cycle_anchor'))
            })

            print(f"✅ Prepared subscription: {sub_id} ({email})")

        if new_subscriptions:
            subscriptions_collection.insert_many(new_subscriptions)
            print(f"✅ Inserted {len(new_subscriptions)} new subscriptions into MongoDB.")
        else:
            print("✅ No new subscriptions to insert.")

    except Exception as e:
        print(f"❌ Error fetching global subscriptions: {e}")

def fetch_and_store_subscriptions_by_email():
    print("🚀 Fetching Stripe subscriptions...")
    try:
        existing_sub_ids = set(s['subscription_id'] for s in subscriptions_collection.find({}, {"subscription_id": 1}))
        for customer in customers_collection.find():
            email = customer.get('email')
            stripe_customer_id = customer.get('gateway_customer_ids', {}).get('stripe')
            if not stripe_customer_id:
                continue

            for attempt in range(3):
                try:
                    subscriptions = list(stripe.Subscription.list(customer=stripe_customer_id, limit=100).auto_paging_iter())
                    break
                except stripe.error.RateLimitError:
                    print(f"⚠️ Rate limit for {email}, retrying in {2 ** attempt}s...")
                    time.sleep(2 ** attempt)
                except Exception as err:
                    print(f"❌ Error fetching subscriptions for {email}: {err}")
                    subscriptions = []
                    break

            for sub in subscriptions[:7]:
                sub_id = sub.get('id')
                if sub_id in existing_sub_ids:
                    continue

                items = sub.get('items', {}).get('data', [])
                plan = items[0].get('plan', {}) if items else {}

                subscriptions_collection.insert_one({
                    "subscription_id": sub_id,
                    "email": email,
                    "gateway": "Stripe",
                    "status": sub.get('status'),
                    "current_period_start": safe_datetime(sub.get('current_period_start')),
                    "current_period_end": safe_datetime(sub.get('current_period_end')),
                    "plan_id": plan.get('id'),
                    "plan_name": plan.get('nickname'),
                    "product_id": plan.get('product'),
                    "price_amount": plan.get('amount', 0) / 100,
                    "currency": plan.get('currency'),
                    "interval": plan.get('interval'),
                    "created_at": safe_datetime(sub.get('created')),
                    "cancel_at_period_end": sub.get('cancel_at_period_end', False),
                    "canceled_at": safe_datetime(sub.get('canceled_at')),
                    "ended_at": safe_datetime(sub.get('ended_at')),
                    "trial_start": safe_datetime(sub.get('trial_start')),
                    "trial_end": safe_datetime(sub.get('trial_end')),
                    "quantity": sub.get('quantity'),
                    "metadata": sub.get('metadata', {}),
                    "latest_invoice": sub.get('latest_invoice'),
                    "collection_method": sub.get('collection_method', 'charge_automatically'),
                    "default_payment_method": sub.get('default_payment_method'),
                    "billing_cycle_anchor": safe_datetime(sub.get('billing_cycle_anchor'))
                })
                print(f"✅ Stored subscription: {sub_id}")
            time.sleep(0.5)
    except Exception as e:
        print(f"❌ General subscription error: {e}")

def fetch_and_store_transactions_by_email():
    print("🚀 Fetching Stripe transactions...")
    try:
        existing_tx_ids = set(t['transaction_id'] for t in transactions_collection.find({}, {"transaction_id": 1}))
        for customer in customers_collection.find():
            email = customer.get('email')
            stripe_customer_id = customer.get('gateway_customer_ids', {}).get('stripe')
            if not stripe_customer_id:
                continue

            charges = list(stripe.Charge.list(customer=stripe_customer_id, limit=100).auto_paging_iter())
            for charge in charges[:7]:
                tx_id = charge.get('id')
                if tx_id in existing_tx_ids:
                    continue

                card = charge.get('payment_method_details', {}).get('card', {})
                outcome = charge.get('outcome', {})
                billing = charge.get('billing_details', {})

                transactions_collection.insert_one({
                    "transaction_id": tx_id,
                    "email": email,
                    "amount": charge.get('amount', 0) / 100,
                    "currency": charge.get('currency'),
                    "gateway": "Stripe",
                    "status": charge.get('status'),
                    "payment_method": charge.get('payment_method_details', {}).get('type'),
                    "card_brand": card.get('brand'),
                    "card_country": card.get('country'),
                    "fingerprint": card.get('fingerprint'),
                    "funding_type": card.get('funding'),
                    "three_d_secure": card.get('three_d_secure'),
                    "cvc_check": card.get('checks', {}).get('cvc_check'),
                    "address_line1_check": card.get('checks', {}).get('address_line1_check'),
                    "postal_code_check": card.get('checks', {}).get('address_postal_code_check'),
                    "risk_level": outcome.get('risk_level', 'unknown'),
                    "risk_score": outcome.get('risk_score', 0),
                    "seller_message": outcome.get('seller_message'),
                    "network_status": outcome.get('network_status'),
                    "outcome_type": outcome.get('type'),
                    "ip_address": card.get('country'),
                    "billing_name": billing.get('name'),
                    "billing_email": billing.get('email'),
                    "billing_phone": billing.get('phone'),
                    "billing_address_country": billing.get('address', {}).get('country'),
                    "billing_address_line1": billing.get('address', {}).get('line1'),
                    "billing_address_line2": billing.get('address', {}).get('line2'),
                    "billing_address_postal_code": billing.get('address', {}).get('postal_code'),
                    "billing_address_city": billing.get('address', {}).get('city'),
                    "billing_address_state": billing.get('address', {}).get('state'),
                    "refunded": charge.get('refunded', False),
                    "amount_refunded": charge.get('amount_refunded', 0) / 100,
                    "disputed": charge.get('disputed', False),
                    "captured": charge.get('captured', False),
                    "paid": charge.get('paid', False),
                    "created_at": safe_datetime(charge.get('created'))
                })
                print(f"✅ Stored transaction: {tx_id}")
    except Exception as e:
        print(f"❌ Error fetching transactions: {e}")
        
        
def fetch_and_store_transactions_globally():
    print("🚀 Fetching all Stripe transactions globally...")
    try:
        customer_map = {
            c['gateway_customer_ids']['stripe']: c['email']
            for c in customers_collection.find(
                {"gateway_customer_ids.stripe": {"$exists": True}},
                {"gateway_customer_ids.stripe": 1, "email": 1}
            )
        }

        # Get all existing transaction IDs
        existing_tx_ids = set(
            tx['transaction_id'] for tx in transactions_collection.find({}, {"transaction_id": 1})
        )

        new_transactions = []

        # Fetch charges globally
        for charge in stripe.Charge.list(limit=100).auto_paging_iter():
            tx_id = charge.get('id')
            if tx_id in existing_tx_ids:
                continue

            stripe_customer_id = charge.get('customer')
            email = customer_map.get(stripe_customer_id, "unknown@example.com")

            card = charge.get('payment_method_details', {}).get('card', {})
            outcome = charge.get('outcome', {})
            billing = charge.get('billing_details', {})

            new_transactions.append({
                "transaction_id": tx_id,
                "email": email,
                "amount": charge.get('amount', 0) / 100,
                "currency": charge.get('currency'),
                "gateway": "Stripe",
                "status": charge.get('status'),
                "payment_method": charge.get('payment_method_details', {}).get('type'),
                "card_brand": card.get('brand'),
                "card_country": card.get('country'),
                "fingerprint": card.get('fingerprint'),
                "funding_type": card.get('funding'),
                "three_d_secure": card.get('three_d_secure'),
                "cvc_check": card.get('checks', {}).get('cvc_check'),
                "address_line1_check": card.get('checks', {}).get('address_line1_check'),
                "postal_code_check": card.get('checks', {}).get('address_postal_code_check'),
                "risk_level": outcome.get('risk_level', 'unknown'),
                "risk_score": outcome.get('risk_score', 0),
                "seller_message": outcome.get('seller_message'),
                "network_status": outcome.get('network_status'),
                "outcome_type": outcome.get('type'),
                "ip_address": card.get('country'),
                "billing_name": billing.get('name'),
                "billing_email": billing.get('email'),
                "billing_phone": billing.get('phone'),
                "billing_address_country": billing.get('address', {}).get('country'),
                "billing_address_line1": billing.get('address', {}).get('line1'),
                "billing_address_line2": billing.get('address', {}).get('line2'),
                "billing_address_postal_code": billing.get('address', {}).get('postal_code'),
                "billing_address_city": billing.get('address', {}).get('city'),
                "billing_address_state": billing.get('address', {}).get('state'),
                "refunded": charge.get('refunded', False),
                "amount_refunded": charge.get('amount_refunded', 0) / 100,
                "disputed": charge.get('disputed', False),
                "captured": charge.get('captured', False),
                "paid": charge.get('paid', False),
                "created_at": safe_datetime(charge.get('created'))
            })

            print(f"✅ Prepared transaction: {tx_id} ({email})")

        if new_transactions:
            transactions_collection.insert_many(new_transactions)
            print(f"✅ Inserted {len(new_transactions)} new transactions.")
        else:
            print("✅ No new transactions to insert.")

    except Exception as e:
        print(f"❌ Error fetching transactions: {e}")


# Run the process
if __name__ == '__main__':
    fetch_and_store_customers_by_email()
    fetch_and_store_subscriptions_globally()
    fetch_and_store_transactions_globally()
    print("✅ All Stripe data fetched and stored successfully!")