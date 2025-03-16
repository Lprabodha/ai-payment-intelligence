import stripe
from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

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

        address = customer.get('address') or {}
        tax_info = customer.get('tax_info') or {}

        default_payment_method = customer.get('invoice_settings', {}).get('default_payment_method', None)

        customer_record = {
            "email": email,
            "name": customer.get('name', None),
            "phone": customer.get('phone', None),
            "currency": customer.get('currency', None),
            "country": address.get('country', None),
            "address_line1": address.get('line1', None),
            "address_line2": address.get('line2', None),
            "city": address.get('city', None),
            "state": address.get('state', None),
            "postal_code": address.get('postal_code', None),
            "created_at": datetime.utcfromtimestamp(customer['created']),
            "delinquent": customer.get('delinquent', False), 
            "default_payment_method": default_payment_method,
            "balance": customer.get('balance', 0), 
            "tax_info": {
                "tax_id": tax_info.get('tax_id', None),
                "type": tax_info.get('type', None)
            },
            "metadata": customer.get('metadata', {}), 
            "invoice_prefix": customer.get('invoice_prefix', None), 
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


from datetime import datetime

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
            items = sub['items']['data']
            first_item = items[0] if items else {}

            plan = first_item.get('plan', {})
            price = plan.get('amount', 0) / 100 if plan else 0
            interval = plan.get('interval', None)
            currency = plan.get('currency', None)
            product_id = plan.get('product', None)

            metadata = sub.get('metadata', {})

            sub_data = {
                "subscription_id": sub['id'],
                "email": email,
                "gateway": "Stripe",
                "status": sub['status'],
                "current_period_start": datetime.utcfromtimestamp(sub['current_period_start']),
                "current_period_end": datetime.utcfromtimestamp(sub['current_period_end']),
                "plan_id": plan.get('id', None),
                "plan_name": plan.get('nickname', None), 
                "product_id": product_id,
                "price_amount": price,
                "currency": currency,
                "interval": interval,  
                "created_at": datetime.utcfromtimestamp(sub['created']),
                "cancel_at_period_end": sub['cancel_at_period_end'],
                "canceled_at": datetime.utcfromtimestamp(sub['canceled_at']) if sub.get('canceled_at') else None,
                "ended_at": datetime.utcfromtimestamp(sub['ended_at']) if sub.get('ended_at') else None,
                "trial_start": datetime.utcfromtimestamp(sub['trial_start']) if sub.get('trial_start') else None,
                "trial_end": datetime.utcfromtimestamp(sub['trial_end']) if sub.get('trial_end') else None,
                "quantity": sub['quantity'],
                "metadata": metadata, 
                "latest_invoice": sub.get('latest_invoice', None),
                "collection_method": sub.get('collection_method', 'charge_automatically'),
                "default_payment_method": sub.get('default_payment_method', None),
                "billing_cycle_anchor": datetime.utcfromtimestamp(sub['billing_cycle_anchor']) if sub.get('billing_cycle_anchor') else None
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
            card_details = charge.get('payment_method_details', {}).get('card', {})

            outcome = charge.get('outcome', {})

            billing_details = charge.get('billing_details', {})

            transaction_data = {
                "transaction_id": charge['id'],
                "email": email,
                "amount": charge['amount'] / 100, 
                "currency": charge['currency'],
                "gateway": "Stripe",
                "status": charge['status'],
                "payment_method": charge['payment_method_details']['type'] if charge.get('payment_method_details') else 'unknown',
                "card_brand": card_details.get('brand', None),
                "card_country": card_details.get('country', None),
                "fingerprint": card_details.get('fingerprint', None),
                "funding_type": card_details.get('funding', None),
                "three_d_secure": card_details.get('three_d_secure', None),
                "cvc_check": card_details.get('checks', {}).get('cvc_check', None),
                "address_line1_check": card_details.get('checks', {}).get('address_line1_check', None),
                "postal_code_check": card_details.get('checks', {}).get('address_postal_code_check', None),
                "risk_level": outcome.get('risk_level', 'unknown'),
                "risk_score": outcome.get('risk_score', 0),
                "seller_message": outcome.get('seller_message', ''),
                "network_status": outcome.get('network_status', ''),
                "outcome_type": outcome.get('type', ''),
                "ip_address": card_details.get('country', 'unknown'),
                "billing_name": billing_details.get('name', None),
                "billing_email": billing_details.get('email', None),
                "billing_phone": billing_details.get('phone', None),
                "billing_address_country": billing_details.get('address', {}).get('country', None),
                "billing_address_line1": billing_details.get('address', {}).get('line1', None),
                "billing_address_line2": billing_details.get('address', {}).get('line2', None),
                "billing_address_postal_code": billing_details.get('address', {}).get('postal_code', None),
                "billing_address_city": billing_details.get('address', {}).get('city', None),
                "billing_address_state": billing_details.get('address', {}).get('state', None),
                "refunded": charge['refunded'],
                "amount_refunded": charge.get('amount_refunded', 0) / 100,
                "disputed": charge['disputed'],
                "captured": charge.get('captured', False),
                "paid": charge.get('paid', False),
                "created_at": datetime.utcfromtimestamp(charge['created'])
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
