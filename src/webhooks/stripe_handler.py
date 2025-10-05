"""
Stripe webhook handlers
"""
import stripe
from datetime import datetime
from database.connection import db
from utils.helpers import sanitize_for_mongo
from config.settings import settings

# Initialize Stripe
stripe.api_key = settings.STRIPE_SECRET_KEY

async def handle_stripe_webhook(event_type: str, obj: dict):
    """Handle Stripe webhook events"""
    try:
        if event_type == "customer.created":
            await _handle_customer_created(obj)
        elif event_type == "customer.subscription.created":
            await _handle_subscription_created(obj)
        elif event_type == "customer.subscription.deleted":
            await _handle_subscription_deleted(obj)
        elif event_type == "refund.created":
            await _handle_refund_created(obj)
        elif event_type == "charge.dispute.closed":
            await _handle_dispute_closed(obj)
        elif event_type == "invoice.paid":
            await _handle_invoice_paid(obj)
        elif event_type == "invoice.payment_succeeded":
            await _handle_invoice_payment_succeeded(obj)
        elif event_type == "invoice.finalized":
            await _handle_invoice_finalized(obj)
        elif event_type == "invoice.updated":
            await _handle_invoice_updated(obj)
        else:
            print(f"Unhandled Stripe event type: {event_type}")
            
    except Exception as e:
        print(f"Error handling Stripe event {event_type}: {e}")

async def _handle_customer_created(obj):
    """Handle customer creation event"""
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
    print(f"Customer saved: {customer_data['email']}")

async def _handle_subscription_created(obj):
    """Handle subscription creation event"""
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
    print(f"Subscription created: {sub_data['subscription_id']}")

async def _handle_subscription_deleted(obj):
    """Handle subscription deletion event"""
    sub_id = obj.get("id")
    db["subscriptions"].update_one({"subscription_id": sub_id}, {"$set": {
        "status": "canceled",
        "canceled_at": datetime.utcnow(),
        "ended_at": datetime.utcnow()
    }})
    print(f"Subscription canceled: {sub_id}")

async def _handle_refund_created(obj):
    """Handle refund creation event"""
    charge_id = obj.get("charge")
    db["transactions"].update_one({"transaction_id": charge_id}, {"$set": {
        "refunded": True,
        "amount_refunded": obj.get("amount", 0) / 100.0,
        "refund_created_at": datetime.utcfromtimestamp(obj.get("created"))
    }})
    print(f"Refund created for transaction: {charge_id}")

async def _handle_dispute_closed(obj):
    """Handle dispute closed event"""
    charge_id = obj.get("charge")
    db["transactions"].update_one({"transaction_id": charge_id}, {"$set": {
        "disputed": True,
        "dispute_status": "closed",
        "dispute_closed_at": datetime.utcnow()
    }})
    print(f"Dispute closed for transaction: {charge_id}")

async def _handle_invoice_paid(obj):
    """Handle invoice paid event"""
    try:
        invoice = obj
        charge_id = invoice.get("charge")
        if charge_id:
            db["transactions"].update_one(
                {"transaction_id": charge_id},
                {"$set": {
                    "status": "paid",
                    "paid": True,
                    "updated_at": datetime.utcnow()
                }},
                upsert=True
            )
            print(f"Invoice paid for transaction: {charge_id}")
    except Exception as e:
        print(f"Invoice paid handling failed: {str(e)}")

async def _handle_invoice_payment_succeeded(obj):
    """Handle invoice payment succeeded event"""
    try:
        invoice = obj
        charge_id = invoice.get("charge")
        if charge_id:
            db["transactions"].update_one(
                {"transaction_id": charge_id},
                {"$set": {
                    "status": "paid",
                    "paid": True,
                    "updated_at": datetime.utcnow()
                }},
                upsert=True
            )
            print(f"Payment succeeded for transaction: {charge_id}")
    except Exception as e:
        print(f"Invoice payment succeeded handling failed: {str(e)}")

async def _handle_invoice_finalized(obj):
    """Handle invoice finalized event"""
    try:
        invoice = obj
        print(f"Invoice finalized: {invoice.get('id')}")
    except Exception as e:
        print(f"Invoice finalized handling failed: {str(e)}")

async def _handle_invoice_updated(obj):
    """Handle invoice updated event"""
    try:
        invoice = obj
        print(f"Invoice updated: {invoice.get('id')}")
    except Exception as e:
        print(f"Invoice updated handling failed: {str(e)}")
