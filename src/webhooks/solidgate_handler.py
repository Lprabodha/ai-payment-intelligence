"""
Solidgate webhook handlers with AI predictions
"""
import numpy as np
from datetime import datetime, timedelta
from database.connection import db
from utils.helpers import sanitize_for_mongo
from predictions.models import model_manager

async def handle_solidgate_webhook(event_type: str, webhook_data: dict, event_id: str = None):
    """Handle Solidgate webhook events with AI predictions"""
    try:
        print(f"Solidgate webhook received: {event_type} (ID: {event_id})")
        
        # Check for duplicate processing using event_id
        if event_id:
            existing_event = db["processed_webhook_events"].find_one({"event_id": event_id})
            if existing_event:
                print(f"Event {event_id} already processed, skipping")
                return
        
        # Route to appropriate handler
        if event_type == "card_gate.order.updated":
            await _handle_solidgate_order_updated(webhook_data, event_id)
        elif event_type == "subscription.updated":
            await _handle_solidgate_subscription_updated(webhook_data, event_id)
        elif event_type == "create":  # New subscription created
            await _handle_solidgate_subscription_created(webhook_data, event_id)
        elif event_type == "active":  # Payment successful, subscription activated
            await _handle_solidgate_subscription_activated(webhook_data, event_id)
        elif event_type == "renew":  # Subscription renewed
            await _handle_solidgate_subscription_renewed(webhook_data, event_id)
        else:
            print(f"Unhandled Solidgate event type: {event_type}")
            
    except Exception as e:
        print(f"Error handling Solidgate webhook {event_type}: {e}")

async def _handle_solidgate_order_updated(webhook_data: dict, event_id: str = None):
    """Handle card_gate.order.updated event from Solidgate"""
    try:
        order_data = webhook_data.get('order', {})
        transaction_id = order_data.get('order_id') or order_data.get('id')
        
        if not transaction_id:
            print("No order ID found in Solidgate order.updated webhook")
            return
        
        # Extract order details
        status = order_data.get('status', '').lower()
        card_data = order_data.get('card', {})
        
        # Determine transaction status based on order status
        if status in ['approved', 'succeeded', 'success']:
            txn_status = "succeeded"
            paid = True
            captured = True
        elif status in ['declined', 'failed', 'failure']:
            txn_status = "failed"
            paid = False
            captured = False
        else:
            txn_status = status
            paid = False
            captured = False
        
        # Check if transaction already exists
        existing_txn = db["transactions"].find_one({"transaction_id": transaction_id})
        if existing_txn:
            print(f"Transaction {transaction_id} already exists, updating status to {txn_status}")
            db["transactions"].update_one(
                {"transaction_id": transaction_id},
                {"$set": {
                    "status": txn_status,
                    "paid": paid,
                    "captured": captured,
                    "updated_at": datetime.utcnow()
                }}
            )
        else:
            # Create new transaction record
            transaction = {
                "transaction_id": transaction_id,
                "email": order_data.get('customer_email', 'unknown@example.com'),
                "amount": float(order_data.get('amount', 0)) / 100,
                "currency": order_data.get('currency', 'usd').lower(),
                "gateway": "Solidgate",
                "status": txn_status,
                "payment_method": order_data.get('payment_method', 'card'),
                "card_brand": card_data.get('brand'),
                "card_country": card_data.get('country'),
                "fingerprint": card_data.get('card_id'),
                "funding_type": card_data.get('card_type', '').lower(),
                "three_d_secure": None,
                "cvc_check": None,
                "address_line1_check": None,
                "postal_code_check": None,
                "risk_level": None,
                "risk_score": order_data.get('risk_score'),
                "seller_message": order_data.get('seller_message'),
                "network_status": order_data.get('network_status'),
                "outcome_type": order_data.get('outcome_type'),
                "ip_address": order_data.get('ip_address', 'unknown'),
                "billing_name": card_data.get('card_holder'),
                "billing_email": order_data.get('customer_email'),
                "billing_phone": None,
                "billing_address_country": order_data.get('geo_country'),
                "billing_address_line1": None,
                "billing_address_line2": None,
                "billing_address_postal_code": None,
                "billing_address_city": None,
                "billing_address_state": None,
                "refunded": False,
                "amount_refunded": 0,
                "disputed": False,
                "captured": captured,
                "paid": paid,
                "created_at": datetime.fromisoformat(order_data.get('created_at').replace('Z', '+00:00')) if order_data.get('created_at') else datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            db["transactions"].insert_one(sanitize_for_mongo(transaction))
            print(f"New Solidgate transaction saved: {transaction_id}")
        
        # Get smart routing prediction for successful payments
        if txn_status == "succeeded":
            routing_prediction = _get_smart_routing_prediction({
                'amount': float(order_data.get('amount', 0)) / 100,
                'card_country': card_data.get('country', 'US'),
                'card_brand': card_data.get('brand', 'VISA'),
                'risk_score': order_data.get('risk_score', 0)
            })
            
            # Store routing prediction
            routing_result = {
                "transaction_id": transaction_id,
                "recommended_gateway": routing_prediction.get('recommended_gateway'),
                "confidence": routing_prediction.get('confidence'),
                "all_scores": routing_prediction.get('all_scores', {}),
                "current_gateway": "Solidgate",
                "prediction_time": datetime.utcnow(),
                "error": routing_prediction.get('error')
            }
            
            db["routing_predictions"].update_one(
                {"transaction_id": transaction_id},
                {"$set": routing_result},
                upsert=True
            )
            
            print(f"Routing prediction for {transaction_id}: {routing_prediction.get('recommended_gateway')} (confidence: {routing_prediction.get('confidence'):.3f})")
        
        # Mark event as processed
        if event_id:
            db["processed_webhook_events"].insert_one({
                "event_id": event_id,
                "event_type": "card_gate.order.updated",
                "processed_at": datetime.utcnow(),
                "transaction_id": transaction_id
            })
        
    except Exception as e:
        print(f"Error handling Solidgate order.updated: {e}")

async def _handle_solidgate_subscription_updated(webhook_data: dict, event_id: str = None):
    """Handle subscription.updated event from Solidgate"""
    try:
        subscription_data = webhook_data.get('subscription', {})
        subscription_id = subscription_data.get('subscription_id') or subscription_data.get('id')
        
        if not subscription_id:
            print("No subscription ID found in Solidgate subscription.updated webhook")
            return
        
        # Extract subscription details
        status = subscription_data.get('status', 'active')
        product_data = subscription_data.get('product', {})
        customer_data = subscription_data.get('customer', {})
        
        # Update subscription record
        subscription_update = {
            "subscription_id": subscription_id,
            "email": customer_data.get('customer_email', 'unknown@example.com'),
            "gateway": "Solidgate",
            "status": status,
            "current_period_start": datetime.fromisoformat(subscription_data.get('started_at').replace('Z', '+00:00')) if subscription_data.get('started_at') else None,
            "current_period_end": datetime.fromisoformat(subscription_data.get('expired_at').replace('Z', '+00:00')) if subscription_data.get('expired_at') else None,
            "plan_id": product_data.get('id'),
            "plan_name": product_data.get('name'),
            "product_id": product_data.get('id'),
            "price_amount": float(product_data.get('amount', 0)) / 100,
            "currency": product_data.get('currency', 'usd').lower(),
            "interval": product_data.get('payment_action', 'month'),
            "created_at": datetime.fromisoformat(subscription_data.get('started_at').replace('Z', '+00:00')) if subscription_data.get('started_at') else datetime.utcnow(),
            "cancel_at_period_end": status == 'canceled',
            "canceled_at": datetime.utcnow() if status == 'canceled' else None,
            "ended_at": datetime.fromisoformat(subscription_data.get('expired_at').replace('Z', '+00:00')) if status == 'canceled' and subscription_data.get('expired_at') else None,
            "trial_start": None,
            "trial_end": None,
            "quantity": 1,
            "metadata": {},
            "latest_invoice": None,
            "collection_method": "charge_automatically",
            "default_payment_method": None,
            "billing_cycle_anchor": datetime.fromisoformat(subscription_data.get('started_at').replace('Z', '+00:00')) if subscription_data.get('started_at') else datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        db["subscriptions"].update_one(
            {"subscription_id": subscription_id},
            {"$set": sanitize_for_mongo(subscription_update)},
            upsert=True
        )
        
        print(f"Solidgate subscription updated: {subscription_id} - Status: {status}")
        
        # Get revenue prediction for active subscriptions
        if status == 'active':
            revenue_prediction = _get_subscription_revenue_prediction({
                'price_amount': subscription_update['price_amount'],
                'account_age_days': (datetime.utcnow() - subscription_update['created_at']).days if subscription_update['created_at'] else 30,
                'renewal_count': 1,  # Default for now
                'subscription_duration_days': 30
            })
            
            # Store revenue prediction
            revenue_result = {
                "subscription_id": subscription_id,
                "predicted_revenue": revenue_prediction.get('predicted_revenue'),
                "current_revenue": revenue_prediction.get('current_revenue'),
                "growth_rate": revenue_prediction.get('growth_rate'),
                "prediction_time": datetime.utcnow(),
                "error": revenue_prediction.get('error')
            }
            
            db["revenue_predictions"].update_one(
                {"subscription_id": subscription_id},
                {"$set": revenue_result},
                upsert=True
            )
        
        # Mark event as processed
        if event_id:
            db["processed_webhook_events"].insert_one({
                "event_id": event_id,
                "event_type": "subscription.updated",
                "processed_at": datetime.utcnow(),
                "subscription_id": subscription_id
            })
        
    except Exception as e:
        print(f"Error handling Solidgate subscription.updated: {e}")

async def _handle_solidgate_subscription_created(webhook_data: dict, event_id: str = None):
    """Handle subscription creation (callback_type: create) from Solidgate"""
    try:
        subscription_data = webhook_data.get('subscription', {})
        subscription_id = subscription_data.get('subscription_id') or subscription_data.get('id')
        
        if not subscription_id:
            print("No subscription ID found in Solidgate subscription created webhook")
            return
        
        # Create subscription record for new subscription
        subscription = {
            "subscription_id": subscription_id,
            "email": subscription_data.get('customer_email', 'unknown@example.com'),
            "gateway": "Solidgate",
            "status": "pending",  # New subscriptions start as pending
            "current_period_start": datetime.fromisoformat(subscription_data.get('created_at').replace('Z', '+00:00')) if subscription_data.get('created_at') else datetime.utcnow(),
            "current_period_end": None,  # Will be set when activated
            "plan_id": subscription_data.get('product_id'),
            "plan_name": subscription_data.get('product'),
            "product_id": subscription_data.get('product_id'),
            "price_amount": float(subscription_data.get('amount', 0)) / 100,
            "currency": subscription_data.get('currency', 'usd').lower(),
            "interval": "month",  # Default
            "created_at": datetime.fromisoformat(subscription_data.get('created_at').replace('Z', '+00:00')) if subscription_data.get('created_at') else datetime.utcnow(),
            "cancel_at_period_end": False,
            "canceled_at": None,
            "ended_at": None,
            "trial_start": None,
            "trial_end": None,
            "quantity": 1,
            "metadata": {},
            "latest_invoice": None,
            "collection_method": "charge_automatically",
            "default_payment_method": None,
            "billing_cycle_anchor": datetime.fromisoformat(subscription_data.get('created_at').replace('Z', '+00:00')) if subscription_data.get('created_at') else datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        db["subscriptions"].update_one(
            {"subscription_id": subscription_id},
            {"$set": sanitize_for_mongo(subscription)},
            upsert=True
        )
        
        print(f"Solidgate subscription created: {subscription_id}")
        
        # Mark event as processed
        if event_id:
            db["processed_webhook_events"].insert_one({
                "event_id": event_id,
                "event_type": "create",
                "processed_at": datetime.utcnow(),
                "subscription_id": subscription_id
            })
        
    except Exception as e:
        print(f"Error handling Solidgate subscription creation: {e}")

async def _handle_solidgate_subscription_activated(webhook_data: dict, event_id: str = None):
    """Handle subscription activation (callback_type: active) from Solidgate"""
    try:
        subscription_data = webhook_data.get('subscription', {})
        subscription_id = subscription_data.get('subscription_id') or subscription_data.get('id')
        
        if not subscription_id:
            print("No subscription ID found in Solidgate subscription activated webhook")
            return
        
        # Update subscription to active status
        db["subscriptions"].update_one(
            {"subscription_id": subscription_id},
            {"$set": {
                "status": "active",
                "current_period_start": datetime.fromisoformat(subscription_data.get('activated_at').replace('Z', '+00:00')) if subscription_data.get('activated_at') else datetime.utcnow(),
                "current_period_end": datetime.fromisoformat(subscription_data.get('expired_at').replace('Z', '+00:00')) if subscription_data.get('expired_at') else datetime.utcnow() + timedelta(days=30),
                "updated_at": datetime.utcnow()
            }}
        )
        
        # Get updated subscription for revenue prediction
        updated_subscription = db["subscriptions"].find_one({"subscription_id": subscription_id})
        if updated_subscription:
            revenue_prediction = _get_subscription_revenue_prediction({
                'price_amount': updated_subscription['price_amount'],
                'account_age_days': 0,  # Just activated
                'renewal_count': 1,
                'subscription_duration_days': 30
            })
            
            # Store revenue prediction
            revenue_result = {
                "subscription_id": subscription_id,
                "predicted_revenue": revenue_prediction.get('predicted_revenue'),
                "current_revenue": revenue_prediction.get('current_revenue'),
                "growth_rate": revenue_prediction.get('growth_rate'),
                "prediction_time": datetime.utcnow(),
                "error": revenue_prediction.get('error')
            }
            
            db["revenue_predictions"].update_one(
                {"subscription_id": subscription_id},
                {"$set": revenue_result},
                upsert=True
            )
            
            print(f"Revenue prediction for activated subscription {subscription_id}: ${revenue_prediction.get('predicted_revenue'):.2f}")
        
        print(f"Solidgate subscription activated: {subscription_id}")
        
        # Mark event as processed
        if event_id:
            db["processed_webhook_events"].insert_one({
                "event_id": event_id,
                "event_type": "active",
                "processed_at": datetime.utcnow(),
                "subscription_id": subscription_id
            })
        
    except Exception as e:
        print(f"Error handling Solidgate subscription activation: {e}")

async def _handle_solidgate_subscription_renewed(webhook_data: dict, event_id: str = None):
    """Handle subscription renewal (callback_type: renew) from Solidgate"""
    try:
        subscription_data = webhook_data.get('subscription', {})
        subscription_id = subscription_data.get('subscription_id') or subscription_data.get('id')
        
        if not subscription_id:
            print("No subscription ID found in Solidgate subscription renewed webhook")
            return
        
        # Update subscription with new period
        db["subscriptions"].update_one(
            {"subscription_id": subscription_id},
            {"$set": {
                "status": "active",
                "current_period_start": datetime.fromisoformat(subscription_data.get('renewed_at').replace('Z', '+00:00')) if subscription_data.get('renewed_at') else datetime.utcnow(),
                "current_period_end": datetime.fromisoformat(subscription_data.get('expired_at').replace('Z', '+00:00')) if subscription_data.get('expired_at') else datetime.utcnow() + timedelta(days=30),
                "updated_at": datetime.utcnow()
            }}
        )
        
        # Get updated subscription for revenue prediction
        updated_subscription = db["subscriptions"].find_one({"subscription_id": subscription_id})
        if updated_subscription:
            # Calculate account age and renewal count
            account_age_days = (datetime.utcnow() - updated_subscription['created_at']).days
            renewal_count = db["subscriptions"].count_documents({"email": updated_subscription['email'], "gateway": "Solidgate"})
            
            revenue_prediction = _get_subscription_revenue_prediction({
                'price_amount': updated_subscription['price_amount'],
                'account_age_days': account_age_days,
                'renewal_count': renewal_count,
                'subscription_duration_days': 30
            })
            
            # Store updated revenue prediction
            revenue_result = {
                "subscription_id": subscription_id,
                "predicted_revenue": revenue_prediction.get('predicted_revenue'),
                "current_revenue": revenue_prediction.get('current_revenue'),
                "growth_rate": revenue_prediction.get('growth_rate'),
                "prediction_time": datetime.utcnow(),
                "error": revenue_prediction.get('error')
            }
            
            db["revenue_predictions"].update_one(
                {"subscription_id": subscription_id},
                {"$set": revenue_result},
                upsert=True
            )
            
            print(f"Revenue prediction for renewed subscription {subscription_id}: ${revenue_prediction.get('predicted_revenue'):.2f}")
        
        print(f"Solidgate subscription renewed: {subscription_id}")
        
        # Mark event as processed
        if event_id:
            db["processed_webhook_events"].insert_one({
                "event_id": event_id,
                "event_type": "renew",
                "processed_at": datetime.utcnow(),
                "subscription_id": subscription_id
            })
        
    except Exception as e:
        print(f"Error handling Solidgate subscription renewal: {e}")

def _get_smart_routing_prediction(transaction_data):
    """Get smart routing prediction for transaction"""
    try:
        # Load smart routing model
        routing_model = model_manager.get_model('routing_model')
        if not routing_model:
            return {"recommended_gateway": "Solidgate", "confidence": 1.0, "error": "Model not found"}
        
        # Extract features
        amount = float(transaction_data.get('amount', 0))
        country = transaction_data.get('card_country', 'US')
        card_type = transaction_data.get('card_brand', 'VISA')
        risk_score = float(transaction_data.get('risk_score', 0))
        hour = datetime.utcnow().hour
        
        # Feature encoding (matching training)
        amount_log = np.log1p(amount)
        amount_sqrt = np.sqrt(amount)
        amount_category = 0 if amount < 100 else (1 if amount < 500 else 2)
        
        country_risk = {
            'US': 0.1, 'CA': 0.1, 'GB': 0.1, 'AU': 0.1, 'DE': 0.1,
            'FR': 0.15, 'IT': 0.2, 'ES': 0.2, 'BR': 0.3, 'MX': 0.3,
            'IN': 0.4, 'CN': 0.5, 'RU': 0.6, 'NG': 0.7
        }.get(country, 0.5)
        
        card_success_rate = {
            'VISA': 0.95, 'MASTERCARD': 0.94, 'AMEX': 0.92, 
            'DISCOVER': 0.88, 'JCB': 0.85, 'DINERS': 0.82
        }.get(card_type, 0.85)
        
        # Time features
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        is_weekend = 1 if datetime.utcnow().weekday() >= 5 else 0
        is_business_hours = 1 if 9 <= hour <= 17 else 0
        is_evening = 1 if hour >= 18 else 0
        is_night = 1 if hour < 6 else 0
        
        # Build state vector
        state = [
            amount_log, amount_sqrt, amount_category,
            country_risk, card_success_rate, risk_score / 100.0,
            hour_sin, hour_cos, is_weekend,
            is_business_hours, is_evening, is_night
        ]
        
        # Get prediction
        predictions = routing_model.predict(np.array(state).reshape(1, -1), verbose=0)[0]
        gateway_map = {0: "Stripe", 1: "PayPal", 2: "Adyen"}
        
        # Calculate confidence
        softmax_scores = np.exp(predictions) / np.sum(np.exp(predictions))
        recommended = gateway_map[np.argmax(predictions)]
        confidence = float(softmax_scores[np.argmax(predictions)])
        
        return {
            "recommended_gateway": recommended,
            "confidence": confidence,
            "all_scores": {
                gateway_map[i]: float(softmax_scores[i]) 
                for i in range(len(gateway_map))
            }
        }
        
    except Exception as e:
        print(f"Error in smart routing prediction: {e}")
        return {"recommended_gateway": "Solidgate", "confidence": 1.0, "error": str(e)}

def _get_subscription_revenue_prediction(subscription_data):
    """Get subscription revenue prediction"""
    try:
        # Load subscription models
        ensemble_model = model_manager.get_model('subscription_ensemble')
        scaler = model_manager.get_model('subscription_scaler')
        
        if not ensemble_model or not scaler:
            return {"predicted_revenue": subscription_data.get('price_amount', 0), "error": "Models not found"}
        
        # Extract features (simplified version)
        features = {
            'account_age_days': subscription_data.get('account_age_days', 30),
            'renewal_count': subscription_data.get('renewal_count', 1),
            'average_subscription_value': subscription_data.get('price_amount', 100),
            'high_value_customer': 1 if subscription_data.get('price_amount', 0) > 100 else 0,
            'subscription_duration_days': subscription_data.get('subscription_duration_days', 30),
            'is_weekend': 1 if datetime.utcnow().weekday() >= 5 else 0
        }
        
        # Convert to array and scale
        feature_array = np.array(list(features.values())).reshape(1, -1)
        scaled_features = scaler.transform(feature_array)
        
        # Get prediction
        predicted_revenue = ensemble_model.predict(scaled_features)[0]
        
        return {
            "predicted_revenue": float(predicted_revenue),
            "current_revenue": subscription_data.get('price_amount', 0),
            "growth_rate": (predicted_revenue - subscription_data.get('price_amount', 0)) / max(subscription_data.get('price_amount', 1), 1)
        }
        
    except Exception as e:
        print(f"Error in subscription revenue prediction: {e}")
        return {"predicted_revenue": subscription_data.get('price_amount', 0), "error": str(e)}
