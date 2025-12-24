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
        elif event_type in ["subscription.updated", "subscription.updated.v2"]:
            await _handle_solidgate_subscription_updated(webhook_data, event_id)
        elif event_type == "create":  # New subscription created
            await _handle_solidgate_subscription_created(webhook_data, event_id)
        elif event_type == "active":  # Payment successful, subscription activated
            await _handle_solidgate_subscription_activated(webhook_data, event_id)
        elif event_type == "renew":  # Subscription renewed
            await _handle_solidgate_subscription_renewed(webhook_data, event_id)
        elif event_type == "init":  # Subscription initialization
            await _handle_solidgate_subscription_created(webhook_data, event_id)
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
        
        # Build comprehensive transaction object
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
            "three_d_secure": order_data.get('three_d_secure'),
            "cvc_check": order_data.get('cvc_check'),
            "address_line1_check": order_data.get('address_line1_check'),
            "postal_code_check": order_data.get('postal_code_check'),
            "risk_score": order_data.get('risk_score', 0),
            "seller_message": order_data.get('seller_message'),
            "network_status": order_data.get('network_status'),
            "outcome_type": order_data.get('outcome_type'),
            "ip_address": order_data.get('ip_address', 'unknown'),
            "billing_name": card_data.get('card_holder'),
            "billing_email": order_data.get('customer_email'),
            "billing_phone": order_data.get('customer_phone'),
            "billing_country": order_data.get('geo_country') or card_data.get('country'),
            "billing_address_line1": order_data.get('billing_address_line1'),
            "billing_address_line2": order_data.get('billing_address_line2'),
            "billing_address_postal_code": order_data.get('billing_address_postal_code'),
            "billing_address_city": order_data.get('billing_address_city'),
            "billing_address_state": order_data.get('billing_address_state'),
            "refunded": order_data.get('refunded', False),
            "amount_refunded": float(order_data.get('amount_refunded', 0)) / 100 if order_data.get('amount_refunded') else 0,
            "disputed": order_data.get('disputed', False),
            "captured": captured,
            "paid": paid,
            "created_at": datetime.fromisoformat(order_data.get('created_at').replace('Z', '+00:00')) if order_data.get('created_at') else datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        # Save or update transaction in database
        existing_txn = db["transactions"].find_one({"transaction_id": transaction_id})
        if existing_txn:
            print(f"Transaction {transaction_id} already exists, updating with latest data")
            db["transactions"].update_one(
                {"transaction_id": transaction_id},
                {"$set": sanitize_for_mongo(transaction)}
            )
        else:
            db["transactions"].insert_one(sanitize_for_mongo(transaction))
            print(f"New Solidgate transaction saved: {transaction_id}")
        
        if transaction_id and txn_status == "succeeded":
            db_transaction = db["transactions"].find_one({"transaction_id": transaction_id}) or transaction
            
            # Ensure all required fields have valid values (no None for required string fields)
            fingerprint_value = db_transaction.get('fingerprint') or card_data.get('card_id') or ""
            funding_type_value = db_transaction.get('funding_type') or card_data.get('card_type', '').lower() or "credit"
            
            transaction_data = {
                "transaction_id": transaction_id,
                "email": db_transaction.get('email', order_data.get('customer_email', 'unknown@example.com')),
                "amount": db_transaction.get('amount', float(order_data.get('amount', 0)) / 100),
                "currency": db_transaction.get('currency', order_data.get('currency', 'usd').lower()),
                "card_brand": db_transaction.get('card_brand', card_data.get('brand', 'VISA')),
                "card_country": db_transaction.get('card_country', card_data.get('country', 'US')),
                "billing_country": db_transaction.get('billing_country') or db_transaction.get('billing_address_country') or card_data.get('country', 'US'),
                "ip_address": db_transaction.get('ip_address', order_data.get('ip_address', 'unknown')),
                "fingerprint": fingerprint_value,
                "funding_type": funding_type_value,
                "risk_score": db_transaction.get('risk_score', order_data.get('risk_score', 0)),
                "three_d_secure": db_transaction.get('three_d_secure'),
                "cvc_check": db_transaction.get('cvc_check'),
                "address_line1_check": db_transaction.get('address_line1_check'),
                "postal_code_check": db_transaction.get('postal_code_check'),
                "outcome_type": db_transaction.get('outcome_type', order_data.get('outcome_type')),
                "seller_message": db_transaction.get('seller_message', order_data.get('seller_message')),
                "network_status": db_transaction.get('network_status', order_data.get('network_status')),
                "status": txn_status,
                "gateway": "Solidgate"
            }
            
            # Run fraud detection using process_fraud_workflow (same as Stripe)
            try:
                from predictions.fraud import process_fraud_workflow
                
                # Use the full transaction object from database (same format as Stripe expects)
                fraud_result = process_fraud_workflow(db_transaction)
                if fraud_result:
                    print(f"✅ Fraud detection for {transaction_id}: {fraud_result.get('is_fraud', False)} (confidence: {fraud_result.get('confidence_score', 0):.3f})")
                else:
                    print(f"⚠️ Fraud detection returned no result for {transaction_id}")
            except Exception as fraud_error:
                print(f"❌ Fraud detection error for {transaction_id}: {fraud_error}")
                import traceback
                traceback.print_exc()
            
            # Run chargeback prediction using predict_chargeback (same as Stripe)
            try:
                from predictions.chargeback import predict_chargeback
                from models.schemas import TransactionRequest
                import math
                
                # Create TransactionRequest from transaction (same as Stripe)
                chargeback_req = TransactionRequest(
                    amount=db_transaction.get("amount", 0.0),
                    currency=db_transaction.get("currency", "usd"),
                    email=db_transaction.get("email", ""),
                    ip_address=db_transaction.get("ip_address", ""),
                    card_country=db_transaction.get("card_country", ""),
                    billing_country=db_transaction.get("billing_country") or db_transaction.get("billing_address_country") or "",
                    card_brand=db_transaction.get("card_brand", ""),
                    funding_type=db_transaction.get("funding_type", "") or "",
                    fingerprint=db_transaction.get("fingerprint", "") or "",
                    risk_score=db_transaction.get("risk_score", 0),
                    three_d_secure=db_transaction.get("three_d_secure"),
                    cvc_check=db_transaction.get("cvc_check"),
                    address_line1_check=db_transaction.get("address_line1_check"),
                    postal_code_check=db_transaction.get("postal_code_check"),
                    outcome_type=db_transaction.get("outcome_type"),
                    seller_message=db_transaction.get("seller_message"),
                    network_status=db_transaction.get("network_status")
                )
                
                chargeback_result = predict_chargeback(chargeback_req)
                
                # Ensure confidence_score is valid (handle NaN, None, etc.)
                if chargeback_result:
                    confidence = chargeback_result.get("confidence_score", 0.0)
                    if confidence is None or (isinstance(confidence, float) and (math.isnan(confidence) or math.isinf(confidence))):
                        confidence = 0.0
                    else:
                        confidence = float(confidence)
                    
                    chargeback_result["confidence_score"] = confidence
                    
                    # Store chargeback prediction (same as Stripe)
                    db["chargeback_predictions"].update_one(
                        {"transaction_id": transaction_id},
                        {"$set": {
                            "transaction_id": transaction_id,
                            "email": db_transaction.get("email"),
                            "chargeback_predicted": chargeback_result.get("chargeback_predicted", False),
                            "confidence_score": confidence,
                            "chargeback_reason": chargeback_result.get("chargeback_reason", ""),
                            "model_type": chargeback_result.get("model_type", "default"),
                            "created_at": datetime.utcnow()
                        }},
                        upsert=True
                    )
                    print(f"✅ Chargeback prediction for {transaction_id}: {chargeback_result.get('chargeback_predicted', False)} (confidence: {confidence:.3f})")
                else:
                    print(f"⚠️ Chargeback prediction returned no result for {transaction_id}")
            except Exception as cb_error:
                print(f"❌ Chargeback prediction error for {transaction_id}: {cb_error}")
                import traceback
                traceback.print_exc()
            
            # Generate recommendations after predictions
            try:
                from utils.recommendation_engine import recommendation_engine
                import math
                
                # Get full transaction from database for recommendations (ensure we have all fields)
                db_transaction_for_rec = db["transactions"].find_one({"transaction_id": transaction_id}) or transaction
                
                # Get fraud and chargeback results from database
                fraud_doc = db["fraud_results"].find_one({"transaction_id": transaction_id}) or {}
                chargeback_doc = db["chargeback_predictions"].find_one({"transaction_id": transaction_id}) or {}
                
                # Prepare fraud prediction dict for recommendation engine
                fraud_pred_dict = None
                if fraud_doc:
                    fraud_conf = fraud_doc.get("confidence_score") or fraud_doc.get("confidence", 0.0)
                    if fraud_conf is None or (isinstance(fraud_conf, float) and (math.isnan(fraud_conf) or math.isinf(fraud_conf))):
                        fraud_conf = 0.0
                    fraud_pred_dict = {
                        "is_fraud": fraud_doc.get("is_fraud", fraud_doc.get("fraud_predicted", False)),
                        "confidence_score": float(fraud_conf),
                        "fraud_reasons": fraud_doc.get("fraud_reasons", [])
                    }
                
                # Prepare chargeback prediction dict for recommendation engine
                chargeback_pred_dict = None
                if chargeback_doc:
                    cb_confidence = chargeback_doc.get("confidence_score") or chargeback_doc.get("confidence", 0.0)
                    if cb_confidence is None or (isinstance(cb_confidence, float) and (math.isnan(cb_confidence) or math.isinf(cb_confidence))):
                        cb_confidence = 0.0
                    chargeback_pred_dict = {
                        "chargeback_predicted": chargeback_doc.get("chargeback_predicted", False),
                        "confidence_score": float(cb_confidence),
                        "chargeback_reason": chargeback_doc.get("chargeback_reason", "")
                    }
                
                # Generate comprehensive recommendations
                recommendations = recommendation_engine.build_comprehensive_recommendations(
                    transaction=db_transaction_for_rec,
                    fraud_pred=fraud_pred_dict,
                    chargeback_pred=chargeback_pred_dict
                )
                
                # Extract risk_level from recommendations
                risk_level = recommendations.get("risk_level", "unknown")
                overall_priority = recommendations.get("overall_priority", "low")
                combined_risk_score = recommendations.get("combined_risk_score", 0.0)
                
                # Save recommendations to separate recommendations collection
                recommendations_doc = sanitize_for_mongo(recommendations)
                recommendations_doc["_id"] = f"rec_{transaction_id}"
                recommendations_doc["transaction_id"] = transaction_id
                
                db["recommendations"].replace_one(
                    {"transaction_id": transaction_id},
                    recommendations_doc,
                    upsert=True
                )
                

                db["transactions"].update_one(
                    {"transaction_id": transaction_id},
                    {"$set": {
                        "risk_level": risk_level,
                        "overall_priority": overall_priority,
                        "combined_risk_score": combined_risk_score,
                        "recommendations_id": f"rec_{transaction_id}",
                        "recommendations": sanitize_for_mongo(recommendations),  # Keep for backward compatibility
                        "updated_at": datetime.utcnow()
                    }}
                )
                
                print(f"✅ Recommendations generated and saved for {transaction_id} (risk_level: {risk_level}, priority: {overall_priority})")
                
            except Exception as rec_error:
                print(f"⚠️ Recommendation generation error for {transaction_id}: {rec_error}")
                import traceback
                traceback.print_exc()
            
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
        subscription_id = subscription_data.get('id') or subscription_data.get('subscription_id')
        
        if not subscription_id:
            print("No subscription ID found in Solidgate subscription.updated webhook")
            return
        
        # Extract subscription details
        status = subscription_data.get('status', 'active')
        product_data = webhook_data.get('product', {})
        customer_data = webhook_data.get('customer', {})
        
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
    """Handle subscription creation (callback_type: create or init) from Solidgate"""
    try:
        subscription_data = webhook_data.get('subscription', {})
        subscription_id = subscription_data.get('id') or subscription_data.get('subscription_id')
        
        if not subscription_id:
            print("No subscription ID found in Solidgate subscription created webhook")
            return
        
        # Get customer and product data from webhook
        customer_data = webhook_data.get('customer', {})
        product_data = webhook_data.get('product', {})
        
        # Save customer data
        if customer_data:
            customer_email = customer_data.get('customer_email') or customer_data.get('email')
            if customer_email:
                customer_record = {
                    "email": customer_email,
                    "name": customer_data.get('customer_name') or customer_data.get('name'),
                    "phone": customer_data.get('customer_phone') or customer_data.get('phone'),
                    "gateway_customer_ids": {"solidgate": customer_data.get('customer_id') or customer_data.get('id')},
                    "created_at": datetime.fromisoformat(customer_data.get('created_at').replace('Z', '+00:00')) if customer_data.get('created_at') else datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
                db["customers"].update_one(
                    {"email": customer_email},
                    {"$set": sanitize_for_mongo(customer_record)},
                    upsert=True
                )
                print(f"✅ Customer saved: {customer_email}")
        
        # Create subscription record for new subscription
        subscription = {
            "subscription_id": subscription_id,
            "email": customer_data.get('customer_email', 'unknown@example.com'),
            "gateway": "Solidgate",
            "status": "pending",  # New subscriptions start as pending
            "current_period_start": datetime.fromisoformat(subscription_data.get('started_at').replace('Z', '+00:00')) if subscription_data.get('started_at') else datetime.utcnow(),
            "current_period_end": datetime.fromisoformat(subscription_data.get('expired_at').replace('Z', '+00:00')) if subscription_data.get('expired_at') else None,
            "plan_id": product_data.get('product_id'),
            "plan_name": product_data.get('name'),
            "product_id": product_data.get('product_id'),
            "price_amount": float(product_data.get('amount', 0)) / 100,
            "currency": product_data.get('currency', 'usd').lower(),
            "interval": "month",  # Default
            "created_at": datetime.fromisoformat(subscription_data.get('started_at').replace('Z', '+00:00')) if subscription_data.get('started_at') else datetime.utcnow(),
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
            "billing_cycle_anchor": datetime.fromisoformat(subscription_data.get('started_at').replace('Z', '+00:00')) if subscription_data.get('started_at') else datetime.utcnow(),
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
        subscription_id = subscription_data.get('id') or subscription_data.get('subscription_id')
        
        if not subscription_id:
            print("No subscription ID found in Solidgate subscription activated webhook")
            return
        
        # Extract customer and order data
        customer_data = webhook_data.get('customer', {})
        order_data = webhook_data.get('order', {})  # May contain payment info
        product_data = webhook_data.get('product', {})
        
        # Save customer data
        if customer_data:
            customer_email = customer_data.get('customer_email') or customer_data.get('email')
            if customer_email:
                customer_record = {
                    "email": customer_email,
                    "name": customer_data.get('customer_name') or customer_data.get('name'),
                    "phone": customer_data.get('customer_phone') or customer_data.get('phone'),
                    "gateway_customer_ids": {"solidgate": customer_data.get('customer_id') or customer_data.get('id')},
                    "created_at": datetime.fromisoformat(customer_data.get('created_at').replace('Z', '+00:00')) if customer_data.get('created_at') else datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
                db["customers"].update_one(
                    {"email": customer_email},
                    {"$set": sanitize_for_mongo(customer_record)},
                    upsert=True
                )
                print(f"✅ Customer saved: {customer_email}")
        
        # Update subscription to active status
        db["subscriptions"].update_one(
            {"subscription_id": subscription_id},
            {"$set": {
                "status": "active",
                "current_period_start": datetime.fromisoformat(subscription_data.get('started_at').replace('Z', '+00:00')) if subscription_data.get('started_at') else datetime.utcnow(),
                "current_period_end": datetime.fromisoformat(subscription_data.get('expired_at').replace('Z', '+00:00')) if subscription_data.get('expired_at') else datetime.utcnow() + timedelta(days=30),
                "updated_at": datetime.utcnow()
            }}
        )
        
        # Create transaction record if order data is available (initial payment)
        transaction_id = order_data.get('order_id') or order_data.get('id') or f"sub_{subscription_id}_{int(datetime.utcnow().timestamp())}"
        card_data = order_data.get('card', {})
        
        if order_data or subscription_data:
            # Build transaction from order data or subscription data
            transaction = {
                "transaction_id": transaction_id,
                "subscription_id": subscription_id,
                "email": customer_data.get('customer_email') or customer_data.get('email', 'unknown@example.com'),
                "amount": float(order_data.get('amount', product_data.get('amount', 0))) / 100,
                "currency": order_data.get('currency', product_data.get('currency', 'usd')).lower(),
                "gateway": "Solidgate",
                "status": "succeeded",  # Subscription activated means payment succeeded
                "payment_method": order_data.get('payment_method', 'card'),
                "card_brand": card_data.get('brand'),
                "card_country": card_data.get('country'),
                "fingerprint": card_data.get('card_id') or "",
                "funding_type": card_data.get('card_type', '').lower() or "credit",
                "three_d_secure": order_data.get('three_d_secure'),
                "cvc_check": order_data.get('cvc_check'),
                "address_line1_check": order_data.get('address_line1_check'),
                "postal_code_check": order_data.get('postal_code_check'),
                "risk_score": order_data.get('risk_score', 0),
                "seller_message": order_data.get('seller_message'),
                "network_status": order_data.get('network_status'),
                "outcome_type": order_data.get('outcome_type'),
                "ip_address": order_data.get('ip_address', 'unknown'),
                "billing_name": card_data.get('card_holder'),
                "billing_email": customer_data.get('customer_email') or customer_data.get('email'),
                "billing_phone": customer_data.get('customer_phone') or customer_data.get('phone'),
                "billing_country": order_data.get('geo_country') or card_data.get('country'),
                "billing_address_line1": order_data.get('billing_address_line1'),
                "billing_address_line2": order_data.get('billing_address_line2'),
                "billing_address_postal_code": order_data.get('billing_address_postal_code'),
                "billing_address_city": order_data.get('billing_address_city'),
                "billing_address_state": order_data.get('billing_address_state'),
                "refunded": False,
                "amount_refunded": 0,
                "disputed": False,
                "captured": True,
                "paid": True,
                "created_at": datetime.fromisoformat(order_data.get('created_at').replace('Z', '+00:00')) if order_data.get('created_at') else datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            # Save transaction
            db["transactions"].update_one(
                {"transaction_id": transaction_id},
                {"$set": sanitize_for_mongo(transaction)},
                upsert=True
            )
            print(f"✅ Transaction saved for activated subscription: {transaction_id}")
            
            # Run fraud and chargeback predictions using same methods as Stripe
            try:
                # Get full transaction from database (same format as Stripe expects)
                db_transaction = db["transactions"].find_one({"transaction_id": transaction_id}) or transaction
                
                # Run fraud detection using process_fraud_workflow (same as Stripe)
                try:
                    from predictions.fraud import process_fraud_workflow
                    
                    fraud_result = process_fraud_workflow(db_transaction)
                    if fraud_result:
                        print(f"✅ Fraud detection for {transaction_id}: {fraud_result.get('is_fraud', False)} (confidence: {fraud_result.get('confidence_score', 0):.3f})")
                    else:
                        print(f"⚠️ Fraud detection returned no result for {transaction_id}")
                except Exception as fraud_error:
                    print(f"❌ Fraud detection error for {transaction_id}: {fraud_error}")
                    import traceback
                    traceback.print_exc()
                
                # Run chargeback prediction using predict_chargeback (same as Stripe)
                try:
                    from predictions.chargeback import predict_chargeback
                    from models.schemas import TransactionRequest
                    import math
                    
                    # Create TransactionRequest from transaction (same as Stripe)
                    chargeback_req = TransactionRequest(
                        amount=db_transaction.get("amount", 0.0),
                        currency=db_transaction.get("currency", "usd"),
                        email=db_transaction.get("email", ""),
                        ip_address=db_transaction.get("ip_address", ""),
                        card_country=db_transaction.get("card_country", ""),
                        billing_country=db_transaction.get("billing_country") or db_transaction.get("billing_address_country") or "",
                        card_brand=db_transaction.get("card_brand", ""),
                        funding_type=db_transaction.get("funding_type", "") or "",
                        fingerprint=db_transaction.get("fingerprint", "") or "",
                        risk_score=db_transaction.get("risk_score", 0),
                        three_d_secure=db_transaction.get("three_d_secure"),
                        cvc_check=db_transaction.get("cvc_check"),
                        address_line1_check=db_transaction.get("address_line1_check"),
                        postal_code_check=db_transaction.get("postal_code_check"),
                        outcome_type=db_transaction.get("outcome_type"),
                        seller_message=db_transaction.get("seller_message"),
                        network_status=db_transaction.get("network_status")
                    )
                    
                    chargeback_result = predict_chargeback(chargeback_req)
                    
                    # Ensure confidence_score is valid (handle NaN, None, etc.)
                    if chargeback_result:
                        confidence = chargeback_result.get("confidence_score", 0.0)
                        if confidence is None or (isinstance(confidence, float) and (math.isnan(confidence) or math.isinf(confidence))):
                            confidence = 0.0
                        else:
                            confidence = float(confidence)
                        
                        chargeback_result["confidence_score"] = confidence
                        
                        # Store chargeback prediction (same as Stripe)
                        db["chargeback_predictions"].update_one(
                            {"transaction_id": transaction_id},
                            {"$set": {
                                "transaction_id": transaction_id,
                                "email": db_transaction.get("email"),
                                "chargeback_predicted": chargeback_result.get("chargeback_predicted", False),
                                "confidence_score": confidence,
                                "chargeback_reason": chargeback_result.get("chargeback_reason", ""),
                                "model_type": chargeback_result.get("model_type", "default"),
                                "created_at": datetime.utcnow()
                            }},
                            upsert=True
                        )
                        print(f"✅ Chargeback prediction for {transaction_id}: {chargeback_result.get('chargeback_predicted', False)} (confidence: {confidence:.3f})")
                    else:
                        print(f"⚠️ Chargeback prediction returned no result for {transaction_id}")
                except Exception as cb_error:
                    print(f"❌ Chargeback prediction error for {transaction_id}: {cb_error}")
                    import traceback
                    traceback.print_exc()
                
                # Generate recommendations
                try:
                    from utils.recommendation_engine import recommendation_engine
                    import math
                    
                    db_transaction_for_rec = db["transactions"].find_one({"transaction_id": transaction_id}) or transaction
                    
                    # Get fraud and chargeback results from database
                    fraud_doc = db["fraud_results"].find_one({"transaction_id": transaction_id}) or {}
                    chargeback_doc = db["chargeback_predictions"].find_one({"transaction_id": transaction_id}) or {}
                    
                    # Prepare fraud prediction dict
                    fraud_pred_dict = None
                    if fraud_doc:
                        fraud_conf = fraud_doc.get("confidence_score") or fraud_doc.get("confidence", 0.0)
                        if fraud_conf is None or (isinstance(fraud_conf, float) and (math.isnan(fraud_conf) or math.isinf(fraud_conf))):
                            fraud_conf = 0.0
                        fraud_pred_dict = {
                            "is_fraud": fraud_doc.get("is_fraud", False),
                            "confidence_score": float(fraud_conf),
                            "fraud_reasons": fraud_doc.get("fraud_reasons", [])
                        }
                    
                    # Prepare chargeback prediction dict
                    chargeback_pred_dict = None
                    if chargeback_doc:
                        cb_confidence = chargeback_doc.get("confidence_score") or chargeback_doc.get("confidence", 0.0)
                        if cb_confidence is None or (isinstance(cb_confidence, float) and (math.isnan(cb_confidence) or math.isinf(cb_confidence))):
                            cb_confidence = 0.0
                        chargeback_pred_dict = {
                            "chargeback_predicted": chargeback_doc.get("chargeback_predicted", False),
                            "confidence_score": float(cb_confidence),
                            "chargeback_reason": chargeback_doc.get("chargeback_reason", "")
                        }
                    
                    # Generate recommendations
                    recommendations = recommendation_engine.build_comprehensive_recommendations(
                        transaction=db_transaction_for_rec,
                        fraud_pred=fraud_pred_dict,
                        chargeback_pred=chargeback_pred_dict
                    )
                    
                    # Extract risk_level from recommendations
                    risk_level = recommendations.get("risk_level", "unknown")
                    overall_priority = recommendations.get("overall_priority", "low")
                    combined_risk_score = recommendations.get("combined_risk_score", 0.0)
                    
                    # Save recommendations
                    recommendations_doc = sanitize_for_mongo(recommendations)
                    recommendations_doc["_id"] = f"rec_{transaction_id}"
                    recommendations_doc["transaction_id"] = transaction_id
                    
                    db["recommendations"].replace_one(
                        {"transaction_id": transaction_id},
                        recommendations_doc,
                        upsert=True
                    )
                    
                    # Update transaction with risk_level, priority
                    db["transactions"].update_one(
                        {"transaction_id": transaction_id},
                        {"$set": {
                            "risk_level": risk_level,
                            "overall_priority": overall_priority,
                            "combined_risk_score": combined_risk_score,
                            "recommendations_id": f"rec_{transaction_id}",
                            "recommendations": sanitize_for_mongo(recommendations),
                            "updated_at": datetime.utcnow()
                        }}
                    )
                    
                    print(f"✅ Recommendations generated for {transaction_id} (risk_level: {risk_level}, priority: {overall_priority})")
                except Exception as rec_error:
                    print(f"⚠️ Recommendation generation error for {transaction_id}: {rec_error}")
                    import traceback
                    traceback.print_exc()
                    
            except Exception as pred_error:
                print(f"⚠️ Prediction error for {transaction_id}: {pred_error}")
                import traceback
                traceback.print_exc()
        
        # Get updated subscription for revenue prediction
        updated_subscription = db["subscriptions"].find_one({"subscription_id": subscription_id})
        if updated_subscription:
            revenue_prediction = _get_subscription_revenue_prediction({
                'price_amount': updated_subscription.get('price_amount', float(product_data.get('amount', 0)) / 100),
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
        import traceback
        traceback.print_exc()

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

def _run_fraud_detection(transaction_data):
    """Run fraud detection for transaction"""
    try:
        from predictions.fraud import run_fraud_prediction
        from models.schemas import TransactionRequest
        
        # Prepare fraud detection request - ensure None values are converted to empty strings
        fraud_request = TransactionRequest(
            amount=transaction_data.get('amount', 0.0),
            currency=transaction_data.get('currency', 'usd'),
            email=transaction_data.get('email', ''),
            ip_address=transaction_data.get('ip_address', ''),
            card_country=transaction_data.get('card_country', ''),
            billing_country=transaction_data.get('billing_country', ''),
            card_brand=transaction_data.get('card_brand', ''),
            funding_type=transaction_data.get('funding_type', '') or '',
            fingerprint=transaction_data.get('fingerprint', '') or '',
            risk_score=transaction_data.get('risk_score', 0),
            three_d_secure=transaction_data.get('three_d_secure'),
            cvc_check=transaction_data.get('cvc_check'),
            address_line1_check=transaction_data.get('address_line1_check'),
            postal_code_check=transaction_data.get('postal_code_check'),
            outcome_type=transaction_data.get('outcome_type'),
            seller_message=transaction_data.get('seller_message'),
            network_status=transaction_data.get('network_status')
        )
        
        # Run fraud prediction
        fraud_result = run_fraud_prediction(fraud_request)
        
        if fraud_result and not fraud_result.get('error'):
            # Ensure confidence_score is valid (handle NaN, None, etc.)
            import math
            confidence = fraud_result.get('confidence_score', 0.0)
            if confidence is None or (isinstance(confidence, float) and (math.isnan(confidence) or math.isinf(confidence))):
                confidence = 0.0
            confidence = float(confidence)
            
            # Store fraud result with correct field names
            db["fraud_results"].update_one(
                {"transaction_id": transaction_data.get('transaction_id')},
                {"$set": {
                    "transaction_id": transaction_data.get('transaction_id'),
                    "is_fraud": fraud_result.get('is_fraud', False),
                    "confidence_score": confidence,
                    "risk_level": fraud_result.get('risk_level', 'medium'),
                    "model_type": fraud_result.get('model_type', 'default'),
                    "fraud_reasons": fraud_result.get('fraud_reasons', []),
                    "created_at": datetime.utcnow()
                }},
                upsert=True
            )
            
            return fraud_result
        
        return None
        
    except Exception as e:
        print(f"Error in fraud detection: {e}")
        return None

def _run_chargeback_prediction(transaction_data):
    """Run chargeback prediction for transaction"""
    try:
        from predictions.chargeback import predict_chargeback
        from models.schemas import TransactionRequest
        
        # Prepare chargeback prediction request - ensure None values are converted to empty strings
        chargeback_request = TransactionRequest(
            amount=transaction_data.get('amount', 0.0),
            currency=transaction_data.get('currency', 'usd'),
            email=transaction_data.get('email', ''),
            ip_address=transaction_data.get('ip_address', ''),
            card_country=transaction_data.get('card_country', ''),
            billing_country=transaction_data.get('billing_country', ''),
            card_brand=transaction_data.get('card_brand', ''),
            funding_type=transaction_data.get('funding_type', '') or '',
            fingerprint=transaction_data.get('fingerprint', '') or '',
            risk_score=transaction_data.get('risk_score', 0),
            three_d_secure=transaction_data.get('three_d_secure'),
            cvc_check=transaction_data.get('cvc_check'),
            address_line1_check=transaction_data.get('address_line1_check'),
            postal_code_check=transaction_data.get('postal_code_check'),
            outcome_type=transaction_data.get('outcome_type'),
            seller_message=transaction_data.get('seller_message'),
            network_status=transaction_data.get('network_status')
        )
        
        # Run chargeback prediction
        chargeback_result = predict_chargeback(chargeback_request)
        
        if chargeback_result and not chargeback_result.get('error'):
            # Ensure confidence_score is valid (handle NaN, None, etc.)
            import math
            confidence = chargeback_result.get('confidence_score', 0.0)
            if confidence is None or (isinstance(confidence, float) and (math.isnan(confidence) or math.isinf(confidence))):
                confidence = 0.0
            confidence = float(confidence)
            
            # Store chargeback result with correct field name
            db["chargeback_predictions"].update_one(
                {"transaction_id": transaction_data.get('transaction_id')},
                {"$set": {
                    "transaction_id": transaction_data.get('transaction_id'),
                    "email": transaction_data.get('email', ''),
                    "chargeback_predicted": chargeback_result.get('chargeback_predicted', False),
                    "confidence_score": confidence,
                    "chargeback_reason": chargeback_result.get('chargeback_reason', 'No specific reason'),
                    "model_type": chargeback_result.get('model_type', 'default'),
                    "created_at": datetime.utcnow()
                }},
                upsert=True
            )
            
            return chargeback_result
        
        return None
        
    except Exception as e:
        print(f"Error in chargeback prediction: {e}")
        return None

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
        ensemble_model = model_manager.get_model('subscription_ensemble_model')
        scaler = model_manager.get_model('subscription_revenue_scaler')
        
        if not ensemble_model or not scaler:
            return {"predicted_revenue": subscription_data.get('price_amount', 0), "error": "Models not found"}
        
        # Extract features with defaults for all 23 expected features
        account_age_days = subscription_data.get('account_age_days', 365)
        renewal_count = subscription_data.get('renewal_count', 1)
        average_subscription_value = subscription_data.get('price_amount', 50)
        high_value_customer = 1 if average_subscription_value > 100 else 0
        subscription_duration_days = subscription_data.get('subscription_duration_days', 30)
        is_weekend = 1 if datetime.utcnow().weekday() >= 5 else 0
        
        # Create complete 23-feature vector with defaults
        feature_array = np.array([[
            account_age_days,                    # 0: account_age_days
            renewal_count,                       # 1: renewal_count
            average_subscription_value,          # 2: average_subscription_value
            high_value_customer,                 # 3: high_value_customer
            subscription_duration_days,          # 4: subscription_duration_days
            is_weekend,                          # 5: is_weekend
            7.5,                                 # 6: customer_satisfaction (default)
            0.95,                                # 7: payment_success_rate (default)
            0.2,                                 # 8: churn_risk_score (default)
            account_age_days / 30,               # 9: account_age_months
            subscription_duration_days / 30,     # 10: subscription_age_months
            1.0,                                 # 11: renewal_frequency (default)
            1 if renewal_count <= 1 else 0,     # 12: is_new_customer
            1 if renewal_count > 3 else 0,      # 13: is_established_customer
            1 if average_subscription_value > 75 else 0,  # 14: is_high_engagement
            average_subscription_value,          # 15: revenue_per_month
            0.05,                                # 16: revenue_growth_rate (default)
            0.1,                                 # 17: potential_upsell (default)
            0.3,                                 # 18: risk_score (default)
            0.9,                                 # 19: payment_reliability (default)
            0.2,                                 # 20: support_burden (default)
            is_weekend,                          # 21: weekend_activity
            2 if average_subscription_value > 100 else (1 if average_subscription_value > 50 else 0)  # 22: satisfaction_tier
        ]])
        
        # Scale features
        scaled_features = scaler.transform(feature_array)
        
        # Get prediction
        predicted_revenue = ensemble_model.predict(scaled_features)[0]
        
        return {
            "predicted_revenue": float(predicted_revenue),
            "current_revenue": subscription_data.get('price_amount', 0),
            "growth_rate": (predicted_revenue - subscription_data.get('price_amount', 0)) / max(subscription_data.get('price_amount', 1), 1),
            "features_used": 23
        }
        
    except Exception as e:
        print(f"Error in subscription revenue prediction: {e}")
        return {"predicted_revenue": subscription_data.get('price_amount', 0), "error": str(e)}
