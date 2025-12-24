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
        elif event_type == "charge.dispute.created":
            await _handle_dispute_created(obj)
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
    try:
        charge_id = obj.get("charge")
        refund_id = obj.get("id")
        refund_amount = obj.get("amount", 0) / 100.0
        refund_reason = obj.get("reason", "not_provided")
        refund_created_at = datetime.utcfromtimestamp(obj.get("created"))
        
        # Update transaction
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
        
        # Create refund document
        refund_doc = {
            "transaction_id": charge_id,
            "refund_id": refund_id,
            "amount_refunded": refund_amount,
            "reason": refund_reason,
            "status": obj.get("status", "succeeded"),
            "created_at": refund_created_at,
            "currency": obj.get("currency", "usd"),
            "metadata": obj.get("metadata", {}),
            "gateway": "Stripe"
        }
        
        db["refunds"].update_one(
            {"refund_id": refund_id},
            {"$set": sanitize_for_mongo(refund_doc)},
            upsert=True
        )
        
        print(f"💸 Refund recorded: {refund_id} for transaction {charge_id}")
        
    except Exception as e:
        print(f"❌ Refund handling failed: {str(e)}")
        import traceback
        traceback.print_exc()

async def _handle_dispute_created(obj):
    """Handle dispute created event"""
    try:
        charge_id = obj.get("charge")
        dispute_id = obj.get("id")
        reason = obj.get("reason", "unspecified")
        amount = obj.get("amount", 0) / 100.0
        status = obj.get("status", "needs_response")
        created_at = datetime.utcfromtimestamp(obj.get("created"))
        
        # Update transaction
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
        
        # Create chargeback document
        chargeback_doc = {
            "dispute_id": dispute_id,
            "transaction_id": charge_id,
            "amount": amount,
            "reason": reason,
            "status": status,
            "created_at": created_at,
            "currency": obj.get("currency", "usd"),
            "evidence_due_by": datetime.utcfromtimestamp(obj.get("evidence_due_by")) if obj.get("evidence_due_by") else None,
            "gateway": "Stripe"
        }
        
        db["chargebacks"].update_one(
            {"dispute_id": dispute_id},
            {"$set": sanitize_for_mongo(chargeback_doc)},
            upsert=True
        )
        
        print(f"⚠️ Dispute created: {dispute_id} for transaction {charge_id}")
        
    except Exception as e:
        print(f"❌ Dispute created handling failed: {str(e)}")
        import traceback
        traceback.print_exc()

async def _handle_dispute_closed(obj):
    """Handle dispute closed event"""
    try:
        charge_id = obj.get("charge")
        dispute_id = obj.get("id")
        outcome = obj.get("status")
        
        # Update transaction
        db["transactions"].update_one(
            {"transaction_id": charge_id},
            {"$set": {
                "dispute_status": "closed",
                "dispute_outcome": outcome,
                "dispute_closed_at": datetime.utcnow()
            }}
        )
        
        # Update chargeback document
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
        print(f"❌ Dispute closed handling failed: {str(e)}")
        import traceback
        traceback.print_exc()

async def _handle_invoice_paid(obj):
    """Handle invoice paid event"""
    try:
        invoice = obj
        charge_id = invoice.get("charge")
        if not charge_id:
            print("No charge ID in invoice.paid event")
            return
        
        # Retrieve the charge object from Stripe to get full details
        charge = stripe.Charge.retrieve(charge_id) if charge_id else {}
        
        # Extract billing details
        billing = charge.get("billing_details", {})
        card = charge.get("payment_method_details", {}).get("card", {})
        outcome = charge.get("outcome", {})
        
        # Build comprehensive transaction data
        transaction = {
            "transaction_id": charge_id,
            "email": invoice.get("customer_email") or billing.get("email") or "unknown@example.com",
            "amount": invoice.get("amount_paid", 0) / 100.0,
            "currency": invoice.get("currency", "usd"),
            "gateway": "Stripe",
            "status": invoice.get("status") or charge.get("status", "unknown"),
            
            # Payment method details
            "payment_method": charge.get("payment_method_details", {}).get("type"),
            "card_brand": card.get("brand"),
            "card_country": card.get("country"),
            "fingerprint": card.get("fingerprint"),
            "funding_type": card.get("funding"),
            "three_d_secure": card.get("three_d_secure", {}).get("authenticated") if isinstance(card.get("three_d_secure"), dict) else card.get("three_d_secure"),
            "cvc_check": card.get("checks", {}).get("cvc_check"),
            "address_line1_check": card.get("checks", {}).get("address_line1_check"),
            "postal_code_check": card.get("checks", {}).get("address_postal_code_check"),
            
            # Risk information
            "risk_level": outcome.get("risk_level", "unknown"),
            "risk_score": outcome.get("risk_score", 0),
            "seller_message": outcome.get("seller_message"),
            "network_status": outcome.get("network_status"),
            "outcome_type": outcome.get("type"),
            
            # Billing details
            "billing_name": billing.get("name"),
            "billing_email": billing.get("email"),
            "billing_phone": billing.get("phone"),
            "billing_country": billing.get("address", {}).get("country"),
            "billing_address_line1": billing.get("address", {}).get("line1"),
            "billing_address_line2": billing.get("address", {}).get("line2"),
            "billing_address_postal_code": billing.get("address", {}).get("postal_code"),
            "billing_address_city": billing.get("address", {}).get("city"),
            "billing_address_state": billing.get("address", {}).get("state"),
            
            # IP address (may not be available for invoice payments, try metadata first)
            "ip_address": charge.get("metadata", {}).get("ip_address") or "unknown",
            
            # Transaction flags
            "refunded": charge.get("refunded", False),
            "amount_refunded": charge.get("amount_refunded", 0) / 100.0,
            "disputed": charge.get("disputed", False),
            "captured": charge.get("captured", False),
            "paid": charge.get("paid", False),
            "created_at": datetime.utcfromtimestamp(charge.get("created")) if charge.get("created") else datetime.utcnow()
        }
        
        # Save to database
        db["transactions"].update_one(
            {"transaction_id": transaction["transaction_id"]},
            {"$set": sanitize_for_mongo(transaction)},
            upsert=True
        )
        
        print(f"✅ Invoice paid - Transaction saved: {charge_id}")
        
        # Trigger fraud and chargeback predictions, then generate recommendations
        try:
            from predictions.fraud import process_fraud_workflow
            from predictions.chargeback import predict_chargeback
            from models.schemas import TransactionRequest
            from utils.recommendation_engine import recommendation_engine
            import math
            
            # Process fraud detection
            fraud_result = process_fraud_workflow(transaction)
            print(f"✅ Fraud prediction completed for {charge_id}")
            
            # Process chargeback prediction
            chargeback_req = TransactionRequest(
                amount=transaction.get("amount", 0.0),
                currency=transaction.get("currency", "usd"),
                email=transaction.get("email", ""),
                ip_address=transaction.get("ip_address", ""),
                card_country=transaction.get("card_country", ""),
                billing_country=transaction.get("billing_country", ""),
                card_brand=transaction.get("card_brand", ""),
                funding_type=transaction.get("funding_type", ""),
                fingerprint=transaction.get("fingerprint", ""),
                risk_score=transaction.get("risk_score", 0),
                three_d_secure=transaction.get("three_d_secure"),
                cvc_check=transaction.get("cvc_check"),
                address_line1_check=transaction.get("address_line1_check"),
                postal_code_check=transaction.get("postal_code_check"),
                outcome_type=transaction.get("outcome_type"),
                seller_message=transaction.get("seller_message"),
                network_status=transaction.get("network_status")
            )
            
            chargeback_result = predict_chargeback(chargeback_req)
            
            # Ensure confidence_score is valid (handle NaN, None, etc.)
            if chargeback_result:
                confidence = chargeback_result.get("confidence_score", 0.0)
                # Check for NaN or invalid values
                if confidence is None or (isinstance(confidence, float) and (math.isnan(confidence) or math.isinf(confidence))):
                    confidence = 0.0
                else:
                    confidence = float(confidence)
                
                chargeback_result["confidence_score"] = confidence
                
                # Store chargeback prediction
                db["chargeback_predictions"].update_one(
                    {"transaction_id": charge_id},
                    {"$set": {
                        "transaction_id": charge_id,
                        "email": transaction.get("email"),
                        "chargeback_predicted": chargeback_result.get("chargeback_predicted", False),
                        "confidence_score": confidence,
                        "chargeback_reason": chargeback_result.get("chargeback_reason", ""),
                        "model_type": chargeback_result.get("model_type", "default"),
                        "created_at": datetime.utcnow()
                    }},
                    upsert=True
                )
                print(f"✅ Chargeback prediction saved for {charge_id} (confidence: {confidence:.4f})")
            
            # Generate recommendations using the recommendation engine
            try:
                # Get fraud and chargeback results from database
                fraud_doc = db["fraud_results"].find_one({"transaction_id": charge_id}) or {}
                chargeback_doc = db["chargeback_predictions"].find_one({"transaction_id": charge_id}) or {}
                
                # Prepare fraud prediction dict for recommendation engine
                fraud_pred_dict = None
                if fraud_doc:
                    fraud_pred_dict = {
                        "is_fraud": fraud_doc.get("is_fraud", False),
                        "confidence_score": float(fraud_doc.get("confidence_score", 0.0)) if fraud_doc.get("confidence_score") is not None else 0.0,
                        "fraud_reasons": fraud_doc.get("fraud_reasons", [])
                    }
                
                # Prepare chargeback prediction dict for recommendation engine
                chargeback_pred_dict = None
                if chargeback_doc:
                    cb_confidence = chargeback_doc.get("confidence_score", 0.0)
                    if cb_confidence is None or (isinstance(cb_confidence, float) and (math.isnan(cb_confidence) or math.isinf(cb_confidence))):
                        cb_confidence = 0.0
                    chargeback_pred_dict = {
                        "chargeback_predicted": chargeback_doc.get("chargeback_predicted", False),
                        "confidence_score": float(cb_confidence),
                        "chargeback_reason": chargeback_doc.get("chargeback_reason", "")
                    }
                
                # Generate comprehensive recommendations
                recommendations = recommendation_engine.build_comprehensive_recommendations(
                    transaction=transaction,
                    fraud_pred=fraud_pred_dict,
                    chargeback_pred=chargeback_pred_dict
                )
                
                # Extract risk_level from recommendations
                risk_level = recommendations.get("risk_level", "unknown")
                overall_priority = recommendations.get("overall_priority", "low")
                combined_risk_score = recommendations.get("combined_risk_score", 0.0)
                
                # Save recommendations to separate recommendations collection
                recommendations_doc = sanitize_for_mongo(recommendations)
                recommendations_doc["_id"] = f"rec_{charge_id}"
                recommendations_doc["transaction_id"] = charge_id
                
                db["recommendations"].replace_one(
                    {"transaction_id": charge_id},
                    recommendations_doc,
                    upsert=True
                )
                
                # Update transaction with risk_level, priority, and link to recommendations
                db["transactions"].update_one(
                    {"transaction_id": charge_id},
                    {"$set": {
                        "risk_level": risk_level,
                        "overall_priority": overall_priority,
                        "combined_risk_score": combined_risk_score,
                        "recommendations_id": f"rec_{charge_id}",
                        "recommendations": sanitize_for_mongo(recommendations),  # Keep for backward compatibility
                        "updated_at": datetime.utcnow()
                    }}
                )
                
                print(f"✅ Recommendations generated and saved for {charge_id} (risk_level: {risk_level}, priority: {overall_priority})")
                
            except Exception as rec_error:
                print(f"⚠️ Recommendation generation error for {charge_id}: {rec_error}")
                import traceback
                traceback.print_exc()
                
        except Exception as pred_error:
            print(f"⚠️ Prediction error for {charge_id}: {pred_error}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ Invoice paid handling failed: {str(e)}")
        import traceback
        traceback.print_exc()

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
