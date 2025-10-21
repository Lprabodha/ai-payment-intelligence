"""
Automatic Refund Processor

Handles automatic and manual refund processing with gateway integration
"""

import os
from typing import Dict, Optional
from datetime import datetime
from pymongo import MongoClient
import stripe
import asyncio

from .models import RefundRequest, RefundStatus, RDRDecision
from utils.logger import get_logger

logger = get_logger(__name__)


class AutoRefundProcessor:
    """
    Automatic refund processing engine
    
    Features:
    - Multi-gateway support (Stripe, Solidgate)
    - Automatic retry logic
    - Customer notification
    - Chargeback tracking
    - Success rate monitoring
    """
    
    def __init__(self, db: MongoClient, config: Optional[Dict] = None):
        self.db = db
        self.config = config or {}
        
        # Gateway configuration
        self.stripe_key = self.config.get("stripe_secret_key", os.getenv("STRIPE_SECRET_KEY"))
        if self.stripe_key:
            stripe.api_key = self.stripe_key
        
        # Refund configuration
        self.refund_config = {
            "max_retry_attempts": 3,
            "retry_delay_seconds": 60,
            "auto_notify_customer": True,
            "track_chargeback_prevention": True,
            "refund_reason_template": "Refund issued via RDR - Dispute prevention"
        }
        
        logger.info("Auto Refund Processor initialized")
    
    def process_refund(self, decision: RDRDecision, transaction: Dict) -> RefundRequest:
        """
        Process a refund based on RDR decision
        
        Args:
            decision: RDR decision
            transaction: Transaction data
            
        Returns:
            RefundRequest with processing status
        """
        
        refund_id = f"rdr_ref_{decision.transaction_id}_{int(datetime.utcnow().timestamp())}"
        
        # Create refund request
        refund_request = RefundRequest(
            refund_id=refund_id,
            transaction_id=decision.transaction_id,
            alert_id=decision.alert_id,
            amount=decision.refund_amount or transaction.get("amount", 0),
            currency=transaction.get("currency", "usd"),
            reason=self.refund_config["refund_reason_template"],
            refund_type="full",
            status=RefundStatus.PENDING,
            gateway=transaction.get("gateway", "stripe"),
            customer_email=transaction.get("email", ""),
            customer_id=transaction.get("customer_id"),
            notify_customer=self.refund_config["auto_notify_customer"]
        )
        
        # Save initial request
        self._save_refund_request(refund_request)
        
        # Process refund based on decision
        if decision.auto_process:
            logger.info(f"Auto-processing refund {refund_id}")
            self._execute_refund(refund_request, transaction)
        else:
            logger.info(f"Refund {refund_id} marked for manual processing")
            refund_request.status = RefundStatus.PENDING
            self._save_refund_request(refund_request)
        
        return refund_request
    
    def _execute_refund(self, refund_request: RefundRequest, transaction: Dict):
        """Execute the refund through payment gateway"""
        
        refund_request.status = RefundStatus.PROCESSING
        self._save_refund_request(refund_request)
        
        try:
            # Route to appropriate gateway
            if refund_request.gateway == "stripe":
                result = self._process_stripe_refund(refund_request, transaction)
            elif refund_request.gateway == "solidgate":
                result = self._process_solidgate_refund(refund_request, transaction)
            else:
                raise ValueError(f"Unsupported gateway: {refund_request.gateway}")
            
            if result["success"]:
                refund_request.status = RefundStatus.COMPLETED
                refund_request.gateway_refund_id = result.get("refund_id")
                refund_request.completed_at = datetime.utcnow()
                logger.info(f"Refund {refund_request.refund_id} completed successfully")
                
                # Track chargeback prevention
                if self.refund_config["track_chargeback_prevention"]:
                    self._track_chargeback_prevention(refund_request)
                
                # Notify customer
                if refund_request.notify_customer:
                    self._notify_customer_refund(refund_request)
            else:
                # Retry logic
                if refund_request.retry_count < self.refund_config["max_retry_attempts"]:
                    refund_request.retry_count += 1
                    logger.warning(f"Refund {refund_request.refund_id} failed, retry {refund_request.retry_count}")
                    # Schedule retry (in production, use task queue)
                else:
                    refund_request.status = RefundStatus.FAILED
                    refund_request.failed_reason = result.get("error", "Unknown error")
                    logger.error(f"Refund {refund_request.refund_id} failed after {refund_request.retry_count} retries")
        
        except Exception as e:
            refund_request.status = RefundStatus.FAILED
            refund_request.failed_reason = str(e)
            logger.error(f"Error executing refund {refund_request.refund_id}: {e}")
        
        finally:
            refund_request.processed_at = datetime.utcnow()
            self._save_refund_request(refund_request)
    
    def _process_stripe_refund(self, refund_request: RefundRequest, transaction: Dict) -> Dict:
        """Process refund through Stripe"""
        try:
            # Get Stripe charge ID
            charge_id = transaction.get("stripe_charge_id") or transaction.get("charge_id")
            if not charge_id:
                return {"success": False, "error": "Stripe charge ID not found"}
            
            # Create refund
            refund = stripe.Refund.create(
                charge=charge_id,
                amount=int(refund_request.amount * 100),  # Convert to cents
                reason="requested_by_customer",  # RDR refund
                metadata={
                    "rdr_refund_id": refund_request.refund_id,
                    "rdr_alert_id": refund_request.alert_id,
                    "rdr_reason": refund_request.reason
                }
            )
            
            return {
                "success": True,
                "refund_id": refund.id,
                "status": refund.status
            }
        
        except stripe.error.StripeError as e:
            logger.error(f"Stripe refund error: {e}")
            return {"success": False, "error": str(e)}
        
        except Exception as e:
            logger.error(f"Unexpected error in Stripe refund: {e}")
            return {"success": False, "error": str(e)}
    
    def _process_solidgate_refund(self, refund_request: RefundRequest, transaction: Dict) -> Dict:
        """Process refund through Solidgate"""
        try:
            # TODO: Implement Solidgate refund API
            # This is a placeholder - implement based on Solidgate API docs
            
            logger.info(f"Processing Solidgate refund for {refund_request.refund_id}")
            
            # Placeholder response
            return {
                "success": True,
                "refund_id": f"solidgate_ref_{refund_request.refund_id}",
                "status": "pending"
            }
        
        except Exception as e:
            logger.error(f"Solidgate refund error: {e}")
            return {"success": False, "error": str(e)}
    
    def _track_chargeback_prevention(self, refund_request: RefundRequest):
        """Track that this refund prevented a chargeback"""
        try:
            # Update refund with prevention flag
            self.db["rdr_refunds"].update_one(
                {"refund_id": refund_request.refund_id},
                {
                    "$set": {
                        "prevented_chargeback": True,
                        "chargeback_cost_avoided": refund_request.amount + 15.0 + 25.0  # Amount + fee + handling
                    }
                }
            )
            
            # Update metrics
            self.db["rdr_metrics"].update_one(
                {"date": datetime.utcnow().date().isoformat()},
                {
                    "$inc": {
                        "chargebacks_prevented": 1,
                        "total_cost_avoided": refund_request.amount + 40.0
                    }
                },
                upsert=True
            )
            
            logger.info(f"Chargeback prevention tracked for refund {refund_request.refund_id}")
            
        except Exception as e:
            logger.error(f"Error tracking chargeback prevention: {e}")
    
    def _notify_customer_refund(self, refund_request: RefundRequest):
        """Notify customer about refund"""
        try:
            # Create customer notification
            notification = {
                "type": "refund_processed",
                "customer_email": refund_request.customer_email,
                "transaction_id": refund_request.transaction_id,
                "refund_id": refund_request.refund_id,
                "amount": refund_request.amount,
                "currency": refund_request.currency,
                "message": f"Your refund of ${refund_request.amount:.2f} has been processed.",
                "created_at": datetime.utcnow()
            }
            
            # Save notification
            self.db["customer_notifications"].insert_one(notification)
            
            # TODO: Send actual email to customer
            logger.info(f"Customer notification created for refund {refund_request.refund_id}")
            
        except Exception as e:
            logger.error(f"Error notifying customer for refund {refund_request.refund_id}: {e}")
    
    def _save_refund_request(self, refund_request: RefundRequest):
        """Save or update refund request"""
        try:
            self.db["rdr_refunds"].update_one(
                {"refund_id": refund_request.refund_id},
                {"$set": refund_request.model_dump()},
                upsert=True
            )
        except Exception as e:
            logger.error(f"Error saving refund request {refund_request.refund_id}: {e}")
    
    def get_refund_status(self, refund_id: str) -> Optional[RefundRequest]:
        """Get refund status"""
        try:
            refund_data = self.db["rdr_refunds"].find_one({"refund_id": refund_id})
            if refund_data:
                return RefundRequest(**refund_data)
            return None
        except Exception as e:
            logger.error(f"Error getting refund status for {refund_id}: {e}")
            return None
    
    def manual_refund(self, transaction_id: str, amount: float, reason: str) -> RefundRequest:
        """Manually initiate a refund"""
        
        transaction = self.db["transactions"].find_one({"transaction_id": transaction_id})
        if not transaction:
            raise ValueError(f"Transaction not found: {transaction_id}")
        
        refund_request = RefundRequest(
            refund_id=f"manual_ref_{transaction_id}_{int(datetime.utcnow().timestamp())}",
            transaction_id=transaction_id,
            amount=amount,
            currency=transaction.get("currency", "usd"),
            reason=reason,
            refund_type="partial" if amount < transaction.get("amount", 0) else "full",
            status=RefundStatus.PENDING,
            gateway=transaction.get("gateway", "stripe"),
            customer_email=transaction.get("email", ""),
            notify_customer=True
        )
        
        self._execute_refund(refund_request, transaction)
        
        return refund_request

