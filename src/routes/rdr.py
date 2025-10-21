"""
RDR API Routes

Endpoints for Rapid Dispute Resolution system
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from datetime import datetime, timedelta

from rdr import RDREngine, RDRAlertManager, AutoRefundProcessor
from rdr.models import RDRAlert, RDRDecision, RefundRequest, RDRAlertType, RDRMetrics
from database.connection import get_db_connection
from services.fraud_service import FraudDetectionService
from services.chargeback_service import ChargebackPredictionService
from utils.logger import get_logger

router = APIRouter(prefix="/rdr", tags=["RDR"])
logger = get_logger(__name__)


# Dependency injection
def get_rdr_engine():
    """Get RDR engine instance"""
    db = get_db_connection()
    fraud_service = FraudDetectionService(db)
    chargeback_service = ChargebackPredictionService(db)
    return RDREngine(db, fraud_service, chargeback_service)


def get_alert_manager():
    """Get alert manager instance"""
    db = get_db_connection()
    return RDRAlertManager(db)


def get_refund_processor():
    """Get refund processor instance"""
    db = get_db_connection()
    return AutoRefundProcessor(db)


@router.post("/alerts", response_model=RDRAlert)
async def create_rdr_alert(
    transaction_id: str,
    alert_type: RDRAlertType,
    transaction_data: dict,
    fraud_score: Optional[float] = None,
    chargeback_score: Optional[float] = None,
    dispute_reason: Optional[str] = None,
    alert_manager: RDRAlertManager = Depends(get_alert_manager)
):
    """
    Create an RDR alert
    
    This endpoint creates an alert for potential dispute/chargeback
    and triggers the RDR decision process.
    """
    try:
        alert = alert_manager.create_alert(
            transaction_id=transaction_id,
            alert_type=alert_type,
            transaction_data=transaction_data,
            fraud_score=fraud_score,
            chargeback_score=chargeback_score,
            dispute_reason=dispute_reason
        )
        
        return alert
    
    except Exception as e:
        logger.error(f"Error creating RDR alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/process", response_model=RDRDecision)
async def process_alert(
    alert_id: str,
    rdr_engine: RDREngine = Depends(get_rdr_engine),
    alert_manager: RDRAlertManager = Depends(get_alert_manager)
):
    """
    Process an RDR alert and get decision
    
    This endpoint processes the alert through the RDR engine
    and returns a recommended action.
    """
    try:
        # Get alert
        db = get_db_connection()
        alert_data = db["rdr_alerts"].find_one({"alert_id": alert_id})
        
        if not alert_data:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        alert = RDRAlert(**alert_data)
        
        # Process alert
        decision = rdr_engine.process_alert(alert)
        
        # Update alert status
        alert_manager.update_alert_status(alert_id, "reviewing")
        
        return decision
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing RDR alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/refunds", response_model=RefundRequest)
async def issue_refund(
    decision_id: str,
    auto_process: bool = True,
    refund_processor: AutoRefundProcessor = Depends(get_refund_processor)
):
    """
    Issue a refund based on RDR decision
    
    This endpoint processes the refund through the payment gateway.
    """
    try:
        # Get decision
        db = get_db_connection()
        decision_data = db["rdr_decisions"].find_one({"decision_id": decision_id})
        
        if not decision_data:
            raise HTTPException(status_code=404, detail="Decision not found")
        
        decision = RDRDecision(**decision_data)
        
        # Get transaction
        transaction = db["transactions"].find_one({"transaction_id": decision.transaction_id})
        if not transaction:
            raise HTTPException(status_code=404, detail="Transaction not found")
        
        # Process refund
        refund_request = refund_processor.process_refund(decision, transaction)
        
        return refund_request
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error issuing refund for decision {decision_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/refunds/manual")
async def manual_refund(
    transaction_id: str,
    amount: float,
    reason: str,
    refund_processor: AutoRefundProcessor = Depends(get_refund_processor)
):
    """
    Manually issue a refund
    
    This endpoint allows manual refund processing outside of RDR alerts.
    """
    try:
        refund_request = refund_processor.manual_refund(transaction_id, amount, reason)
        return refund_request
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing manual refund: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts", response_model=List[RDRAlert])
async def get_alerts(
    priority: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    alert_manager: RDRAlertManager = Depends(get_alert_manager)
):
    """
    Get RDR alerts
    
    Returns list of RDR alerts, optionally filtered by priority and status.
    """
    try:
        alerts = alert_manager.get_active_alerts(priority=priority)
        
        # Filter by status if provided
        if status:
            alerts = [a for a in alerts if a.status == status]
        
        return alerts[:limit]
    
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/{alert_id}", response_model=RDRAlert)
async def get_alert(alert_id: str):
    """Get specific RDR alert by ID"""
    try:
        db = get_db_connection()
        alert_data = db["rdr_alerts"].find_one({"alert_id": alert_id})
        
        if not alert_data:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        return RDRAlert(**alert_data)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/refunds/{refund_id}", response_model=RefundRequest)
async def get_refund_status(
    refund_id: str,
    refund_processor: AutoRefundProcessor = Depends(get_refund_processor)
):
    """Get refund status by ID"""
    try:
        refund = refund_processor.get_refund_status(refund_id)
        
        if not refund:
            raise HTTPException(status_code=404, detail="Refund not found")
        
        return refund
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting refund status for {refund_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=RDRMetrics)
async def get_rdr_metrics(
    days: int = 30,
    rdr_engine: RDREngine = Depends(get_rdr_engine)
):
    """
    Get RDR system metrics
    
    Returns metrics for the specified time period (default: last 30 days).
    """
    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        metrics = rdr_engine.get_metrics(start_date, end_date)
        
        return RDRMetrics(**metrics)
    
    except Exception as e:
        logger.error(f"Error getting RDR metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/alerts/{alert_id}/status")
async def update_alert_status(
    alert_id: str,
    status: str,
    resolution_method: Optional[str] = None,
    alert_manager: RDRAlertManager = Depends(get_alert_manager)
):
    """Update alert status"""
    try:
        alert_manager.update_alert_status(alert_id, status, resolution_method)
        
        return {"status": "success", "alert_id": alert_id, "new_status": status}
    
    except Exception as e:
        logger.error(f"Error updating alert status for {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transactions/{transaction_id}/check")
async def check_transaction_for_rdr(
    transaction_id: str,
    rdr_engine: RDREngine = Depends(get_rdr_engine),
    alert_manager: RDRAlertManager = Depends(get_alert_manager)
):
    """
    Check if a transaction should trigger an RDR alert
    
    This is typically called after a fraud/chargeback prediction
    to determine if RDR process should be initiated.
    """
    try:
        db = get_db_connection()
        transaction = db["transactions"].find_one({"transaction_id": transaction_id})
        
        if not transaction:
            raise HTTPException(status_code=404, detail="Transaction not found")
        
        # Check for early warning signals
        alert = alert_manager.detect_early_warnings(transaction_id, transaction)
        
        if alert:
            # Process the alert automatically
            decision = rdr_engine.process_alert(alert)
            
            return {
                "rdr_triggered": True,
                "alert": alert,
                "decision": decision
            }
        else:
            return {
                "rdr_triggered": False,
                "message": "No RDR alert required"
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking transaction {transaction_id} for RDR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

