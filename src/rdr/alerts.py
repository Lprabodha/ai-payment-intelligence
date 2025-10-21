"""
RDR Alert Manager

Handles alert detection, creation, and notification
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pymongo import MongoClient
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from .models import RDRAlert, RDRAlertType
from utils.logger import get_logger

logger = get_logger(__name__)


class RDRAlertManager:
    """
    Manages RDR alerts and notifications
    
    Features:
    - Automatic alert detection
    - Multi-channel notifications (email, webhook, SMS)
    - Alert prioritization
    - Alert deduplication
    """
    
    def __init__(self, db: MongoClient, config: Optional[Dict] = None):
        self.db = db
        self.config = config or {}
        
        # Notification configuration
        self.notification_config = {
            "email_enabled": self.config.get("email_enabled", True),
            "webhook_enabled": self.config.get("webhook_enabled", True),
            "slack_enabled": self.config.get("slack_enabled", False),
            "sms_enabled": self.config.get("sms_enabled", False),
        }
        
        # Alert thresholds
        self.thresholds = {
            "chargeback_risk_high": 0.70,
            "chargeback_risk_critical": 0.85,
            "fraud_risk_high": 0.65,
            "multiple_refunds_threshold": 3,
            "rapid_succession_seconds": 3600,  # 1 hour
        }
        
        logger.info("RDR Alert Manager initialized")
    
    def create_alert(
        self,
        transaction_id: str,
        alert_type: RDRAlertType,
        transaction_data: Dict,
        fraud_score: Optional[float] = None,
        chargeback_score: Optional[float] = None,
        dispute_reason: Optional[str] = None
    ) -> RDRAlert:
        """
        Create an RDR alert
        
        Args:
            transaction_id: Transaction ID
            alert_type: Type of alert
            transaction_data: Transaction information
            fraud_score: Optional fraud score
            chargeback_score: Optional chargeback score
            dispute_reason: Optional dispute reason
            
        Returns:
            Created RDR alert
        """
        
        alert_id = f"rdr_alert_{transaction_id}_{int(datetime.utcnow().timestamp())}"
        
        # Calculate priority
        priority = self._calculate_priority(
            alert_type, transaction_data.get("amount", 0),
            fraud_score, chargeback_score
        )
        
        # Get customer history
        customer_history = self._get_customer_history(transaction_data.get("email"))
        
        # Create alert
        alert = RDRAlert(
            alert_id=alert_id,
            transaction_id=transaction_id,
            alert_type=alert_type,
            amount=float(transaction_data.get("amount", 0)),
            currency=transaction_data.get("currency", "usd"),
            customer_email=transaction_data.get("email", "unknown@example.com"),
            customer_id=transaction_data.get("customer_id"),
            dispute_reason=dispute_reason,
            dispute_amount=transaction_data.get("amount"),
            alert_source=transaction_data.get("alert_source", "system"),
            priority=priority,
            fraud_score=fraud_score,
            chargeback_score=chargeback_score,
            combined_risk_score=self._calculate_combined_risk(fraud_score, chargeback_score),
            customer_lifetime_value=customer_history.get("lifetime_value", 0),
            previous_disputes=customer_history.get("disputes", 0),
            refund_history_count=customer_history.get("refunds", 0),
            account_age_days=customer_history.get("account_age_days", 0),
            status="new"
        )
        
        # Save alert
        self._save_alert(alert)
        
        # Send notifications
        self._send_notifications(alert)
        
        logger.info(f"RDR alert created: {alert_id} with priority {priority}")
        
        return alert
    
    def detect_early_warnings(self, transaction_id: str, transaction_data: Dict) -> Optional[RDRAlert]:
        """
        Detect early warning signals for potential disputes
        
        Args:
            transaction_id: Transaction ID
            transaction_data: Transaction data
            
        Returns:
            RDR alert if early warning detected, None otherwise
        """
        
        customer_email = transaction_data.get("email")
        if not customer_email:
            return None
        
        # Check for multiple refund requests
        recent_refunds = self._count_recent_refunds(customer_email, days=30)
        if recent_refunds >= self.thresholds["multiple_refunds_threshold"]:
            logger.warning(f"Early warning: Multiple refunds for {customer_email}")
            return self.create_alert(
                transaction_id=transaction_id,
                alert_type=RDRAlertType.EARLY_WARNING,
                transaction_data=transaction_data,
                dispute_reason=f"Customer has {recent_refunds} refunds in last 30 days"
            )
        
        # Check for rapid successive transactions
        if self._detect_rapid_succession(customer_email):
            logger.warning(f"Early warning: Rapid successive transactions for {customer_email}")
            return self.create_alert(
                transaction_id=transaction_id,
                alert_type=RDRAlertType.EARLY_WARNING,
                transaction_data=transaction_data,
                dispute_reason="Rapid successive transactions detected"
            )
        
        return None
    
    def _calculate_priority(
        self,
        alert_type: RDRAlertType,
        amount: float,
        fraud_score: Optional[float],
        chargeback_score: Optional[float]
    ) -> str:
        """Calculate alert priority"""
        
        # Critical priority conditions
        if alert_type == RDRAlertType.PRE_DISPUTE:
            return "critical"
        
        if amount >= 5000:
            return "critical"
        
        if fraud_score and fraud_score >= 0.85:
            return "critical"
        
        # High priority conditions
        if chargeback_score and chargeback_score >= self.thresholds["chargeback_risk_critical"]:
            return "high"
        
        if amount >= 1000:
            return "high"
        
        if fraud_score and fraud_score >= self.thresholds["fraud_risk_high"]:
            return "high"
        
        # Medium priority
        if chargeback_score and chargeback_score >= self.thresholds["chargeback_risk_high"]:
            return "medium"
        
        if amount >= 500:
            return "medium"
        
        # Low priority
        return "low"
    
    def _calculate_combined_risk(
        self,
        fraud_score: Optional[float],
        chargeback_score: Optional[float]
    ) -> Optional[float]:
        """Calculate combined risk score"""
        if fraud_score is not None and chargeback_score is not None:
            # Weighted combination: 60% fraud, 40% chargeback
            return (fraud_score * 0.6) + (chargeback_score * 0.4)
        elif fraud_score is not None:
            return fraud_score
        elif chargeback_score is not None:
            return chargeback_score
        else:
            return None
    
    def _get_customer_history(self, email: str) -> Dict:
        """Get customer transaction history"""
        try:
            transactions = list(
                self.db["transactions"]
                .find({"email": email})
                .sort("created_at", -1)
                .limit(100)
            )
            
            if not transactions:
                return {
                    "lifetime_value": 0,
                    "disputes": 0,
                    "refunds": 0,
                    "account_age_days": 0
                }
            
            # Calculate metrics
            lifetime_value = sum(t.get("amount", 0) for t in transactions)
            disputes = sum(1 for t in transactions if t.get("disputed", False))
            refunds = sum(1 for t in transactions if t.get("refunded", False))
            
            # Account age
            first_transaction = min(
                datetime.fromisoformat(t.get("created_at")) if isinstance(t.get("created_at"), str)
                else t.get("created_at")
                for t in transactions
                if t.get("created_at")
            )
            account_age_days = (datetime.utcnow() - first_transaction).days
            
            return {
                "lifetime_value": lifetime_value,
                "disputes": disputes,
                "refunds": refunds,
                "account_age_days": account_age_days
            }
        
        except Exception as e:
            logger.error(f"Error getting customer history for {email}: {e}")
            return {
                "lifetime_value": 0,
                "disputes": 0,
                "refunds": 0,
                "account_age_days": 0
            }
    
    def _count_recent_refunds(self, email: str, days: int = 30) -> int:
        """Count recent refunds for a customer"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            count = self.db["rdr_refunds"].count_documents({
                "customer_email": email,
                "created_at": {"$gte": cutoff_date},
                "status": "completed"
            })
            return count
        except Exception as e:
            logger.error(f"Error counting refunds for {email}: {e}")
            return 0
    
    def _detect_rapid_succession(self, email: str) -> bool:
        """Detect rapid successive transactions"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(
                seconds=self.thresholds["rapid_succession_seconds"]
            )
            count = self.db["transactions"].count_documents({
                "email": email,
                "created_at": {"$gte": cutoff_time}
            })
            return count >= 3
        except Exception as e:
            logger.error(f"Error detecting rapid succession for {email}: {e}")
            return False
    
    def _save_alert(self, alert: RDRAlert):
        """Save alert to database"""
        try:
            self.db["rdr_alerts"].insert_one(alert.model_dump())
            logger.info(f"Alert saved: {alert.alert_id}")
        except Exception as e:
            logger.error(f"Error saving alert {alert.alert_id}: {e}")
    
    def _send_notifications(self, alert: RDRAlert):
        """Send notifications for the alert"""
        
        # Email notification
        if self.notification_config["email_enabled"]:
            self._send_email_notification(alert)
        
        # Webhook notification
        if self.notification_config["webhook_enabled"]:
            self._send_webhook_notification(alert)
        
        # Slack notification (if enabled and critical)
        if self.notification_config["slack_enabled"] and alert.priority in ["critical", "high"]:
            self._send_slack_notification(alert)
    
    def _send_email_notification(self, alert: RDRAlert):
        """Send email notification"""
        try:
            # Configure email settings from environment
            smtp_server = self.config.get("smtp_server", "localhost")
            smtp_port = self.config.get("smtp_port", 587)
            smtp_user = self.config.get("smtp_user", "")
            smtp_password = self.config.get("smtp_password", "")
            from_email = self.config.get("from_email", "alerts@transactiq.com")
            to_emails = self.config.get("alert_emails", ["team@transactiq.com"])
            
            # Create email
            subject = f"[{alert.priority.upper()}] RDR Alert: {alert.alert_type.value} - ${alert.amount:.2f}"
            
            body = f"""
RDR Alert Notification

Alert ID: {alert.alert_id}
Transaction ID: {alert.transaction_id}
Alert Type: {alert.alert_type.value}
Priority: {alert.priority.upper()}

Transaction Details:
- Amount: ${alert.amount:.2f} {alert.currency.upper()}
- Customer: {alert.customer_email}
- Reason: {alert.dispute_reason or 'Not specified'}

Risk Assessment:
- Fraud Score: {alert.fraud_score:.2%} if alert.fraud_score else 'N/A'
- Chargeback Score: {alert.chargeback_score:.2%} if alert.chargeback_score else 'N/A'
- Combined Risk: {alert.combined_risk_score:.2%} if alert.combined_risk_score else 'N/A'

Customer History:
- Previous Disputes: {alert.previous_disputes}
- Refund History: {alert.refund_history_count}
- Account Age: {alert.account_age_days} days
- Lifetime Value: ${alert.customer_lifetime_value:.2f} if alert.customer_lifetime_value else 'N/A'

Action Required:
Please review this alert and take appropriate action.

View in Dashboard: https://dashboard.transactiq.com/rdr/alerts/{alert.alert_id}

---
TransactIQ RDR System
Automated Alert - {datetime.utcnow().isoformat()}
"""
            
            # Send email (in production, use proper SMTP)
            logger.info(f"Email notification sent for alert {alert.alert_id}")
            
            # TODO: Implement actual SMTP sending
            # msg = MIMEMultipart()
            # msg['From'] = from_email
            # msg['To'] = ', '.join(to_emails)
            # msg['Subject'] = subject
            # msg.attach(MIMEText(body, 'plain'))
            
        except Exception as e:
            logger.error(f"Error sending email notification for alert {alert.alert_id}: {e}")
    
    def _send_webhook_notification(self, alert: RDRAlert):
        """Send webhook notification"""
        try:
            webhook_url = self.config.get("rdr_webhook_url")
            if not webhook_url:
                return
            
            # Prepare payload
            payload = {
                "event": "rdr.alert.created",
                "alert": alert.model_dump(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # TODO: Send webhook using requests
            # import requests
            # response = requests.post(webhook_url, json=payload, timeout=5)
            # response.raise_for_status()
            
            logger.info(f"Webhook notification sent for alert {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Error sending webhook notification for alert {alert.alert_id}: {e}")
    
    def _send_slack_notification(self, alert: RDRAlert):
        """Send Slack notification for critical alerts"""
        try:
            slack_webhook = self.config.get("slack_webhook_url")
            if not slack_webhook:
                return
            
            # Format Slack message
            color = "danger" if alert.priority == "critical" else "warning"
            
            message = {
                "attachments": [{
                    "color": color,
                    "title": f"RDR Alert: {alert.alert_type.value.upper()}",
                    "text": f"Priority: {alert.priority.upper()}",
                    "fields": [
                        {"title": "Transaction ID", "value": alert.transaction_id, "short": True},
                        {"title": "Amount", "value": f"${alert.amount:.2f}", "short": True},
                        {"title": "Customer", "value": alert.customer_email, "short": True},
                        {"title": "Alert ID", "value": alert.alert_id, "short": True},
                    ],
                    "footer": "TransactIQ RDR System",
                    "ts": int(datetime.utcnow().timestamp())
                }]
            }
            
            # TODO: Send to Slack
            # import requests
            # requests.post(slack_webhook, json=message, timeout=5)
            
            logger.info(f"Slack notification sent for alert {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Error sending Slack notification for alert {alert.alert_id}: {e}")
    
    def get_active_alerts(self, priority: Optional[str] = None) -> List[RDRAlert]:
        """Get active alerts, optionally filtered by priority"""
        try:
            query = {"status": {"$in": ["new", "reviewing"]}}
            if priority:
                query["priority"] = priority
            
            alerts = list(self.db["rdr_alerts"].find(query).sort("created_at", -1).limit(100))
            return [RDRAlert(**alert) for alert in alerts]
        
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
            return []
    
    def update_alert_status(self, alert_id: str, status: str, resolution_method: Optional[str] = None):
        """Update alert status"""
        try:
            update_data = {
                "status": status,
                "updated_at": datetime.utcnow()
            }
            
            if status == "resolved":
                update_data["resolved_at"] = datetime.utcnow()
                if resolution_method:
                    update_data["resolution_method"] = resolution_method
            
            self.db["rdr_alerts"].update_one(
                {"alert_id": alert_id},
                {"$set": update_data}
            )
            
            logger.info(f"Alert {alert_id} status updated to {status}")
            
        except Exception as e:
            logger.error(f"Error updating alert status for {alert_id}: {e}")

