"""
Rapid Dispute Resolution (RDR) Module

This module handles:
- RDR alert detection and processing
- Automatic refund processing
- Chargeback prevention
- Alert notifications
- Dispute monitoring
"""

from .engine import RDREngine
from .alerts import RDRAlertManager
from .refund_processor import AutoRefundProcessor
from .models import RDRAlert, RDRDecision, RefundRequest

__all__ = [
    'RDREngine',
    'RDRAlertManager',
    'AutoRefundProcessor',
    'RDRAlert',
    'RDRDecision',
    'RefundRequest'
]

