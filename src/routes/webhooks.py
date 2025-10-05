"""
Webhook API routes
"""
from fastapi import APIRouter, Request, Header, BackgroundTasks, HTTPException
import stripe
import json
import hmac
import hashlib
from config.settings import settings
from webhooks.stripe_handler import handle_stripe_webhook
from webhooks.solidgate_handler import handle_solidgate_webhook
from models.schemas import WebhookResponse

router = APIRouter(prefix="/webhook", tags=["webhooks"])

def verify_solidgate_signature(payload: str, signature: str, secret_key: str) -> bool:
    """Verify Solidgate webhook signature"""
    try:
        expected_signature = hmac.new(
            secret_key.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(signature, expected_signature)
    except Exception as e:
        print(f"Error verifying Solidgate signature: {e}")
        return False

@router.post("/stripe", response_model=WebhookResponse)
async def stripe_webhook(request: Request, background_tasks: BackgroundTasks, stripe_signature: str = Header(None)):
    """Handle Stripe webhook events"""
    try:
        payload = await request.body()
        event = stripe.Webhook.construct_event(payload, stripe_signature, settings.STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        return WebhookResponse(status=400, message=f"Webhook signature verification failed: {str(e)}")

    event_type = event["type"]
    obj = event["data"]["object"]

    # Handle the event in background
    background_tasks.add_task(handle_stripe_webhook, event_type, obj)

    return WebhookResponse(status=200, message=f"Event {event_type} processed successfully")

@router.post("/solidgate", response_model=WebhookResponse)
async def solidgate_webhook(request: Request, background_tasks: BackgroundTasks):
    """Handle Solidgate webhook events for payments and subscriptions"""
    try:
        # Get raw payload
        payload = await request.body()
        payload_str = payload.decode('utf-8')
        
        # Get signature from headers
        signature = request.headers.get('X-Solidgate-Signature', '')
        
        # Verify signature if secret is available
        if settings.SOLIDGATE_API_SECRET and signature:
            if not verify_solidgate_signature(payload_str, signature, settings.SOLIDGATE_API_SECRET):
                return WebhookResponse(status=401, message="Invalid signature")
        
        # Parse webhook data
        webhook_data = json.loads(payload_str)
        
        # Get event type from the webhook data structure
        event_type = webhook_data.get('event_type') or webhook_data.get('callback_type')
        
        # Get event ID and timestamp for idempotency
        event_id = request.headers.get('solidgate-event-id')
        event_created_at = request.headers.get('solidgate-event-created-at')
        
        print(f"Solidgate webhook received: {event_type} (ID: {event_id})")
        
        # Handle the event using the complete Solidgate handler
        await handle_solidgate_webhook(event_type, webhook_data, event_id)
        
        return WebhookResponse(status=200, message=f"Solidgate event {event_type} processed successfully")
        
    except Exception as e:
        print(f"Error processing Solidgate webhook: {e}")
        return WebhookResponse(status=500, message=str(e))
