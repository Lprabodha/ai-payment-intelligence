# TransactIQ API Documentation

## Table of Contents
1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Base URL](#base-url)
4. [Endpoints](#endpoints)
   - [Health Check](#health-check)
   - [Fraud Detection](#fraud-detection)
   - [Chargeback Prediction](#chargeback-prediction)
   - [Smart Payment Routing](#smart-payment-routing)
   - [Subscription Revenue Forecasting](#subscription-revenue-forecasting)
   - [Risk Scoring Engine](#risk-scoring-engine)
   - [Webhooks](#webhooks)
5. [Data Models](#data-models)
6. [Error Handling](#error-handling)
7. [Rate Limits](#rate-limits)
8. [Examples](#examples)

## Overview

The TransactIQ API provides advanced machine learning-powered payment processing, fraud detection, chargeback prediction, smart routing, and risk assessment capabilities. Built with FastAPI and powered by multiple AI models including XGBoost, Random Forest, Neural Networks, and ensemble methods.

### Key Features
- **Real-time Fraud Detection**: ML-powered fraud detection with 95%+ accuracy
- **Chargeback Prediction**: Advanced chargeback risk assessment
- **Smart Payment Routing**: AI-driven gateway selection optimization
- **Subscription Revenue Forecasting**: Predictive analytics for subscription businesses
- **Real-time Risk Scoring**: Comprehensive risk assessment engine
- **Webhook Integration**: Stripe and Solidgate webhook processing

## Authentication

Currently, the API does not require authentication for development/testing purposes. In production, implement proper API key authentication.

```bash
# Example with API key (future implementation)
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     https://api.transactiq.com/health
```

## Base URL

```
http://localhost:8010
```

## Endpoints

### Health Check

#### GET /
Get API health status and basic information.

**Response:**
```json
{
  "status": "healthy",
  "message": "TransactIQ API is running",
  "version": "1.0.0",
  "timestamp": "2025-10-05T14:35:04.441905Z"
}
```

#### GET /health
Detailed health check with system status.

**Response:**
```json
{
  "status": "healthy",
  "database": "connected",
  "models": {
    "fraud_detection": "loaded",
    "chargeback_prediction": "loaded",
    "smart_routing": "not_found",
    "subscription_revenue": "loaded"
  },
  "timestamp": "2025-10-05T14:35:04.441905Z"
}
```

### Fraud Detection

#### POST /predict/fraud
Predict fraud risk for a transaction.

**Request Body:**
```json
{
  "amount": 150.0,
  "currency": "usd",
  "email": "user@example.com",
  "ip_address": "192.168.1.1",
  "card_country": "US",
  "billing_country": "US",
  "card_brand": "VISA",
  "funding_type": "credit",
  "fingerprint": "fp_123456789",
  "risk_score": 25,
  "three_d_secure": "authenticated",
  "cvc_check": "pass",
  "address_line1_check": "pass",
  "postal_code_check": "pass",
  "outcome_type": "authorized",
  "seller_message": "Your transaction was successful.",
  "network_status": "approved_by_network"
}
```

**Response:**
```json
{
  "is_fraud": false,
  "confidence_score": 0.85,
  "risk_level": "low",
  "fraud_reasons": [
    "Transaction amount is within normal range",
    "Customer has good transaction history"
  ],
  "model_type": "ensemble",
  "prediction_time": "2025-10-05T14:35:04.441905Z"
}
```

### Chargeback Prediction

#### POST /predict/chargeback
Predict chargeback risk for a transaction.

#### GET /predict/jobs/predict-chargebacks
Run background job to predict chargebacks for recent transactions.

**Request Body for POST /predict/chargeback:**
```json
{
  "amount": 150.0,
  "currency": "usd",
  "email": "user@example.com",
  "ip_address": "192.168.1.1",
  "card_country": "US",
  "billing_country": "US",
  "card_brand": "VISA",
  "funding_type": "credit",
  "fingerprint": "fp_123456789",
  "risk_score": 25,
  "three_d_secure": "authenticated",
  "cvc_check": "pass",
  "address_line1_check": "pass",
  "postal_code_check": "pass",
  "outcome_type": "authorized",
  "seller_message": "Your transaction was successful.",
  "network_status": "approved_by_network"
}
```

**No request body needed for GET /predict/jobs/predict-chargebacks**

**Response for POST /predict/chargeback:**
```json
{
  "chargeback_predicted": false,
  "confidence_score": 0.92,
  "chargeback_reason": "No strong indicators",
  "model_type": "ensemble",
  "prediction_time": "2025-10-05T14:35:04.441905Z"
}
```

**Response for GET /predict/jobs/predict-chargebacks:**
```json
{
  "message": "Processed 1 transactions for chargeback prediction",
  "total_transactions": 13,
  "predictions_made": 1
}
```

### AI-Powered Recommendations

#### GET /predict/transactions/{transaction_id}/recommendations
Get comprehensive AI-powered recommendations for a transaction based on fraud risk, chargeback risk, customer history, and behavioral patterns.

**URL Parameters:**
- `transaction_id` (string, required): The transaction ID to get recommendations for

**Features:**
- Real-time customer history analysis
- Behavioral pattern detection
- Risk-based recommendation levels
- Actionable, specific recommendations
- Rich contextual insights with emoji indicators
- Categorized action plans (immediate, short-term, long-term, monitoring)

**Response:**
```json
{
  "transaction_id": "txn_abc123",
  "created_at": "2025-10-07T14:35:04.441905",
  "overall_priority": "high",
  "risk_assessment": {
    "fraud": {
      "detected": true,
      "confidence": 0.7234,
      "level": "high",
      "recommendations": [
        "HIGH FRAUD RISK: Hold transaction for 4-hour manual review",
        "VERIFICATION: Require 3DS authentication and SMS confirmation",
        "DOCUMENTATION: Request billing address verification",
        "MONITORING: Flag account for enhanced monitoring for 30 days",
        "LIMITS: Reduce daily transaction limit to $1,000 for 7 days",
        "VELOCITY ALERT: Customer made 5+ transactions in 1 hour - Implement 30-minute cooling period",
        "HIGH VALUE: Transaction over $2,000 - Require 3DS authentication and email confirmation"
      ]
    },
    "chargeback": {
      "predicted": false,
      "confidence": 0.3421,
      "level": "medium",
      "recommendations": [
        "MEDIUM CHARGEBACK RISK: Customer has 30%+ dispute probability",
        "CONFIRMATION: Send transaction confirmation email",
        "DOCUMENTATION: Standard transaction documentation",
        "FOLLOW-UP: Customer satisfaction check within 72 hours",
        "MONITORING: Monitor for customer service complaints",
        "WEEKEND RISK: Weekend transactions have 40% higher dispute rate - Send immediate confirmation"
      ]
    },
    "routing": {
      "recommendations": [
        "Route to high-security gateway with advanced fraud detection",
        "Use gateway with strongest 3DS and authentication capabilities",
        "High-value routing: Use premium gateway with enhanced support"
      ]
    }
  },
  "action_plan": {
    "immediate_actions": [
      "HIGH FRAUD RISK: Hold transaction for 4-hour manual review",
      "VELOCITY ALERT: Customer made 5+ transactions in 1 hour - Implement 30-minute cooling period"
    ],
    "short_term_actions": [
      "VERIFICATION: Require 3DS authentication and SMS confirmation",
      "DOCUMENTATION: Request billing address verification",
      "CONFIRMATION: Send transaction confirmation email",
      "FOLLOW-UP: Customer satisfaction check within 72 hours"
    ],
    "long_term_actions": [
      "MONITORING: Flag account for enhanced monitoring for 30 days",
      "MONITORING: Monitor for customer service complaints"
    ],
    "monitoring_actions": [
      "MONITORING: Flag account for enhanced monitoring for 30 days",
      "MONITORING: Monitor for customer service complaints"
    ]
  },
  "insights": [
    "Fraud indicators: High transaction velocity, Unusual amount pattern, Device risk detected",
    "High fraud risk detected (confidence: 72.3%)",
    "Velocity spike detected - 7 transactions in past hour",
    "Device fingerprint reused 3 times - potential account takeover risk",
    "New customer account (less than 7 days old) - higher risk profile"
  ],
  "amount_context": {
    "amount": 2234.56,
    "currency": "usd",
    "tier": "high"
  },
  "ttl_days": 30
}
```

**Recommendation Categories:**

1. **Fraud-Specific Recommendations:**
   - Velocity alerts (transaction frequency)
   - Device risk detection
   - IP reputation checks
   - Amount testing pattern detection
   - Time anomaly alerts
   - Card BIN risk assessment
   - High-value transaction protocols
   - Customer chargeback history
   - Account age-based verification

2. **Chargeback-Specific Recommendations:**
   - Weekend transaction risk management
   - Subscription transaction protocols
   - Digital goods protection
   - Cross-border transaction verification
   - Delivery confirmation requirements
   - Customer satisfaction monitoring
   - Dispute prevention strategies
   - Industry-specific recommendations (travel, software, events)

3. **Payment Routing Recommendations:**
   - Gateway selection based on risk profile
   - 3DS authentication requirements
   - Dispute management tool recommendations
   - High-value routing strategies

**Insight Categories:**
- Fraud indicators and detection
- High risk alerts and warnings
- Chargeback risk assessment
- Cross-border transaction analysis
- Velocity spike detection
- Device pattern analysis
- Time anomaly detection
- Amount testing patterns
- Customer history analysis
- Refund pattern evaluation
- New customer risk profiling
- Low risk / legitimate transactions

**Error Responses:**
```json
{
  "detail": "Transaction not found"
}
```

**Example Usage:**
```bash
curl -X GET "http://localhost:8010/api/predict/transactions/txn_abc123/recommendations"
```

### Smart Payment Routing

#### GET /transactions/{txid}/recommendations
Get payment gateway recommendations for a transaction.

**Path Parameters:**
- `txid` (string): Transaction ID

**Response:**
```json
{
  "transaction_id": "tx_123456789",
  "recommended_gateway": "Stripe",
  "confidence": 0.88,
  "all_scores": {
    "Stripe": 0.88,
    "PayPal": 0.65,
    "Adyen": 0.42
  },
  "reasoning": [
    "High success rate for US transactions",
    "Low fees for this amount range",
    "Good fraud protection"
  ],
  "prediction_time": "2025-10-05T14:35:04.441905Z"
}
```

### Subscription Revenue Forecasting

#### POST /predict/subscription-revenue
Predict subscription revenue and growth.

**Request Body:**
```json
{
  "subscription_id": "sub_123456789",
  "account_age_days": 365,
  "renewal_count": 3,
  "average_subscription_value": 50.0,
  "subscription_duration_days": 30,
  "customer_satisfaction": 8.5,
  "payment_success_rate": 0.95,
  "churn_risk_score": 0.2
}
```

**Response:**
```json
{
  "predicted_revenue": 65.5,
  "current_revenue": 50.0,
  "growth_rate": 0.31,
  "confidence": 0.87,
  "model_type": "ensemble",
  "prediction_time": "2025-10-05T14:35:04.441905Z"
}
```

### Risk Scoring Engine

#### POST /risk/score
Score a transaction for comprehensive risk assessment.

**Request Body:**
```json
{
  "transaction_id": "tx_123456789",
  "user_id": "user_123",
  "email": "user@example.com",
  "amount": 150.0,
  "currency": "USD",
  "payment_method": "card",
  "card_brand": "VISA",
  "card_country": "US",
  "ip_address": "192.168.1.1",
  "device_fingerprint": "fp_123456789",
  "user_agent": "Mozilla/5.0...",
  "billing_country": "US",
  "merchant_id": "merchant_123",
  "product_category": "electronics"
}
```

**Response:**
```json
{
  "transaction_id": "tx_123456789",
  "overall_risk_score": 0.35,
  "risk_level": "low",
  "decision_action": "approve",
  "confidence": 0.88,
  "reasoning": [
    "Low risk indicators",
    "Customer has good transaction history",
    "IP address is clean"
  ],
  "ml_prediction": {
    "fraud_probability": 0.25,
    "confidence": 0.85,
    "model_version": "1.0"
  },
  "rule_evaluation": {
    "rules_triggered": [],
    "total_rule_score": 0.1
  },
  "velocity_features": {
    "transactions_last_hour": 1,
    "transactions_last_day": 3,
    "amount_last_hour": 150.0,
    "amount_last_day": 450.0
  },
  "ip_features": {
    "is_proxy": false,
    "is_vpn": false,
    "is_tor": false,
    "risk_score": 0.1
  },
  "device_features": {
    "is_mobile": false,
    "is_bot": false,
    "browser_risk": 0.1,
    "os_risk": 0.2
  },
  "geo_features": {
    "country_risk_score": 0.1,
    "timezone_mismatch": false,
    "location_consistency": 0.9
  },
  "assessment_time": "2025-10-05T14:35:04.441905Z"
}
```

#### POST /risk/score/batch
Score multiple transactions for risk assessment.

**Request Body:**
```json
{
  "transactions": [
    {
      "transaction_id": "tx_1",
      "amount": 100.0,
      "email": "user1@example.com"
    },
    {
      "transaction_id": "tx_2",
      "amount": 200.0,
      "email": "user2@example.com"
    }
  ]
}
```

**Response:**
```json
{
  "results": [
    {
      "transaction_id": "tx_1",
      "overall_risk_score": 0.25,
      "risk_level": "low",
      "decision_action": "approve"
    },
    {
      "transaction_id": "tx_2",
      "overall_risk_score": 0.45,
      "risk_level": "medium",
      "decision_action": "challenge"
    }
  ],
  "total_transactions": 2
}
```

#### GET /risk/assessment/{transaction_id}
Get cached risk assessment for a transaction.

**Response:**
```json
{
  "transaction_id": "tx_123456789",
  "overall_risk_score": 0.35,
  "risk_level": "low",
  "decision_action": "approve",
  "confidence": 0.88,
  "reasoning": ["Low risk indicators"],
  "assessment_time": "2025-10-05T14:35:04.441905Z"
}
```

#### GET /risk/profile/{user_id}
Get risk profile for a user.

**Response:**
```json
{
  "user_id": "user_123",
  "average_risk_score": 0.32,
  "high_risk_transactions": 2,
  "total_assessments": 15,
  "risk_trend": "decreasing",
  "last_updated": "2025-10-05T14:35:04.441905Z"
}
```

#### PUT /risk/thresholds
Update risk scoring thresholds.

**Request Body:**
```json
{
  "low_threshold": 0.3,
  "medium_threshold": 0.6,
  "high_threshold": 0.8,
  "critical_threshold": 0.9,
  "approve_threshold": 0.4,
  "review_threshold": 0.7,
  "decline_threshold": 0.9
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Risk thresholds updated successfully",
  "thresholds": {
    "low_threshold": 0.3,
    "medium_threshold": 0.6,
    "high_threshold": 0.8,
    "critical_threshold": 0.9,
    "approve_threshold": 0.4,
    "review_threshold": 0.7,
    "decline_threshold": 0.9
  }
}
```

#### GET /risk/metrics
Get risk scoring metrics and statistics.

**Response:**
```json
{
  "total_assessments": 15420,
  "risk_level_distribution": {
    "low": 8540,
    "medium": 4320,
    "high": 2100,
    "critical": 460
  },
  "decision_action_distribution": {
    "approve": 12000,
    "review": 2500,
    "challenge": 800,
    "decline": 120
  },
  "average_risk_score_today": 0.42,
  "assessments_today": 156,
  "cache_stats": {
    "status": "active",
    "hit_rate": 0.87
  }
}
```

### Webhooks

#### POST /webhook/stripe
Handle Stripe webhook events.

**Headers:**
- `Stripe-Signature`: Webhook signature for verification

**Request Body:**
```json
{
  "id": "evt_123456789",
  "object": "event",
  "type": "payment_intent.succeeded",
  "data": {
    "object": {
      "id": "pi_123456789",
      "amount": 15000,
      "currency": "usd",
      "status": "succeeded"
    }
  }
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Webhook processed successfully",
  "event_id": "evt_123456789",
  "processed_at": "2025-10-05T14:35:04.441905Z"
}
```

#### POST /webhook/solidgate
Handle Solidgate webhook events.

**Headers:**
- `Solidgate-Event-Type`: Event type (e.g., "card_gate.order.updated")
- `Solidgate-Event-Id`: Unique event identifier
- `Signature`: Webhook signature for verification

**Request Body:**
```json
{
  "order": {
    "order_id": "order_123456789",
    "amount": 15000,
    "currency": "USD",
    "status": "approved",
    "customer_email": "user@example.com",
    "card": {
      "brand": "VISA",
      "country": "US"
    }
  }
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Webhook processed successfully",
  "event_id": "event_123456789",
  "processed_at": "2025-10-05T14:35:04.441905Z"
}
```

## Data Models

### TransactionRequest
```json
{
  "amount": 150.0,
  "currency": "usd",
  "email": "user@example.com",
  "ip_address": "192.168.1.1",
  "card_country": "US",
  "billing_country": "US",
  "card_brand": "VISA",
  "funding_type": "credit",
  "fingerprint": "fp_123456789",
  "risk_score": 25,
  "three_d_secure": "authenticated",
  "cvc_check": "pass",
  "address_line1_check": "pass",
  "postal_code_check": "pass",
  "outcome_type": "authorized",
  "seller_message": "Your transaction was successful.",
  "network_status": "approved_by_network"
}
```

### FraudPredictionResponse
```json
{
  "is_fraud": false,
  "confidence_score": 0.85,
  "risk_level": "low",
  "fraud_reasons": ["Transaction amount is within normal range"],
  "model_type": "ensemble",
  "prediction_time": "2025-10-05T14:35:04.441905Z"
}
```

### ChargebackPredictionResponse
```json
{
  "chargeback_predicted": false,
  "confidence_score": 0.92,
  "chargeback_reason": "No strong indicators",
  "model_type": "ensemble",
  "prediction_time": "2025-10-05T14:35:04.441905Z"
}
```

### RiskAssessment
```json
{
  "transaction_id": "tx_123456789",
  "overall_risk_score": 0.35,
  "risk_level": "low",
  "decision_action": "approve",
  "confidence": 0.88,
  "reasoning": ["Low risk indicators"],
  "ml_prediction": {
    "fraud_probability": 0.25,
    "confidence": 0.85,
    "model_version": "1.0"
  },
  "rule_evaluation": {
    "rules_triggered": [],
    "total_rule_score": 0.1
  },
  "assessment_time": "2025-10-05T14:35:04.441905Z"
}
```

## Error Handling

### Error Response Format
```json
{
  "error": "Validation Error",
  "message": "Invalid request data",
  "details": {
    "field": "amount",
    "issue": "Amount must be greater than 0"
  },
  "timestamp": "2025-10-05T14:35:04.441905Z"
}
```

### HTTP Status Codes
- `200 OK`: Request successful
- `400 Bad Request`: Invalid request data
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error

### Common Error Messages
- `"Model not available"`: AI model not loaded
- `"Invalid transaction data"`: Request validation failed
- `"Database connection error"`: Database connectivity issue
- `"Webhook signature invalid"`: Webhook verification failed

## Rate Limits

Currently, no rate limits are implemented. In production, implement appropriate rate limiting:

- **Fraud Detection**: 1000 requests/minute
- **Chargeback Prediction**: 500 requests/minute
- **Risk Scoring**: 2000 requests/minute
- **Webhooks**: 5000 requests/minute

## Examples

### Python Example
```python
import requests

# Fraud detection
response = requests.post('http://localhost:8010/predict/fraud', json={
    "amount": 150.0,
    "currency": "usd",
    "email": "user@example.com",
    "ip_address": "192.168.1.1",
    "card_country": "US",
    "card_brand": "VISA"
})

result = response.json()
print(f"Fraud prediction: {result['is_fraud']}")
print(f"Confidence: {result['confidence_score']}")
```

### JavaScript Example
```javascript
// Risk scoring
const response = await fetch('http://localhost:8010/risk/score', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    transaction_id: 'tx_123456789',
    amount: 150.0,
    email: 'user@example.com',
    ip_address: '192.168.1.1'
  })
});

const result = await response.json();
console.log(`Risk level: ${result.risk_level}`);
console.log(`Decision: ${result.decision_action}`);
```

### cURL Example
```bash
# Health check
curl -X GET http://localhost:8010/health

# Fraud detection
curl -X POST http://localhost:8010/predict/fraud \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 150.0,
    "currency": "usd",
    "email": "user@example.com",
    "ip_address": "192.168.1.1",
    "card_country": "US",
    "card_brand": "VISA"
  }'

# Risk scoring
curl -X POST http://localhost:8010/risk/score \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "tx_123456789",
    "amount": 150.0,
    "email": "user@example.com",
    "ip_address": "192.168.1.1"
  }'
```

## Support

For API support and questions:
- **Email**: support@transactiq.com
- **Documentation**: https://docs.transactiq.com
- **Status Page**: https://status.transactiq.com

## Changelog

### Version 1.0.0 (2025-10-05)
- Initial release
- Fraud detection API
- Chargeback prediction API
- Smart payment routing
- Subscription revenue forecasting
- Real-time risk scoring engine
- Stripe and Solidgate webhook integration
