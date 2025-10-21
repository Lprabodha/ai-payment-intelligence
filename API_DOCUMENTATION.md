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

**cURL Example:**
```bash
curl -X GET http://localhost:8010/
```

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

**cURL Example:**
```bash
curl -X GET http://localhost:8010/health
```

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

**cURL Example:**
```bash
curl -X POST http://localhost:8010/predict/fraud \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

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

#### POST /jobs/predict-chargebacks
Predict chargeback risk for a transaction.

**cURL Example:**
```bash
curl -X POST http://localhost:8010/jobs/predict-chargebacks \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 150.0,
    "currency": "usd",
    "email": "user@example.com",
    "ip_address": "192.168.1.1",
    "card_country": "US",
    "billing_country": "US",
    "card_brand": "VISA",
    "funding_type": "credit",
    "fingerprint": "fp_123456789",
    "risk_score": 25
  }'
```

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
  "chargeback_predicted": false,
  "confidence_score": 0.92,
  "chargeback_reason": "No strong indicators",
  "model_type": "ensemble",
  "prediction_time": "2025-10-05T14:35:04.441905Z"
}
```

### Smart Payment Routing

#### GET /transactions/{txid}/recommendations
Get payment gateway recommendations for a transaction.

**Path Parameters:**
- `txid` (string): Transaction ID

**cURL Example:**
```bash
curl -X GET http://localhost:8010/transactions/tx_123456789/recommendations
```

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

**cURL Example:**
```bash
curl -X POST http://localhost:8010/predict/subscription-revenue \
  -H "Content-Type: application/json" \
  -d '{
    "subscription_id": "sub_123456789",
    "account_age_days": 365,
    "renewal_count": 3,
    "average_subscription_value": 50.0,
    "subscription_duration_days": 30,
    "customer_satisfaction": 8.5,
    "payment_success_rate": 0.95,
    "churn_risk_score": 0.2
  }'
```

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

**cURL Example:**
```bash
curl -X POST http://localhost:8010/risk/score \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "tx_123456789",
    "amount": 150.0,
    "currency": "usd",
    "email": "user@example.com",
    "ip_address": "192.168.1.1",
    "card_country": "US",
    "billing_country": "US",
    "card_brand": "VISA"
  }'
```

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

**cURL Example:**
```bash
curl -X POST http://localhost:8010/risk/score/batch \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {
        "transaction_id": "tx_001",
        "amount": 100.0,
        "email": "user1@example.com"
      },
      {
        "transaction_id": "tx_002",
        "amount": 200.0,
        "email": "user2@example.com"
      }
    ]
  }'
```

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

**cURL Example:**
```bash
curl -X GET http://localhost:8010/risk/assessment/tx_123456789
```

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

**cURL Example:**
```bash
curl -X GET http://localhost:8010/risk/profile/user_123456789
```

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

**cURL Example:**
```bash
curl -X PUT http://localhost:8010/risk/thresholds \
  -H "Content-Type: application/json" \
  -d '{
    "low_threshold": 0.3,
    "medium_threshold": 0.6,
    "high_threshold": 0.8,
    "auto_decline_threshold": 0.95
  }'
```

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

**cURL Example:**
```bash
curl -X GET http://localhost:8010/risk/metrics
```

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

**cURL Example:**
```bash
curl -X POST http://localhost:8010/webhook/stripe \
  -H "Content-Type: application/json" \
  -H "Stripe-Signature: t=1234567890,v1=signature_here" \
  -d '{
    "id": "evt_123456789",
    "object": "event",
    "type": "payment_intent.succeeded",
    "data": {
      "object": {
        "id": "pi_123456789",
        "amount": 1500,
        "currency": "usd"
      }
    }
  }'
```

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

**cURL Example:**
```bash
curl -X POST http://localhost:8010/webhook/solidgate \
  -H "Content-Type: application/json" \
  -H "Solidgate-Event-Type: card_gate.order.updated" \
  -H "Solidgate-Event-Id: evt_123456789" \
  -H "Signature: signature_here" \
  -d '{
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
  }'
```

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
