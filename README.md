<!-- PROJECT LOGO -->
<p align="center">
  <!-- If you have a project logo, place it in the repo and update the filename below.
       If no logo is available, you may use an emoji or remove this section. -->
  <!-- <img src="logo.png" alt="TransactIQ Logo" width="160"/> -->
  <img src="https://img.shields.io/badge/AI%20Powered-TransactIQ-4B8DF8.svg?style=for-the-badge&logo=python&logoColor=white" alt="TransactIQ Badge"/>
</p>

<h1 align="center">AI-Powered Payment Intelligence System - TransactIQ</h1>

---

## 🚀 Introduction

**TransactIQ** is a robust, AI-driven payment intelligence platform that empowers businesses to unify, analyze, and operationalize payment data in real-time. Seamlessly integrating with Stripe (with extensibility for other gateways), TransactIQ delivers actionable insights for fraud detection, chargeback risk, payment routing, and revenue forecasting—all in one unified system.

---

## 📝 Overview

This project is an **AI-Powered Payment Intelligence System** designed to fetch, store, and analyze payment data from Stripe (and later extendable to other payment gateways). The system utilizes AI/ML models to detect fraud, predict chargebacks, recommend smart payment routing, and analyze subscription revenue trends.

---

## 🛠️ Features

- ✅ **Fetch and Store Stripe Data**
  - Customers, Subscriptions, and Transactions are stored in MongoDB.
  - Data is unified via `email` for multi-gateway compatibility.

- ✅ **Fraud Detection**
  - Real-time fraud prediction using trained ML models.
  - Flags issues like country mismatch, public email domains, and unusual amounts.
  - Fraud results include confidence score, reasons, and model metadata.

- ✅ **Chargeback Prediction**
  - Predicts likelihood of a chargeback using customer transaction features.

- ✅ **Smart Payment Routing**
  - Recommends optimal gateway (Stripe, PayPal, Adyen) using deep learning.

- ✅ **Subscription Revenue Forecasting**
  - Predicts next month’s revenue per customer using regression models.

- ✅ **Webhook Integration**
  - Stripe webhook listens to `invoice.paid` events and triggers fraud prediction automatically.

- ✅ **Sanitized MongoDB Storage**
  - All AI outputs and transaction data are stored in a clean, queryable structure.

---

## 📦 Technology Stack

| Component        | Tech                                    |
|------------------|-----------------------------------------|
| Backend          | FastAPI (Python)                        |
| Database         | MongoDB                                 |
| Payments         | Stripe (extensible)                     |
| Machine Learning | scikit-learn, XGBoost, TensorFlow/Keras |
| Deployment       | Docker (containerized microservices)    |

---

## 🌟 Real-World Use Cases

- **E-commerce Fraud Prevention:** Instantly flag high-risk transactions to prevent fulfillment of fraudulent orders.
- **Subscription Businesses:** Accurately predict recurring revenue and proactively reduce customer churn.
- **Global SaaS:** Optimize transaction costs and reliability by smart routing to the best payment gateway.
- **Finance Teams:** Automate chargeback risk scoring and streamline intervention workflows.

---

## 🏁 Getting Started

### Prerequisites

- Python 3.8+
- Docker (optional, for containerized deployment)
- MongoDB instance (local or cloud)
- Stripe account & API keys

### Installation

```bash
git clone https://github.com/Lprabodha/ai-payment-intelligence.git
cd ai-payment-intelligence
pip install -r requirements.txt
cp .env.example .env  # Edit to add your Stripe/MongoDB config
```

### Quickstart

```bash
uvicorn app.main:app --reload
```
> The API will be available at [http://localhost:8000](http://localhost:8000)

---

## 📚 API Usage Examples

### Fraud Prediction

```bash
curl -X POST http://localhost:8000/fraud/predict \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "cus_xxx", "amount": 99.00, "currency": "USD"}'
```

**Python Example:**
```python
import requests
resp = requests.post("http://localhost:8000/fraud/predict", json={
    "customer_id": "cus_xxx",
    "amount": 99.00,
    "currency": "USD"
})
print(resp.json())
```

### Revenue Forecast

```bash
curl http://localhost:8000/revenue/forecast?customer_id=cus_xxx
```

_For full API reference and interactive docs, see [`/docs`](./docs) or open Swagger UI at `/docs` when running the server._

```bash
stripe listen  --forward-to http://127.0.0.1:8010//webhook/stripe

```

---

## 🤝 Contributing

We welcome contributions! To get started:

1. Fork this repo and clone your fork.
2. Create a new branch for your feature or bugfix.
3. Add your code and tests.
4. Open a pull request with a clear description of your changes.

Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for code style, issue reporting, and more details.

---

## 📞 Contact & Community

- **Maintainer:** [@Lprabodha](https://github.com/Lprabodha)
- **Issues & Feature Requests:** [GitHub Issues](https://github.com/Lprabodha/ai-payment-intelligence/issues)
- **Community & Q&A:** [GitHub Discussions](https://github.com/Lprabodha/ai-payment-intelligence/discussions)

---

## 📄 Documentation & References

- See the [`/docs`](./docs) folder for setup, API reference, and ML model details.
- [Stripe API Documentation](https://stripe.com/docs/api)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

## 🚧 Future Enhancements

- 🚀 Add support for Solidgate integrations
- 📊 Add a real-time dashboard to visualize fraud trends and KPIs
- 🧠 Auto-retrain models based on new transaction patterns
- 📩 Alerts via email/Slack for high-risk transactions
- 🌍 IP geolocation enrichment

---

<p align="center"><sub>&copy; 2025 Lprabodha. All rights reserved.</sub></p>