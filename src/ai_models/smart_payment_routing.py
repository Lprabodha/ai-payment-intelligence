import numpy as np
import random
import json
from collections import deque
from pymongo import MongoClient
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.models import load_model
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client['payment_intelligence']
transactions_collection = db['transactions']

def encode_country(country):
    mapping = {'US': 0, 'EU': 1, 'IN': 2, 'Others': 3}
    return mapping.get(country, 3)

def encode_card(card_type):
    mapping = {'VISA': 0, 'MASTERCARD': 1, 'AMEX': 2, 'Others': 3}
    return mapping.get(card_type, 3)

def get_weekday(hour):
    return 1 if hour in [5, 6] else 0

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = models.Sequential([
            layers.Dense(128, input_dim=self.state_size, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state[np.newaxis], verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state[np.newaxis], verbose=0)[0])
            target_f = self.model.predict(state[np.newaxis], verbose=0)
            target_f[0][action] = target
            self.model.fit(state[np.newaxis], target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def get_transaction_data(limit=3000):
    transactions = list(transactions_collection.find().limit(limit))
    dataset = []

    for txn in transactions:
        hour = txn.get('hour', datetime.utcnow().hour)
        amount = txn.get('amount', 0)
        state = [
            np.log1p(amount),                            # amount log scale
            encode_country(txn.get('country_code', 'US')),
            encode_card(txn.get('card_type', 'VISA')),
            round(float(txn.get('customer_risk_score', 0)), 2),
            hour,
            get_weekday(hour),
            1 if hour >= 18 else 0                       # evening flag
        ]

        if txn.get('status') == 'succeeded':
            reward = 10 - txn.get('gateway_fee', 0)
        elif txn.get('fraud_flag', False):
            reward = -10
        else:
            reward = -5

        dataset.append((state, reward))
    return dataset

def train_agent(episodes=20):
    state_size = 7  # Updated with new features
    action_size = 3  # Stripe, PayPal, Adyen
    agent = DQNAgent(state_size, action_size)
    data = get_transaction_data()

    best_total = float("-inf")
    reward_history = []

    for e in range(episodes):
        total_reward = 0
        for (state, reward) in data:
            action = agent.act(np.array(state))
            agent.remember(np.array(state), action, reward, np.array(state), True)
            total_reward += reward

        agent.replay(batch_size=64)
        reward_history.append(total_reward)
        print(f"🎯 Episode {e+1}/{episodes} - Total Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.4f}")

        if total_reward > best_total:
            best_total = total_reward

    os.makedirs("/src/data/models", exist_ok=True)
    agent.model.save("/src/data/models/smart_payment_routing_model.h5")
    with open("/src/data/models/smart_payment_metadata.json", "w") as f:
        json.dump({
            "created_at": datetime.utcnow().isoformat(),
            "episodes": episodes,
            "state_features": [
                "amount_log", "country_encoded", "card_type_encoded",
                "risk_score", "hour", "is_weekend", "is_evening"
            ],
            "final_epsilon": agent.epsilon,
            "reward_history": reward_history
        }, f, indent=4)
    print("✅ Model and metadata saved successfully.")

def predict_gateway(state):
    model = load_model("/src/data/models/smart_payment_routing_model.h5")
    predictions = model.predict(np.array(state).reshape(1, -1), verbose=0)[0]
    gateway_map = {0: "Stripe", 1: "PayPal", 2: "Adyen"}
    for i, prob in enumerate(predictions):
        print(f"{gateway_map[i]}: {prob:.4f}")
    return gateway_map[np.argmax(predictions)]

if __name__ == "__main__":
    train_agent()

    test_state = [np.log1p(250), encode_country('US'), encode_card('VISA'), 0.3, 14, 0, 0]
    print(f"Recommended Gateway: {predict_gateway(test_state)}")
