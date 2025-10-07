import numpy as np
import pandas as pd
import random
import json
from collections import deque
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client['payment_intelligence']
transactions_collection = db['transactions']

class FeatureEncoder:
    """Enhanced feature encoding for payment routing"""
    
    def __init__(self):
        self.country_encoder = LabelEncoder()
        self.card_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def encode_country(self, country):
        """Encode country with risk weighting"""
        risk_scores = {
            'US': 0.1, 'CA': 0.1, 'GB': 0.1, 'AU': 0.1, 'DE': 0.1,
            'FR': 0.15, 'IT': 0.2, 'ES': 0.2, 'BR': 0.3, 'MX': 0.3,
            'IN': 0.4, 'CN': 0.5, 'RU': 0.6, 'NG': 0.7
        }
        return risk_scores.get(country, 0.5)
    
    def encode_card(self, card_type):
        """Encode card type with success rates"""
        success_rates = {
            'VISA': 0.95, 'MASTERCARD': 0.94, 'AMEX': 0.92, 
            'DISCOVER': 0.88, 'JCB': 0.85, 'DINERS': 0.82
        }
        return success_rates.get(card_type, 0.85)
    
    def get_time_features(self, hour, day_of_week=None):
        """Extract comprehensive time-based features"""
        return {
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'is_weekend': 1 if day_of_week in [5, 6] else 0,
            'is_business_hours': 1 if 9 <= hour <= 17 else 0,
            'is_evening': 1 if hour >= 18 else 0,
            'is_night': 1 if hour < 6 else 0
        }
    
    def get_amount_features(self, amount):
        """Extract amount-based features"""
        return {
            'amount_log': np.log1p(amount),
            'amount_sqrt': np.sqrt(amount),
            'amount_category': 0 if amount < 100 else (1 if amount < 500 else 2)
        }

# Initialize global encoder
encoder = FeatureEncoder()

class EnhancedDQNAgent:
    """Enhanced DQN Agent with better architecture and training"""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)  # Increased memory
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        
        # Dual network architecture
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()
        
        # Training tracking
        self.training_history = []
        
    def _build_model(self):
        """Build enhanced neural network with dropout and batch normalization"""
        model = models.Sequential([
            layers.Dense(256, input_dim=self.state_size, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.1),
            
            layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size=None):
        """Experience replay with target network"""
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([e[0] for e in minibatch])
        actions = np.array([e[1] for e in minibatch])
        rewards = np.array([e[2] for e in minibatch])
        next_states = np.array([e[3] for e in minibatch])
        dones = np.array([e[4] for e in minibatch])

        # Current Q values
        current_q_values = self.model.predict(states, verbose=0)
        
        # Next Q values from target network
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        # Calculate target Q values
        target_q_values = rewards + self.gamma * np.max(next_q_values, axis=1) * (1 - dones)
        
        # Update Q values
        for i, action in enumerate(actions):
            current_q_values[i][action] = target_q_values[i]
        
        # Train the model
        history = self.model.fit(states, current_q_values, epochs=1, verbose=0)
        self.training_history.append(history.history['loss'][0])
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def get_enhanced_transaction_data(limit=5000):
    """Enhanced transaction data processing with better features"""
    transactions = list(transactions_collection.find().limit(limit))
    dataset = []

    for txn in transactions:
        # Extract basic transaction info
        amount = float(txn.get('amount', 0))
        hour = txn.get('hour', datetime.utcnow().hour)
        day_of_week = txn.get('day_of_week', datetime.utcnow().weekday())
        country = txn.get('card_country', 'US')
        card_type = txn.get('card_brand', 'VISA')
        risk_score = float(txn.get('risk_score', 0))
        
        # Enhanced feature extraction
        amount_features = encoder.get_amount_features(amount)
        time_features = encoder.get_time_features(hour, day_of_week)
        
        # Build comprehensive state vector
        state = [
            amount_features['amount_log'],
            amount_features['amount_sqrt'], 
            amount_features['amount_category'],
            encoder.encode_country(country),
            encoder.encode_card(card_type),
            risk_score / 100.0,  # Normalize risk score
            time_features['hour_sin'],
            time_features['hour_cos'],
            time_features['is_weekend'],
            time_features['is_business_hours'],
            time_features['is_evening'],
            time_features['is_night']
        ]
        
        # Enhanced reward calculation
        reward = calculate_enhanced_reward(txn, amount)
        
        dataset.append((state, reward))
    
    return dataset

def calculate_enhanced_reward(txn, amount):
    """Calculate comprehensive reward based on transaction outcome"""
    base_reward = 0
    
    # Success/failure reward
    if txn.get('status') == 'succeeded':
        base_reward = 10
        
        # Bonus for high-value transactions
        if amount > 1000:
            base_reward += 5
            
        # Penalty for high fees
        gateway_fee = txn.get('gateway_fee', 0)
        if gateway_fee > 0:
            base_reward -= gateway_fee * 0.1
            
    elif txn.get('disputed', False):
        base_reward = -20  # High penalty for chargebacks
    elif txn.get('refunded', False):
        base_reward = -10  # Medium penalty for refunds
    else:
        base_reward = -5   # Failure penalty
    
    # Risk-based adjustments
    risk_score = txn.get('risk_score', 0)
    if risk_score > 80:
        base_reward -= 5  # Penalty for high-risk transactions
    
    # Time-based adjustments
    hour = txn.get('hour', 12)
    if 2 <= hour <= 6:  # Night transactions
        base_reward -= 2
    
    return base_reward

def train_enhanced_agent(episodes=50):
    """Enhanced training with better architecture and monitoring"""
    state_size = 12  # Enhanced feature set
    action_size = 3  # Stripe, PayPal, Adyen
    
    agent = EnhancedDQNAgent(state_size, action_size)
    data = get_enhanced_transaction_data()

    best_total = float("-inf")
    reward_history = []
    loss_history = []
    
    print("Starting enhanced DQN training...")
    print(f"Training on {len(data)} transactions over {episodes} episodes")

    for e in range(episodes):
        total_reward = 0
        episode_losses = []
        
        # Shuffle data for better training
        random.shuffle(data)
        
        for i, (state, reward) in enumerate(data):
            state = np.array(state)
            
            # Choose action
            action = agent.act(state, training=True)
            
            # Simulate next state (simplified - in real scenario this would be dynamic)
            next_state = state.copy()
            
            # Determine if episode is done
            done = (i == len(data) - 1)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            total_reward += reward
            
            # Train every few steps
            if len(agent.memory) > agent.batch_size and i % 10 == 0:
                agent.replay()
                if agent.training_history:
                    episode_losses.append(agent.training_history[-1])
        
        # Update target network every 10 episodes
        if e % 10 == 0:
            agent.update_target_network()
        
        reward_history.append(total_reward)
        if episode_losses:
            avg_loss = np.mean(episode_losses)
            loss_history.append(avg_loss)
        
        print(f"Episode {e+1}/{episodes} - Reward: {total_reward:.2f} - "
              f"Epsilon: {agent.epsilon:.4f} - "
              f"Avg Loss: {avg_loss:.4f}" if episode_losses else "No training")

        if total_reward > best_total:
            best_total = total_reward
            # Save best model
            agent.model.save("/src/data/models/smart_payment_routing_best.h5")

    # Save final model and metadata
    os.makedirs("/src/data/models", exist_ok=True)
    agent.model.save("/src/data/models/smart_payment_routing_model.h5")
    
    metadata = {
        "created_at": datetime.utcnow().isoformat(),
        "model_version": "2.0.0",
        "episodes": episodes,
        "state_size": state_size,
        "action_size": action_size,
        "state_features": [
            "amount_log", "amount_sqrt", "amount_category",
            "country_risk", "card_success_rate", "risk_score_norm",
            "hour_sin", "hour_cos", "is_weekend", 
            "is_business_hours", "is_evening", "is_night"
        ],
        "final_epsilon": agent.epsilon,
        "best_total_reward": best_total,
        "reward_history": reward_history,
        "loss_history": loss_history,
        "training_samples": len(data)
    }
    
    with open("/src/data/models/smart_payment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    
    print("Enhanced model and metadata saved successfully.")
    print(f"Best total reward: {best_total:.2f}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    
    return agent

def predict_gateway_enhanced(amount, country, card_type, risk_score, hour, day_of_week=None):
    """Enhanced gateway prediction with comprehensive feature extraction"""
    try:
        model = load_model("/src/data/models/smart_payment_routing_model.h5")
        
        # Extract features using the same encoder
        amount_features = encoder.get_amount_features(amount)
        time_features = encoder.get_time_features(hour, day_of_week)
        
        state = [
            amount_features['amount_log'],
            amount_features['amount_sqrt'], 
            amount_features['amount_category'],
            encoder.encode_country(country),
            encoder.encode_card(card_type),
            risk_score / 100.0,
            time_features['hour_sin'],
            time_features['hour_cos'],
            time_features['is_weekend'],
            time_features['is_business_hours'],
            time_features['is_evening'],
            time_features['is_night']
        ]
        
        predictions = model.predict(np.array(state).reshape(1, -1), verbose=0)[0]
        gateway_map = {0: "Stripe", 1: "PayPal", 2: "Adyen"}
        
        # Calculate confidence scores
        softmax_scores = np.exp(predictions) / np.sum(np.exp(predictions))
        
        results = {}
        for i, gateway in gateway_map.items():
            results[gateway] = {
                'q_value': float(predictions[i]),
                'confidence': float(softmax_scores[i])
            }
        
        recommended = gateway_map[np.argmax(predictions)]
        
        return {
            'recommended_gateway': recommended,
            'gateway_scores': results,
            'state_features': state
        }
        
    except Exception as e:
        print(f"Error in gateway prediction: {e}")
        return {
            'recommended_gateway': 'Stripe',  # Fallback
            'gateway_scores': {'Stripe': {'confidence': 1.0}},
            'error': str(e)
        }

def train_ensemble_routing_model():
    """Train ensemble model combining DQN with traditional ML"""
    print("Training ensemble routing model...")
    
    # Train DQN
    dqn_agent = train_enhanced_agent(episodes=30)
    
    # Prepare data for traditional ML
    data = get_enhanced_transaction_data()
    X = np.array([state for state, _ in data])
    y = np.array([reward for _, reward in data])
    
    # Train Random Forest for comparison
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    rf_model.fit(X, y)
    
    # Save ensemble components
    import joblib
    joblib.dump(rf_model, "/src/data/models/smart_routing_rf.pkl")
    
    print("Ensemble routing model training completed.")

if __name__ == "__main__":
    # Train the enhanced model
    train_ensemble_routing_model()
