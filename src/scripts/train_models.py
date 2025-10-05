#!/usr/bin/env python3
"""
Training script for AI Payment Intelligence models
Trains all models with proper configuration
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from ai_models.fraud_detection import train as train_fraud
from ai_models.chargeback_prediction import train as train_chargeback
from ai_models.smart_payment_routing import train as train_routing
from ai_models.subscription_revenue_forecasting import train as train_revenue


def train_fraud_detection():
    """Train fraud detection model"""
    print("🔄 Starting fraud detection model training...")
    
    try:
        ensemble, metrics = train_fraud()
        print("✅ Fraud detection model training completed successfully!")
        print(f"📊 Final metrics: {metrics}")
        return True
    except Exception as e:
        print(f"❌ Fraud detection training failed: {e}")
        return False


def train_chargeback_prediction():
    """Train chargeback prediction model"""
    print("🔄 Starting chargeback prediction model training...")
    
    try:
        ensemble, metrics = train_chargeback()
        print("✅ Chargeback prediction model training completed successfully!")
        print(f"📊 Final metrics: {metrics}")
        return True
    except Exception as e:
        print(f"❌ Chargeback prediction training failed: {e}")
        return False


def train_smart_routing():
    """Train smart routing model"""
    print("🔄 Starting smart routing model training...")
    
    try:
        models = train_routing()
        print("✅ Smart routing model training completed successfully!")
        print(f"📊 Trained routing models")
        return True
    except Exception as e:
        print(f"❌ Smart routing training failed: {e}")
        return False


def train_revenue_forecasting():
    """Train revenue forecasting model"""
    print("🔄 Starting revenue forecasting model training...")
    
    try:
        models = train_revenue()
        print("✅ Revenue forecasting model training completed successfully!")
        print(f"📊 Trained revenue forecasting models")
        return True
    except Exception as e:
        print(f"❌ Revenue forecasting training failed: {e}")
        return False


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train AI Payment Intelligence models")
    parser.add_argument(
        "--models", 
        nargs="+", 
        choices=["fraud", "chargeback", "routing", "revenue", "all"],
        default=["all"],
        help="Models to train"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip training if models already exist"
    )
    
    args = parser.parse_args()
    
    print("🚀 Starting AI Payment Intelligence model training...")
    print(f"📋 Models to train: {args.models}")
    
    # Check if models already exist
    existing_models = []
    model_path = Path("/src/data/models")
    
    if (model_path / "fraud_detection_pipeline.pkl").exists():
        existing_models.append("fraud")
    if (model_path / "chargeback_prediction_pipeline.pkl").exists():
        existing_models.append("chargeback")
    if (model_path / "smart_payment_routing_model.h5").exists():
        existing_models.append("routing")
    if (model_path / "subscription_revenue_forecasting_model.pkl").exists():
        existing_models.append("revenue")
    
    if args.skip_existing and existing_models:
        print(f"⚠️ Found existing models: {existing_models}")
        print("Skipping existing models. Use --force to retrain.")
    
    # Determine which models to train
    models_to_train = args.models
    if "all" in models_to_train:
        models_to_train = ["fraud", "chargeback", "routing", "revenue"]
    
    # Train models
    results = {}
    
    if "fraud" in models_to_train:
        if not (args.skip_existing and "fraud" in existing_models):
            results["fraud"] = train_fraud_detection()
        else:
            print("⏭️ Skipping fraud model (already exists)")
            results["fraud"] = True
    
    if "chargeback" in models_to_train:
        if not (args.skip_existing and "chargeback" in existing_models):
            results["chargeback"] = train_chargeback_prediction()
        else:
            print("⏭️ Skipping chargeback model (already exists)")
            results["chargeback"] = True
    
    if "routing" in models_to_train:
        if not (args.skip_existing and "routing" in existing_models):
            results["routing"] = train_smart_routing()
        else:
            print("⏭️ Skipping routing model (already exists)")
            results["routing"] = True
    
    if "revenue" in models_to_train:
        if not (args.skip_existing and "revenue" in existing_models):
            results["revenue"] = train_revenue_forecasting()
        else:
            print("⏭️ Skipping revenue model (already exists)")
            results["revenue"] = True
    
    # Summary
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    
    print(f"\n📊 Training Summary:")
    print(f"✅ Successful: {successful}/{total}")
    
    for model, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"   {model}: {status}")
    
    if successful == total:
        print("🎉 All models trained successfully!")
        return 0
    else:
        print("❌ Some models failed to train")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
