import subprocess

def run_script(script_name):
    print(f"\n🚀 Training: {script_name}")
    subprocess.run(["python", f"ai_models/{script_name}"], check=True)

if __name__ == "__main__":
    scripts = [
        "chargeback_prediction.py",
        "fraud_detection.py",
        "smart_payment_routing.py",
        "subscription_revenue_forecasting.py"
    ]

    for script in scripts:
        run_script(script)

    print("\n✅ All models trained successfully.")
