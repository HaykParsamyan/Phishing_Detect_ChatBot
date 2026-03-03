# main.py
import os
from my_model.model_manager import load_trained_model
from my_model.train import train_model
from bot import start_bot

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in environment variables.")

def main():
    print("--- Phishing Detector Bot ---")
    print("Checking for trained model...")

    try:
        load_trained_model()
        print("✅ Model loaded from disk.")
    except RuntimeError as e:
        print(f"⚠️ No trained model found: {e}")
        print("➡️ Training  to verify everything works...")
        train_model(sample_frac=1.0, batch_size=2, epochs=3, lr=2e-5, accumulation_steps=8)
        print("✅ Training complete. Reloading model...")
        load_trained_model()

    print("🚀 Model is loaded. Starting Telegram bot...")
    start_bot(TELEGRAM_BOT_TOKEN)

if __name__ == "__main__":
    main()