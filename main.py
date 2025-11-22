# main.py
import os
from bot import start_bot
from my_model.train import train_model
from my_model.model_manager import load_trained_model

TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN",
    "7716630538:AAE_Gac-S4nfeYkXEmOnJStD5kyQlIuOvt8"
)

def main():
    print("--- Phishing Detector Bot ---")
    print("Loading or training model...")

    if not load_trained_model():
        print("No trained model found. Training...")
        train_model(sample_frac=1.0)  # Quick test, change to 1.0 for full dataset
        print("Training complete!")

    print("Starting Telegram bot...")
    start_bot(TELEGRAM_BOT_TOKEN)

if __name__ == "__main__":
    main()
