# main.py
from my_model.model_manager import load_trained_model
from my_model.train import train_model
from bot import start_bot
import os

TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN",
    "7716630538:AAE_Gac-S4nfeYkXEmOnJStD5kyQlIuOvt8"
)

def main():
    print("--- Phishing Detector Bot ---")

    print("Checking for trained model...")
    model_loaded = load_trained_model()

    if not model_loaded:
        print("No trained model found. Starting training...")
        train_model(sample_frac=0.01, epochs=1)  # Use smaller sample for quick test
        print("Training complete!")
        # Reload model after training
        load_trained_model()

    print("Model is loaded. Starting Telegram bot...")
    start_bot(TELEGRAM_BOT_TOKEN)

if __name__ == "__main__":
    main()
