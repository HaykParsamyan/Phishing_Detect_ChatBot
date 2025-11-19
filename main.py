# main.py
import os
from bot import start_bot
from model import load_trained_model, train_model

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7716630538:AAE_Gac-S4nfeYkXEmOnJStD5kyQlIuOvt8")

def main():
    print("--- Phishing Detector Bot ---")
    print("Loading or training model...")

    if not load_trained_model():
        print("No trained model found. Training...")
        train_model(sample_frac=0.5)   # train on 1% of data

    print("Starting Telegram bot...")
    start_bot(TELEGRAM_BOT_TOKEN)

if __name__ == "__main__":
    main()
