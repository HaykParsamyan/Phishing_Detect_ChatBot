# main.py

import os
import sys
import threading
from bot import start_bot
from model import start_background_training

# --- Configuration ---
# Replace with your actual Telegram Bot Token
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7716630538:AAE_Gac-S4nfeYkXEmOnJStD5kyQlIuOvt8")

if TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN_HERE":
    print("⚠️ WARNING: Please set the TELEGRAM_BOT_TOKEN environment variable or update main.py.")
    # Exit if the token is not set to prevent issues
    sys.exit(1)




def main():
    print("--- Phishing Detector Bot ---")

    # 1. Start Model Training in Background
    # This will load and train the model without blocking the bot's startup.
    print("Starting model training in background thread...")
    start_background_training()

    # 2. Start Telegram Bot
    print("Starting Telegram bot...")
    start_bot(TELEGRAM_BOT_TOKEN)


if __name__ == '__main__':
    main()
