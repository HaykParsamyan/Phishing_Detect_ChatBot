# main.py (Refactored for merged dataset)

import os
import sys
from bot import start_bot

# --- Configuration ---
# Telegram Bot Token from environment variable or hard-coded (replace with your token)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7716630538:AAE_Gac-S4nfeYkXEmOnJStD5kyQlIuOvt8")

if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN_HERE":
    print("⚠️ ERROR: Please set the TELEGRAM_BOT_TOKEN environment variable or update main.py with your token.")
    sys.exit(1)

def main():
    print("--- Phishing Detector Bot ---")
    print("Starting Telegram bot...")
    start_bot(TELEGRAM_BOT_TOKEN)

if __name__ == "__main__":
    main()
