# bot.py
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from my_model.predict import predict_email
from my_model.model_manager import training_in_progress  # moved to separate module

import asyncio

# --- Telegram Bot Handlers ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 Welcome! I'm a **Phishing Detector Bot**.\n"
        "Send me the **full text** of an email and I will analyze it for potential phishing threats.\n"
        "Use /status to check if the model is ready.",
        parse_mode='Markdown'
    )

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if training_in_progress:
        await update.message.reply_text(
            "⏳ The AI model is currently **training** in the background.\n"
            "Please wait a few moments before sending an email for analysis.",
            parse_mode='Markdown'
        )
    else:
        await update.message.reply_text(
            "✅ The AI model is **ready** for predictions!\n"
            "Go ahead and send me the email text.",
            parse_mode='Markdown'
        )

async def handle_email_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_text = update.message.text

    if training_in_progress:
        await update.message.reply_text("Model is still training... please try again shortly.")
        return

    await update.message.reply_text("Analyzing email text... 🕵️‍♀️ Please wait a moment.")

    # Prediction
    result, phishing_prob, safe_prob = predict_email(user_text)
    phishing_percent = int(round(phishing_prob * 100))
    safe_percent = int(round(safe_prob * 100))

    if result == "phishing":
        response_text = (
            f"🛑 **PHISHING ALERT!** 🛑\n\n"
            f"⚠️ Threat Level: HIGH\n"
            f"The model predicts this email is **PHISHING**.\n\n"
            f"📊 Confidence Score:\n"
            f"   - **Phishing:** {phishing_percent}% 🚨\n"
            f"   - Legitimate: {safe_percent}%\n\n"
            f"--- 🚫 ACTION REQUIRED --- 🚫\n"
            f"**DO NOT** click links, open attachments, or reply with sensitive info.\n"
            f"Report this email immediately."
        )
    elif result == "legitimate":
        response_text = (
            f"✅ **Email Analysis Complete** ✅\n\n"
            f"**Prediction: LEGITIMATE**\n"
            f"The model predicts this email is safe and **Legitimate**.\n\n"
            f"📊 Confidence Score:\n"
            f"   - Legitimate: {safe_percent}% ✨\n"
            f"   - Phishing: {phishing_percent}%\n\n"
            f"--- 💡 REMINDER --- 💡\n"
            f"While the AI finds it safe, **always be cautious** of requests for personal data or unexpected messages."
        )
    else:
        response_text = f"An error occurred during analysis: {result}"

    await update.message.reply_text(response_text, parse_mode='Markdown')

def start_bot(token: str):
    """Initializes and runs the Telegram bot using the Application builder."""
    application = Application.builder().token(token).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_email_text))

    print("Bot is polling... Press Ctrl-C to stop.")
    application.run_polling()
