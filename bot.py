# bot.py (Fully Updated for merged dataset & PTB v20+)

import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from model import predict_email, training_in_progress, start_background_training

# --- Telegram Bot Handlers ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message when the command /start is issued."""
    await update.message.reply_text(
        "👋 Welcome! I'm a **Phishing Detector Bot**.\n"
        "Send me the **full text** of an email and I will analyze it for potential phishing threats.\n"
        "Use /status to check if the model is ready.",
        parse_mode='Markdown'
    )

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reports the current training status of the AI model."""
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
    """Analyzes the received email text using the trained model."""
    user_text = update.message.text or ""

    if training_in_progress:
        await update.message.reply_text("Model is still training... please try again shortly.")
        return

    # Acknowledge receipt
    await update.message.reply_text("Analyzing email text... 🕵️‍♀️ Please wait a moment.")

    # Prediction
    result, phishing_prob_float, safe_prob_float = predict_email(user_text)

    # Convert probabilities to integer percentages
    phishing_prob = int(round(phishing_prob_float))
    safe_prob = int(round(safe_prob_float))

    # Generate response
    if result == "phishing":
        response_text = (
            f"🛑 **PHISHING ALERT!** 🛑\n\n"
            f"⚠️ Threat Level: HIGH\n"
            f"The model predicts this email is **PHISHING**.\n\n"
            f"📊 Confidence Score:\n"
            f"   - Phishing: `{phishing_prob}%` 🚨\n"
            f"   - Legitimate: `{safe_prob}%`\n\n"
            f"--- 🚫 ACTION REQUIRED 🚫 ---\n"
            f"**DO NOT** click any links, open attachments, or reply with sensitive information.\n"
            f"Report this email to your security team immediately."
        )
    elif result == "legitimate":
        response_text = (
            f"✅ **Email Analysis Complete** ✅\n\n"
            f"👍 Prediction: LEGITIMATE\n"
            f"The model predicts this email is safe.\n\n"
            f"📊 Confidence Score:\n"
            f"   - Legitimate: `{safe_prob}%` ✨\n"
            f"   - Phishing: `{phishing_prob}%`\n\n"
            f"--- 💡 REMINDER 💡 ---\n"
            f"While the AI finds it safe, always be cautious of unexpected requests for personal data."
        )
    else:
        response_text = f"⚠️ An error occurred during analysis: {result}"

    # Send response using Markdown
    await update.message.reply_text(response_text, parse_mode='Markdown')


# --- Bot Runner ---

def start_bot(token: str):
    """Initializes and runs the Telegram bot asynchronously (PTB v20+)."""
    application = Application.builder().token(token).build()

    # Add command and message handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_email_text))

    # Start background training automatically
    start_background_training()

    print("Bot is polling... Press Ctrl-C to stop.")

    # Run the bot asynchronously
    asyncio.run(application.run_polling())
