# bot.py

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from model import predict_email, training_in_progress  # Import the necessary functions/variables


# --- Telegram Bot Handlers ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message when the command /start is issued."""
    await update.message.reply_text(
        "👋 Welcome! I'm a Phishing Detector Bot.\n"
        "Send me the **full text** of an email and I will analyze it for potential phishing threats.\n"
        "Use /status to check if the model is ready."
    )


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reports the current training status of the AI model."""
    if training_in_progress:
        await update.message.reply_text(
            "⏳ The AI model is currently **training** in the background.\n"
            "Please wait a few moments before sending an email for analysis."
        )
    else:
        await update.message.reply_text(
            "✅ The AI model is **ready** for predictions!\n"
            "Go ahead and send me the email text."
        )


async def handle_email_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Analyzes the received text using the trained model."""
    user_text = update.message.text

    if training_in_progress:
        await update.message.reply_text("Model is still training... please try again shortly.")
        return

    await update.message.reply_text("Analyzing email text... 🧐")

    # Call the prediction function from model.py
    result, phishing_prob, safe_prob = predict_email(user_text)

    # Prepare the response
    if result == "phishing":
        response_text = (
            f"🚨 **PHISHING DETECTED** 🚨\n"
            f"**Prediction:** This email is highly likely **Phishing**.\n"
            f"**Confidence:** {phishing_prob}% Phishing / {safe_prob}% Legitimate\n\n"
            f"🚫 **Do NOT** click any links or provide personal information."
        )
    elif result == "legitimate":
        response_text = (
            f"✅ **LEGITIMATE** ✅\n"
            f"**Prediction:** This email appears to be **Legitimate**.\n"
            f"**Confidence:** {safe_prob}% Legitimate / {phishing_prob}% Phishing\n\n"
            f"🔎 *Always be cautious. If in doubt, verify the sender through other means.*"
        )
    else:
        # Handle errors or the "Model is still training" message from predict_email
        response_text = f"An error occurred during analysis: {result}"

    await update.message.reply_text(response_text, parse_mode='Markdown')


def start_bot(token: str):
    """Initializes and runs the Telegram bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(token).build()

    # Register handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("status", status_command))

    # Handle incoming text messages that are not commands
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_email_text))

    # Run the bot until the user presses Ctrl-C
    print("Bot is polling... Press Ctrl-C to stop.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)