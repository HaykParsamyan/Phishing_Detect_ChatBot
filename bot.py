# bot.py
import re
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

import my_model.model_manager as model_manager
from my_model.predict import predict_email

# -----------------------------
# Simple URL heuristics (no retrain)
# -----------------------------

SUSPICIOUS_TLDS = {
    ".zip", ".top", ".xyz", ".click", ".icu", ".cfd", ".rest", ".cam", ".mom", ".work"
}
SUSPICIOUS_WORDS = {
    "login", "verify", "password", "update", "secure", "unlock", "billing", "invoice",
    "suspended", "confirm", "account", "payment"
}

def _extract_urls(text: str):
    return re.findall(r"(https?://[^\s]+|www\.[^\s]+)", (text or "").lower())

def _url_signals(url: str):
    signals = []
    if "xn--" in url:
        signals.append("punycode domain (look-alike risk)")
    if re.search(r"https?://\d{1,3}(\.\d{1,3}){3}", url):
        signals.append("IP-based link")
    if url.count(".") >= 4:
        signals.append("many subdomains")
    if any(url.endswith(tld) or (tld + "/") in url for tld in SUSPICIOUS_TLDS):
        signals.append("suspicious TLD")
    if any(w in url for w in SUSPICIOUS_WORDS):
        signals.append("security/credential keywords in link")
    if "@" in url:
        signals.append("URL contains '@' trick")
    return signals

# -----------------------------
# Helper: format response text
# -----------------------------

def _risk_emoji(risk: str) -> str:
    return {"HIGH": "🛑", "MEDIUM": "⚠️", "LOW": "✅"}.get(risk, "ℹ️")

def _progress_bar(p: int, filled_char: str, length: int = 10) -> str:
    """
    Emoji bar with explicit color/character to avoid bugs.
    Example:
      phishing -> 🟥
      legit    -> 🟩
    """
    filled = int(round((p / 100) * length))
    filled = max(0, min(length, filled))
    empty = "⬜"
    return filled_char * filled + empty * (length - filled)

def _format_result(result: str, phishing_prob: float, safe_prob: float, user_text: str) -> str:
    phish_pct = int(round(phishing_prob * 100))
    legit_pct = int(round(safe_prob * 100))

    # ✅ Correct colors
    bar_p = _progress_bar(phish_pct, "🟥")   # phishing = red
    bar_l = _progress_bar(legit_pct, "🟩")   # legit = green

    urls = _extract_urls(user_text)
    url_notes = []
    if urls:
        for u in urls[:2]:
            url_notes.extend(_url_signals(u))

    # unique notes, keep max 4
    uniq = []
    for s in url_notes:
        if s not in uniq:
            uniq.append(s)
    uniq = uniq[:4]

    # ---- Risk logic (NO contradictions) ----
    if result == "phishing":
        risk = "HIGH" if (phish_pct >= 60 or len(uniq) >= 2) else "MEDIUM"
    else:
        # legitimate: if links + signals exist, never show LOW
        if urls and len(uniq) > 0:
            risk = "MEDIUM"
        else:
            risk = "HIGH" if phish_pct >= 85 else ("MEDIUM" if phish_pct >= 60 else "LOW")

    emoji = _risk_emoji(risk)

    signals_block = ""
    if urls:
        signals_block += f"\n🔗 **Links detected:** {len(urls)}\n"
        if uniq:
            signals_block += "🧠 **Quick signals:**\n" + "\n".join([f"• {s}" for s in uniq]) + "\n"
        else:
            signals_block += "🧠 **Quick signals:** none obvious (still verify the domain)\n"

    scores_block = (
        "📊 **Model Confidence**\n"
        f"• Phishing: **{phish_pct}%** {bar_p}\n"
        f"• Legitimate: **{legit_pct}%** {bar_l}\n"
    )

    if result == "phishing":
        return (
            f"{emoji} **Phishing Risk: {risk}**\n\n"
            "📌 **Verdict:** This message is **risky** and may be phishing.\n\n"
            f"{scores_block}"
            f"{signals_block}\n"
            "✅ **What to do now**\n"
            "1) Don’t click links or open attachments\n"
            "2) Don’t share passwords, OTP codes, or card details\n"
            "3) Verify via the official website/app (type the domain yourself)\n"
        )

    return (
        f"{emoji} **Risk Level: {risk}**\n\n"
        "📌 **Verdict:** This message **looks legitimate**, but verify if it contains links.\n\n"
        f"{scores_block}"
        f"{signals_block}\n"
        "💡 **Reminder**\n"
        "If it asks for urgent action, money, passwords, or OTP codes — verify independently.\n"
    )

# -----------------------------
# Telegram Bot Handlers
# -----------------------------

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 Hi! I’m **Phishing Detector Bot**.\n\n"
        "Send me the **full message text** (subject + body, or a URL), and I’ll estimate whether it looks phishing.\n\n"
        "Commands:\n"
        "• `/status` — model status\n"
        "• `/help` — usage tips",
        parse_mode="Markdown"
    )

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if model_manager.training_in_progress:
        await update.message.reply_text(
            "⏳ The model is **training** right now.\n"
            "Try again later.",
            parse_mode="Markdown"
        )
        return

    ready = (model_manager.model is not None and model_manager.tokenizer is not None)
    if ready:
        await update.message.reply_text(
            "✅ The model is **ready**. Send a message or URL to scan.",
            parse_mode="Markdown"
        )
    else:
        await update.message.reply_text(
            "⚠️ The model is **not loaded**.\n"
            "Restart the bot or check your model paths.",
            parse_mode="Markdown"
        )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "📌 **How to use**\n"
        "• Paste the email/message text (subject + body), or paste a URL\n"
        "• The bot will return a risk estimate\n\n"
        "🔒 **Privacy tip**\n"
        "Don’t send real passwords, OTP codes, or sensitive personal data.",
        parse_mode="Markdown"
    )

async def handle_email_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_text = (update.message.text or "").strip()

    if not user_text:
        await update.message.reply_text("⚠️ I didn’t receive any text. Please paste the message content.")
        return

    if model_manager.training_in_progress:
        await update.message.reply_text("⏳ The model is still training… try again soon.")
        return

    msg = await update.message.reply_text("🔎 Scanning…")

    try:
        result, phishing_prob, safe_prob = predict_email(user_text)
        response_text = _format_result(result, phishing_prob, safe_prob, user_text)
        await msg.edit_text(response_text, parse_mode="Markdown")
    except Exception as e:
        await msg.edit_text(
            "⚠️ Something went wrong while scanning.\n"
            f"`{e}`",
            parse_mode="Markdown"
        )

def start_bot(token: str):
    application = Application.builder().token(token).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_email_text))

    print("Bot is polling... Press Ctrl-C to stop.")
    application.run_polling()