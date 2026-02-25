# bot.py
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

import my_model.model_manager as model_manager
from my_model.predict import predict_email

# -----------------------------
# Helper: format response text
# -----------------------------

def _risk_label(phishing_prob: float) -> str:
    # Simple, explainable thresholds
    if phishing_prob >= 0.85:
        return "ԲԱՐՁՐ"
    if phishing_prob >= 0.60:
        return "ՄԻՋԻՆ"
    return "ՑԱԾՐ"

def _progress_bar(p: int, length: int = 10) -> str:
    filled = int(round((p / 100) * length))
    filled = max(0, min(length, filled))
    return "█" * filled + "░" * (length - filled)

def _format_result(result: str, phishing_prob: float, safe_prob: float) -> str:
    phishing_percent = int(round(phishing_prob * 100))
    safe_percent = int(round(safe_prob * 100))
    risk = _risk_label(phishing_prob)

    bar_p = _progress_bar(phishing_percent)
    bar_s = _progress_bar(safe_percent)

    if result == "phishing":
        return (
            "🛑 **Ֆիշինգի զգուշացում**\n"
            f"**Ռիսկի մակարդակ՝ {risk}**\n\n"
            "📌 **Արդյունք**: Այս նամակը մեծ հավանականությամբ ֆիշինգային է։\n\n"
            "📊 **Վստահություն (Confidence)**\n"
            f"• Ֆիշինգ: **{phishing_percent}%** `{bar_p}`\n"
            f"• Օրինական (Legitimate): {safe_percent}% `{bar_s}`\n\n"
            "✅ **Ինչ անել հիմա**\n"
            "1) Մի՛ բացեք հղումներ կամ attachment-ներ\n"
            "2) Մի՛ ուղարկեք գաղտնաբառ, կոդ, քարտի տվյալներ\n"
            "3) Ստուգեք ուղարկողի հասցեն և domain-ը\n"
        )

    if result == "legitimate":
        return (
            "✅ **Վերլուծությունը պատրաստ է**\n"
            f"**Ռիսկի մակարդակ՝ {risk}**\n\n"
            "📌 **Արդյունք**: Այս նամակը մեծ հավանականությամբ օրինական է։\n\n"
            "📊 **Վստահություն (Confidence)**\n"
            f"• Օրինական (Legitimate): **{safe_percent}%** `{bar_s}`\n"
            f"• Ֆիշինգ: {phishing_percent}% `{bar_p}`\n\n"
            "💡 **Հիշեցում**\n"
            "Եթե նամակը խնդրում է գաղտնի տվյալներ կամ շտապ գործողություն՝ միշտ կրկնակի ստուգեք։"
        )

    return f"⚠️ **Սխալ**: {result}"

# -----------------------------
# Telegram Bot Handlers
# -----------------------------

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 Բարև։ Ես **Phishing Detector Bot** եմ։\n\n"
        "Ուղարկիր email-ի **ամբողջ տեքստը** (subject+body), և ես կասեմ՝ ֆիշինգ է, թե ոչ։\n"
        "Օգտագործիր `/status`՝ տեսնելու համար մոդելը պատրաստ է, թե ոչ։",
        parse_mode="Markdown"
    )

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if model_manager.training_in_progress:
        await update.message.reply_text(
            "⏳ Մոդելը հիմա **ուսուցվում է**։\n"
            "Մի քանի րոպեից փորձիր նորից։",
            parse_mode="Markdown"
        )
    else:
        ready = (model_manager.model is not None and model_manager.tokenizer is not None)
        if ready:
            await update.message.reply_text(
                "✅ Մոդելը **պատրաստ է**։ Ուղարկիր email-ի տեքստը։",
                parse_mode="Markdown"
            )
        else:
            await update.message.reply_text(
                "⚠️ Մոդելը դեռ **չի բեռնվել**։ Վերագործարկիր bot-ը կամ ստուգիր model paths-ը։",
                parse_mode="Markdown"
            )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "📌 **Օգտագործում**\n"
        "• Ուղարկիր email-ի ամբողջ տեքստը\n"
        "• `/status` — մոդելի կարգավիճակ\n\n"
        "🔒 **Խորհուրդ**\n"
        "Մի՛ ուղարկիր իրական գաղտնաբառեր կամ անձնական տվյալներ։",
        parse_mode="Markdown"
    )

async def handle_email_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_text = (update.message.text or "").strip()

    if not user_text:
        await update.message.reply_text("⚠️ Տեքստ չգտա։ Ուղարկիր email-ի բովանդակությունը։")
        return

    # If training, block predictions
    if model_manager.training_in_progress:
        await update.message.reply_text("⏳ Մոդելը դեռ ուսուցվում է… փորձիր քիչ հետո։")
        return

    # Quick UX feedback
    msg = await update.message.reply_text("🔎 Վերլուծում եմ…")

    try:
        result, phishing_prob, safe_prob = predict_email(user_text)
        response_text = _format_result(result, phishing_prob, safe_prob)
        await msg.edit_text(response_text, parse_mode="Markdown")
    except Exception as e:
        await msg.edit_text(
            "⚠️ Տեղի ունեցավ սխալ վերլուծության ժամանակ։\n"
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