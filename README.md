# 🛡️ AI Phishing Detector Telegram Bot

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-PyTorch-ee4c2c)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/NLP-DistilBERT-yellow)](https://huggingface.co/distilbert-base-uncased)

An end-to-end machine learning solution that detects phishing emails using **Transformers**. This bot allows users to analyze suspicious emails directly from their Telegram app with professional-grade accuracy.

---

## 🚀 Key Features

* **Transformer-Powered:** Fine-tuned `distilbert-base-uncased` model for deep semantic understanding of email text.
* **Confidence Analytics:** Not just a "Yes/No" answer—get a detailed probability breakdown for both Phishing and Legitimate classes.
* **Mixed Precision Training:** Implements `torch.amp` (Automatic Mixed Precision) to speed up training and reduce GPU memory consumption.
* **Seamless Bot Interface:** Built on `python-telegram-bot` (v20+) with asynchronous handling for multiple users.
* **Automatic Setup:** Intelligent startup logic that detects if a model exists; if not, it triggers the training pipeline automatically.

---

## 📂 Project Structure

```text
├── data/                  # Local storage for datasets (CSV files)
├── models/                # Saved weights, config, and tokenizer files
├── my_model/              # Core Logic Package
│   ├── config.py          # Global settings (MAX_LEN, DEVICE, Path constants)
│   ├── data_utils.py      # Pandas logic for text cleaning and loading
│   ├── dataset.py         # PyTorch Dataset implementation
│   ├── evaluation.py      # Metrics calculation (Precision, Recall, F1)
│   ├── model_manager.py   # State management (Loading/Checking models)
│   ├── predict.py         # The "Inference" engine for the bot
│   └── train.py           # Training loop with validation and testing
├── bot.py                 # Telegram command and message handlers
├── main.py                # Entry point: Orchestrates training and bot launch
└── requirements.txt       # List of required Python packages
