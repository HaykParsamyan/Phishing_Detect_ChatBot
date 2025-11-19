# features.py
import pandas as pd
import torch
from transformers import AutoTokenizer

# --- Config ---
DATA_PATH = "final_data/all_phishing_master_dataset.csv"
MAX_LEN = 256
GLOBAL_NUMERIC_COLS = []  # empty since you don't have numeric features

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def load_and_prepare_dataset():
    """Load CSV and prepare 'text' and 'labels' columns."""
    df = pd.read_csv(DATA_PATH, dtype={'subject': str, 'body': str, 'label': int}, low_memory=False)
    # Combine body and subject into one text column
    df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")
    df["label"] = df["label"].astype(int)
    return df


def encode_texts(texts):
    """Tokenize texts for BERT input."""
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
