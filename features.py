# features.py (Refactored for merged dataset)

import pandas as pd
import numpy as np
import re
import os

# --- CONFIGURATION ---
DATA_PATH = "final_data/all_phishing_master_dataset.csv"
MAX_EMAIL_LENGTH = 10000  # Max characters for TF-IDF to process

# Optional numeric features (can be removed if you want pure text-based model)
GLOBAL_NUMERIC_COLS = [
    'has_html_tags',
    'contains_urgent_words',
    'link_count',
    'digits_count',
    'subject_length',
    'body_length',
    'url_to_body_ratio',
    'suspicious_tld',
    'non_standard_chars_ratio'
]

URGENT_KEYWORDS = [
    'urgent', 'immediate', 'action required', 'alert', 'suspended',
    'verify', 'reset password', 'account locked', 'transaction',
    'confirmation', 'security', 'expired', 'click here'
]

# --- FEATURE EXTRACTION ---
def extract_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract numeric features for phishing detection (optional)."""
    df['full_text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')
    df['body_text'] = df['body'].fillna('')

    # HTML Tags
    df['has_html_tags'] = df['body_text'].apply(lambda x: 1 if bool(re.search(r'</?\w+>', x)) else 0)

    # Urgent Words
    df['contains_urgent_words'] = df['full_text'].apply(
        lambda x: 1 if any(word in x.lower() for word in URGENT_KEYWORDS) else 0
    )

    # Link Count
    df['link_count'] = df['body_text'].apply(
        lambda x: len(re.findall(r'https?://[^\s]+', x))
    )

    # Digits Count
    df['digits_count'] = df['body_text'].apply(lambda x: len(re.findall(r'\d', x)))

    # Length features
    df['subject_length'] = df['subject'].apply(lambda x: len(str(x)))
    df['body_length'] = df['body_text'].apply(lambda x: len(x))

    # URL to Body ratio
    df['url_to_body_ratio'] = np.where(df['body_length'] > 0, df['link_count'] / df['body_length'], 0)

    # Suspicious TLDs
    SUSPICIOUS_TLDS = ['.xyz', '.click', '.info', '.biz', '.top', '.loan']
    def suspicious_tld(text):
        domains = re.findall(r'https?://([a-zA-Z0-9.-]+)', text)
        for domain in domains:
            if any(domain.endswith(tld) for tld in SUSPICIOUS_TLDS):
                return 1
        return 0
    df['suspicious_tld'] = df['body_text'].apply(suspicious_tld)

    # Non-standard characters ratio
    def non_standard_ratio(text):
        if not text:
            return 0
        return len(re.findall(r'[^\w\s\.\,\-\?\!\:\;]', text)) / len(text)
    df['non_standard_chars_ratio'] = df['body_text'].apply(non_standard_ratio)

    return df

# --- LOAD AND PREPARE DATASET ---
def load_and_prepare_dataset() -> pd.DataFrame:
    """Load merged dataset and extract numeric features (optional)."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Merged dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if 'body' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset must have 'body' and 'label' columns")
    if 'subject' not in df.columns:
        df['subject'] = ''  # Add empty subject if missing

    df = extract_additional_features(df)

    # Combine subject + body into single text column
    df['email_text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')
    df['label'] = df['label'].astype(int)

    return df
