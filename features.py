# features.py

import pandas as pd
import numpy as np
import re
import os
import csv
import warnings

# Filter DtypeWarning from pandas when reading CSVs with mixed types
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
csv.field_size_limit(2 ** 31 - 1)

# --- CONFIGURATION ---
DATASET_PATH = "data/dataset.csv"
PHISHING_DATASET_PATH = "data/Phishing_Email.csv"
CEAS_DATASET_PATH = "data/CEAS_08.csv"
MAX_EMAIL_LENGTH = 2000

# Urgency keywords for feature engineering
URGENCY_KEYWORDS = [
    "immediate", "urgent", "required", "action now", "expire", "suspend",
    "warning", "security alert", "violates", "failed", "unauthorized",
    "click here", "don't miss", "last chance", "act now", "reply within"
]

# Column Mappings
COLUMN_MAPPING = {
    'Email Text': 'email_text', 'Email Type': 'label', 'URL Count': 'links_count',
    'Email Length': 'email_length_csv', 'Punctuation Count': 'special_chars_csv',
    'Subject Length': 'subject_length_csv', 'Subject': 'subject',
    'Email Count': 'email_count', 'Keyword Count': 'keyword_count',
    'Misspelled Words Count': 'misspelled_words_count', 'Subject Keyword Count': 'subject_keyword_count',
}

CEAS_COLUMN_MAPPING = {
    'body': 'email_text',
    'subject': 'subject',
    'label': 'label',
}

# Define the set of features the model will use (for consistency)
GLOBAL_NUMERIC_COLS = [
    'email_length', 'subject_length', 'link_density', 'special_chars',
    'html_tags', 'email_count', 'keyword_count', 'misspelled_words_count',
    'subject_keyword_count', 'urgency_score'
]


def extract_additional_features(df):
    """Calculates all engineered numeric features."""

    df['email_length'] = df.apply(
        lambda row: row['email_length_csv'] if pd.notnull(row.get('email_length_csv')) else len(str(row['email_text'])),
        axis=1)
    df['subject_length'] = df.apply(
        lambda row: row['subject_length_csv'] if pd.notnull(row.get('subject_length_csv')) else len(
            str(row.get('subject', ''))), axis=1)

    df['links_count'] = pd.to_numeric(df['links_count'], errors='coerce').fillna(0).astype(float)
    df['link_density'] = df['links_count'] / (df['email_length'] + 1)

    df['special_chars'] = df.apply(
        lambda row: row['special_chars_csv'] if pd.notnull(row.get('special_chars_csv')) else len(
            re.findall(r'[!$%^&*()_+|~=`{}\[\]:";\'<>?,./]', str(row['email_text']))), axis=1)

    tag_pattern = re.compile(r'<(table|div|img|p|a|script|iframe)', re.IGNORECASE)
    df['html_tags'] = df['email_text'].apply(lambda x: len(re.findall(tag_pattern, str(x))))

    new_numeric_cols = ['email_count', 'keyword_count', 'misspelled_words_count', 'subject_keyword_count']
    for col in new_numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)

    urgency_pattern = re.compile('|'.join(re.escape(k) for k in URGENCY_KEYWORDS), re.IGNORECASE)
    df['urgency_score'] = df['email_text'].apply(lambda x: len(re.findall(urgency_pattern, str(x))))

    return df


def load_and_prepare_dataset():
    """Loads, merges, and prepares all three datasets."""

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Primary dataset not found at {DATASET_PATH}.")

    dataframes = []
    canonical_cols = list(COLUMN_MAPPING.values())

    def load_df(path, mapping, name):
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                print(f"Loaded rows ({name}): {len(df)}")
                df.rename(columns=mapping, inplace=True)
                df['email_text'] = df['email_text'].astype(str)
                if 'subject' in df.columns:
                    df['subject'] = df['subject'].astype(str)
                return df.dropna(subset=['email_text', 'label'])
            except Exception as e:
                print(f"Warning: Could not load {name}. Error: {e}")
        return pd.DataFrame()

    df_main = load_df(DATASET_PATH, COLUMN_MAPPING, "dataset.csv")
    df_phish = load_df(PHISHING_DATASET_PATH, COLUMN_MAPPING, "Phishing_Email.csv")
    df_ceas = load_df(CEAS_DATASET_PATH, CEAS_COLUMN_MAPPING, "CEAS_08.csv")

    dataframes = [df_main, df_phish, df_ceas]

    df = pd.concat([d.reindex(columns=canonical_cols) for d in dataframes], ignore_index=True)

    df.drop_duplicates(subset=['email_text'], inplace=True)
    df['label'] = df['label'].astype(str).str.lower().apply(lambda x: 1 if 'phishing' in x else 0)
    df = extract_additional_features(df)
    df.dropna(subset=['email_text', 'label'], inplace=True)

    print(f"Total rows after merge and cleaning: {len(df)}")
    return df