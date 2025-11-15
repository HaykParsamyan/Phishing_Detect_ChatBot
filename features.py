# features.py

import pandas as pd
import numpy as np
import re
import os
import csv
import warnings

warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
csv.field_size_limit(2 ** 31 - 1)

# --- CONFIGURATION ---
DATASET_PATH = "data/dataset.csv"
PHISHING_DATASET_PATH = "data/Phishing_Email.csv"
CEAS_DATASET_PATH = "data/CEAS_08.csv"
PHISHING_URLS_DATASET_PATH = "data/phishing_site_urls.csv"
PHISHING_URLS2_DATASET_PATH = "data/phishing_site_urls2.csv"  # <--- NEW FILE PATH
MAX_EMAIL_LENGTH = 2000

URGENCY_KEYWORDS = [
    "immediate", "urgent", "required", "action now", "expire", "suspend",
    "warning", "security alert", "violates", "failed", "unauthorized",
    "click here", "don't miss", "last chance", "act now", "reply within"
]

# Column Mappings for Email Datasets
COLUMN_MAPPING = {
    'Email Text': 'email_text', 'Email Type': 'label', 'URL Count': 'links_count',
    'Email Length': 'email_length_csv', 'Punctuation Count': 'special_chars_csv',
    'Subject Length': 'subject_length_csv', 'Subject': 'subject',
    'Email Count': 'email_count', 'Keyword Count': 'keyword_count',
    'Misspelled Words Count': 'misspelled_words_count', 'Subject Keyword Count': 'subject_keyword_count',
}

# Mapping for the CEAS Dataset
CEAS_COLUMN_MAPPING = {
    'body': 'email_text',
    'subject': 'subject',
    'label': 'label',
}

# Mapping for the Phishing URL Dataset 1 (URL, Label)
URL_COLUMN_MAPPING = {
    'URL': 'email_text',
    'Label': 'label',
}

# Mapping for the Phishing URL Dataset 2 (url, type)
URL_COLUMN_MAPPING2 = {
    'url': 'email_text',
    'type': 'label',
}

# Define the set of features the model will use
GLOBAL_NUMERIC_COLS = [
    'email_length', 'subject_length', 'link_density', 'special_chars',
    'html_tags', 'email_count', 'keyword_count', 'misspelled_words_count',
    'subject_keyword_count', 'urgency_score', 'link_anomaly_score'
]


def is_url_suspicious(url):
    """Calculates a suspicious score based on URL structure."""
    if not isinstance(url, str):
        return 0

    score = 0
    url_lower = url.lower()

    # 1. IP Address Check
    if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url_lower):
        score += 2

    # 2. URL Shortener Check
    if any(s in url_lower for s in ['bit.ly', 'tinyurl', 'goo.gl', 't.co']):
        score += 1

        # 3. Use of the @ symbol
    if '@' in url_lower:
        score += 3

    return min(score, 5)


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

    # Fill NaNs for existing numeric columns from CSVs
    new_numeric_cols = ['email_count', 'keyword_count', 'misspelled_words_count', 'subject_keyword_count']
    for col in new_numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)

    # Urgency Score
    urgency_pattern = re.compile('|'.join(re.escape(k) for k in URGENCY_KEYWORDS), re.IGNORECASE)
    df['urgency_score'] = df['email_text'].apply(lambda x: len(re.findall(urgency_pattern, str(x))))

    # Link Anomaly Score
    df['link_anomaly_score'] = df['email_text'].apply(lambda x:
                                                      max([is_url_suspicious(url) for url in
                                                           re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+',
                                                                      str(x))] or [0])
                                                      if pd.notnull(x) else 0)

    return df


def load_and_prepare_dataset():
    """Loads, merges, and prepares all five datasets."""

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Primary dataset not found at {DATASET_PATH}.")

    canonical_cols = list(COLUMN_MAPPING.values())

    def load_df(path, mapping, name):
        if os.path.exists(path):
            print(f"Attempting to load {name} from {path}...")
            encodings_to_try = ['utf-8', 'latin-1', 'cp1252']

            for encoding in encodings_to_try:
                try:
                    df = pd.read_csv(
                        path,
                        encoding=encoding,
                        engine='python',
                        on_bad_lines='skip'
                    )

                    print(f"Successfully loaded {name} with encoding: {encoding}. Rows: {len(df)}")

                    df.rename(columns=mapping, inplace=True)

                    df['email_text'] = df['email_text'].astype(str)
                    if 'subject' in df.columns:
                        df['subject'] = df['subject'].astype(str)

                    return df.dropna(subset=['email_text', 'label'])

                except (UnicodeDecodeError, KeyError) as e:
                    if isinstance(e, KeyError):
                        print(f"Warning: Column mapping failed for {name}. Check column names in the CSV file.")
                        break
                    print(f"Failed to load {name} with {encoding} due to Unicode error. Trying next...")
                except Exception as e:
                    print(f"Warning: Could not load {name}. Unexpected Error: {e}")
                    break

        return pd.DataFrame()

    # --- Loading all 5 datasets ---
    df_main = load_df(DATASET_PATH, COLUMN_MAPPING, "dataset.csv")
    df_phish = load_df(PHISHING_DATASET_PATH, COLUMN_MAPPING, "Phishing_Email.csv")
    df_ceas = load_df(CEAS_DATASET_PATH, CEAS_COLUMN_MAPPING, "CEAS_08.csv")
    df_urls = load_df(PHISHING_URLS_DATASET_PATH, URL_COLUMN_MAPPING, "phishing_site_urls.csv")
    df_urls2 = load_df(PHISHING_URLS2_DATASET_PATH, URL_COLUMN_MAPPING2, "phishing_site_urls2.csv")  # <--- NEW LOAD

    dataframes = [df_main, df_phish, df_ceas, df_urls, df_urls2]  # <--- NEW DATAFRAME ADDED

    # Concatenate all datasets
    df = pd.concat([d.reindex(columns=canonical_cols) for d in dataframes if not d.empty], ignore_index=True)

    df.drop_duplicates(subset=['email_text'], inplace=True)

    # Map all negative indicators (phishing, spam, bad, legitimate) to the positive class (1 or 0)
    df['label'] = df['label'].astype(str).str.lower().apply(
        lambda x: 1 if ('phishing' in x) or ('spam' in x) or ('bad' in x) or (x == '1') else 0)

    df = extract_additional_features(df)
    df.dropna(subset=['email_text', 'label'], inplace=True)

    print(f"Total rows after merge and cleaning: {len(df)}")
    return df