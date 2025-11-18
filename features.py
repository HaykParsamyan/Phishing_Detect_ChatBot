# features.py

import pandas as pd
import numpy as np
import re
import os
import csv
import warnings
import Levenshtein  # Required for Typosquatting detection

# Ignore pandas DtypeWarning which can occur during concatenation of mixed types
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
# Increase CSV field size limit for large text fields
csv.field_size_limit(2 ** 31 - 1)

# --- CONFIGURATION ---
DATASET_PATH = "data/dataset.csv"
PHISHING_DATASET_PATH = "data/Phishing_Email.csv"
CEAS_DATASET_PATH = "data/CEAS_08.csv"
PHISHING_URLS_DATASET_PATH = "data/phishing_site_urls.csv"
PHISHING_URLS2_DATASET_PATH = "data/phishing_site_urls2.csv"
PHISHING_EMAIL2_DATASET_PATH = "data/phishing_email_2.csv"
MALICIOUS_PHISHING_DATASET_PATH = "data/malicious_phishing_dataset.csv"
MAX_EMAIL_LENGTH = 2000

URGENCY_KEYWORDS = [
    "immediate", "urgent", "required", "action now", "expire", "suspend",
    "warning", "security alert", "violates", "failed", "unauthorized",
    "click here", "don't miss", "last chance", "act now", "reply within"
]

# High-value targets for Typosquatting checks
TARGET_BRANDS = [
    "google", "microsoft", "apple", "amazon", "facebook",
    "paypal", "netflix", "linkedin", "twitter", "instagram"
]

# Common URL shortener domains (short URLs often hide the destination)
SHORTENER_DOMAINS = [
    "bit.ly", "goo.gl", "t.co", "tinyurl.com", "ow.ly", "is.gd", "buff.ly", "cutt.ly"
]

# Column Mappings for merging various datasets
COLUMN_MAPPING = {
    'Email Text': 'email_text', 'Email Type': 'label', 'URL Count': 'links_count',
    'Email Length': 'email_length_csv', 'Punctuation Count': 'special_chars_csv',
    'Subject Length': 'subject_length_csv', 'Subject': 'subject',
}
CEAS_COLUMN_MAPPING = {'body': 'email_text', 'subject': 'subject', 'label': 'label'}
URL_COLUMN_MAPPING = {'URL': 'email_text', 'Label': 'label'}
URL_COLUMN_MAPPING2 = {'url': 'email_text', 'type': 'label'}
PHISHING_EMAIL2_MAPPING = {'text_combined': 'email_text', 'label': 'label'}
MALICIOUS_PHISHING_MAPPING = {'url': 'email_text', 'type': 'label'}

# Define the set of features the model will use
GLOBAL_NUMERIC_COLS = [
    'email_length', 'subject_length', 'link_density', 'special_chars',
    'html_tags', 'urgency_score', 'link_anomaly_score',
    'non_latin_chars',  # Homograph detection (Cyrillic, Greek, etc.)
    'typosquatting_score',  # Typosquatting detection
    'short_url_suspicion',  # Shortened URLs detection
    'incomplete_tld_score'  # Incomplete TLDs detection
]


# --- FEATURE EXTRACTION FUNCTIONS ---

# Pre-compiled Regex patterns (improves efficiency when used repeatedly)
IP_PATTERN = re.compile(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')
URL_PATTERN = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')
HTML_TAG_PATTERN = re.compile(r'<(table|div|img|p|a|script|iframe)', re.IGNORECASE)
PUNCTUATION_PATTERN = re.compile(r'[!$%^&*()_+|~=`{}\[\]:";\'<>?,./]')
INCOMPLETE_TLD_PATTERN = re.compile(r'(\.co|\.or|\.n\w|\.in|\.ru)[^\w/]', re.IGNORECASE)


def is_url_suspicious(url):
    """Calculates a suspicious score based on URL structure."""
    if not isinstance(url, str):
        return 0

    score = 0
    url_lower = url.lower()

    # IP Address Check (using pre-compiled IP_PATTERN)
    if IP_PATTERN.search(url_lower):
        score += 2
    # Use of the @ symbol (Credential obfuscation)
    if '@' in url_lower:
        score += 3

    return min(score, 5)


def check_non_latin(text):
    """
    Counts characters that are letters that belong to scripts often used in homograph attacks
    (Cyrillic, Greek, etc.). This is the **HARDENED** check.
    """
    if not isinstance(text, str):
        return 0

    count = 0

    # Define ranges for common homograph scripts (Cyrillic, Greek, etc.)
    cyrillic_range = (0x0400, 0x04FF)
    greek_range = (0x0370, 0x03FF)

    for char in text:
        char_ord = ord(char)

        if char.isalpha():
            if (char_ord >= cyrillic_range[0] and char_ord <= cyrillic_range[1]):
                count += 1
            elif (char_ord >= greek_range[0] and char_ord <= greek_range[1]):
                count += 1
            # Add a catch-all for anything outside the basic ASCII/Latin range (U+0080 and above)
            elif char_ord > 127:
                count += 1

    return count


def extract_root_domain(url):
    """
    Extracts the most likely root domain name from a full URL.
    """
    if not isinstance(url, str):
        return ""

    url = re.sub(r'^https?://', '', url, flags=re.IGNORECASE)
    url = re.sub(r'/.*$', '', url)
    url = url.lower()
    url = url.replace('www.', '').replace('mail.', '').replace('login.', '')

    parts = url.split('.')
    if len(parts) >= 2:
        return parts[-2]
    elif len(parts) == 1:
        return parts[0]
    else:
        return ""


def calculate_typosquatting_score(text):
    """
    Calculates a Typosquatting Suspicion Score (higher = more suspicious).
    Score = max(0, 5 - Min Levenshtein Distance).
    """
    if not isinstance(text, str):
        return 0

    urls = URL_PATTERN.findall(str(text))

    if not urls:
        return 0

    min_distance = 100

    for url in urls:
        root_domain = extract_root_domain(url)

        if not root_domain:
            continue

        for brand in TARGET_BRANDS:
            distance = Levenshtein.distance(root_domain, brand)
            min_distance = min(min_distance, distance)

    # Invert the score: distance 1 -> suspicion 4; distance 2 -> suspicion 3
    typosquatting_suspicion = max(0, 5 - min_distance)

    return typosquatting_suspicion


def check_shortened_url(text):
    """
    Checks if the email contains any common URL shortener links.
    """
    if not isinstance(text, str):
        return 0

    # Check for any shortener domain in the text/URL
    for domain in SHORTENER_DOMAINS:
        if domain in text:
            return 1  # 1 means a shortened URL was found (suspicious)

    return 0


def check_incomplete_tld(text):
    """
    Checks for suspicious URLs where the TLD is obviously incomplete
    (e.g., domain.co or domain.or followed by a non-word character).
    """
    if not isinstance(text, str):
        return 0

    # Use pre-compiled INCOMPLETE_TLD_PATTERN
    if INCOMPLETE_TLD_PATTERN.search(text):
        return 1

    return 0


def extract_additional_features(df):
    """Calculates all engineered numeric features."""

    # Fallback calculations for length and special characters if CSV data is missing
    df['email_length'] = df.apply(
        lambda row: row['email_length_csv'] if pd.notnull(row.get('email_length_csv')) else len(str(row['email_text'])),
        axis=1)
    df['subject_length'] = df.apply(
        lambda row: row['subject_length_csv'] if pd.notnull(row.get('subject_length_csv')) else len(
            str(row.get('subject', ''))), axis=1)

    df['links_count'] = pd.to_numeric(df['links_count'], errors='coerce').fillna(0).astype(float)
    df['link_density'] = df['links_count'] / (df['email_length'] + 1)

    # Special Chars (using pre-compiled PUNCTUATION_PATTERN)
    df['special_chars'] = df.apply(
        lambda row: row['special_chars_csv'] if pd.notnull(row.get('special_chars_csv')) else len(
            PUNCTUATION_PATTERN.findall(str(row['email_text']))), axis=1)

    # HTML Tags (using pre-compiled HTML_TAG_PATTERN)
    df['html_tags'] = df['email_text'].apply(lambda x: len(HTML_TAG_PATTERN.findall(str(x))))

    # Urgency Score
    urgency_pattern = re.compile('|'.join(re.escape(k) for k in URGENCY_KEYWORDS), re.IGNORECASE)
    df['urgency_score'] = df['email_text'].apply(lambda x: len(re.findall(urgency_pattern, str(x))))

    # Link Anomaly Score (IP address, @ symbol in URL)
    df['link_anomaly_score'] = df['email_text'].apply(lambda x:
                                                      max([is_url_suspicious(url) for url in
                                                           URL_PATTERN.findall(str(x))] or [0])
                                                      if pd.notnull(x) else 0)

    # All four advanced features are calculated here:
    df['non_latin_chars'] = df['email_text'].apply(check_non_latin)
    df['typosquatting_score'] = df['email_text'].apply(calculate_typosquatting_score)
    df['short_url_suspicion'] = df['email_text'].apply(check_shortened_url)
    df['incomplete_tld_score'] = df['email_text'].apply(check_incomplete_tld)

    return df


# --- DATA LOADING AND PREPARATION ---

def load_and_prepare_dataset():
    """Loads, merges, and prepares all seven datasets."""

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Primary dataset not found at {DATASET_PATH}.")

    canonical_cols = list(COLUMN_MAPPING.values())

    def load_df(path, mapping, name):
        """Helper function to load a single CSV with robust encoding handling."""
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

    # --- Loading all 7 datasets ---
    df_main = load_df(DATASET_PATH, COLUMN_MAPPING, "dataset.csv")
    df_phish = load_df(PHISHING_DATASET_PATH, COLUMN_MAPPING, "Phishing_Email.csv")
    df_ceas = load_df(CEAS_DATASET_PATH, CEAS_COLUMN_MAPPING, "CEAS_08.csv")
    df_urls = load_df(PHISHING_URLS_DATASET_PATH, URL_COLUMN_MAPPING, "phishing_site_urls.csv")
    df_urls2 = load_df(PHISHING_URLS2_DATASET_PATH, URL_COLUMN_MAPPING2, "phishing_site_urls2.csv")
    df_email2 = load_df(PHISHING_EMAIL2_DATASET_PATH, PHISHING_EMAIL2_MAPPING, "phishing_email_2.csv")
    df_malicious = load_df(MALICIOUS_PHISHING_DATASET_PATH, MALICIOUS_PHISHING_MAPPING,
                           "malicious_phishing_dataset.csv")

    dataframes = [df_main, df_phish, df_ceas, df_urls, df_urls2, df_email2, df_malicious]

    # Concatenate all datasets
    df = pd.concat([d.reindex(columns=canonical_cols) for d in dataframes if not d.empty], ignore_index=True)

    df.drop_duplicates(subset=['email_text'], inplace=True)

    # Map all string/text labels to 0 (Legitimate) or 1 (Malicious/Phishing)
    df['label'] = df['label'].astype(str).str.lower()
    df['label'] = df['label'].apply(
        lambda x: 1
        if ('phishing' in x) or ('spam' in x) or ('bad' in x) or (x == '1') or ('defacament' in x)
        else 0
    )

    df = extract_additional_features(df)
    df.dropna(subset=['email_text', 'label'], inplace=True)

    print(f"Total rows after merge and cleaning: {len(df)}")
    return df