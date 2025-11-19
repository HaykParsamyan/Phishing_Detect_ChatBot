import pandas as pd
import os

# --- Configuration ---
MERGED_DATA_PATH = "final_data/all_phishing_master_dataset.csv"
MAX_EMAIL_LENGTH = 10000  # max chars for TF-IDF

# --- Feature Extraction Functions ---

def load_and_prepare_dataset() -> pd.DataFrame:
    """
    Loads the merged phishing dataset and ensures required columns exist.
    """
    if not os.path.exists(MERGED_DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {MERGED_DATA_PATH}")

    df = pd.read_csv(MERGED_DATA_PATH, encoding='latin-1', low_memory=False)

    # Ensure columns exist
    for col in ['body', 'subject', 'label']:
        if col not in df.columns:
            raise ValueError(f"Dataset must contain column '{col}'")

    # Rename for model compatibility
    df = df.rename(columns={'body': 'email_text'})

    df['email_text'] = df['email_text'].astype(str)
    df['subject'] = df['subject'].fillna('').astype(str)
    df['label'] = df['label'].astype(int)

    return df


def extract_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal feature extraction: only ensures text columns are strings.
    """
    df['email_text'] = df['email_text'].astype(str)
    df['subject'] = df['subject'].astype(str)
    return df
