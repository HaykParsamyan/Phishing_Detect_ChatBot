# model.py (Final, Fully Corrected Code with Path Fix)

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
import os
from scipy.sparse import hstack
import threading
import csv
import sys
import warnings

# Filter DtypeWarning from pandas when reading CSVs with mixed types
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
csv.field_size_limit(2 ** 31 - 1)

# Global variables
clf = None
tfidf_vectorizer = None
scaler = None
training_in_progress = False
MAX_EMAIL_LENGTH = 2000

# --- Dataset Paths (CORRECTED PHISHING FILE PATH) ---
DATASET_PATH = "data/dataset.csv"
PHISHING_DATASET_PATH = "data/Phishing_Email.csv"  # <--- CORRECTED FILENAME (Case Sensitive)
MODEL_PATH = "models/phishing_detection_model.pkl"

# Define global_numeric_cols to store the features used in training for consistency
global_numeric_cols = []

# --- Column Mapping for Unification ---
COLUMN_MAPPING = {
    'Email Text': 'email_text',
    'Email Type': 'label',
    'URL Count': 'links_count',
    'Email Length': 'email_length_csv',
    'Punctuation Count': 'special_chars_csv',
    'Subject Length': 'subject_length_csv',
    'URLs': 'urls_present',

    'Subject': 'subject',

    # New features to add directly
    'Email Count': 'email_count',
    'Keyword Count': 'keyword_count',
    'Misspelled Words Count': 'misspelled_words_count',
    'Subject Keyword Count': 'subject_keyword_count',
}


def extract_additional_features(df):
    """
    Extract engineered features, prioritizing CSV data over calculation where possible.
    Includes robust handling for non-numeric data in CSV columns.
    """

    # 1. Standardize/Calculate basic lengths
    df['email_length'] = df.apply(
        lambda row: row['email_length_csv'] if pd.notnull(row.get('email_length_csv')) else len(str(row['email_text'])),
        axis=1
    )
    df['subject_length'] = df.apply(
        lambda row: row['subject_length_csv'] if pd.notnull(row.get('subject_length_csv')) else len(
            str(row.get('subject', ''))), axis=1
    )

    # 2. Calculated features
    # Ensure links_count is treated as numeric before calculation (Fixes potential conversion error)
    df['links_count'] = pd.to_numeric(df['links_count'], errors='coerce').fillna(0).astype(float)
    df['link_density'] = df['links_count'] / (df['email_length'] + 1)

    # 3. Standardize/Calculate special characters
    df['special_chars'] = df.apply(
        lambda row: row['special_chars_csv'] if pd.notnull(row.get('special_chars_csv')) else len(
            re.findall(r'[!$%^&*()_+|~=`{}\[\]:";\'<>?,./]', str(row['email_text']))), axis=1
    )

    # 4. Retain HTML tags count
    df['html_tags'] = df['email_text'].apply(lambda x: len(re.findall(r'<[^>]+>', str(x).lower())))

    # 5. Fill NA and ROBUST TYPE CONVERSION for new numeric features
    new_numeric_cols = [
        'email_count',
        'keyword_count',
        'misspelled_words_count',
        'subject_keyword_count'
    ]
    for col in new_numeric_cols:
        if col in df.columns:
            # CRITICAL FIX: Convert mixed types to numeric, coercing errors to NaN, then fill
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)

    return df


def load_and_prepare_dataset():
    """Loads, merges, and prepares both datasets."""

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Primary dataset not found at {DATASET_PATH}.")

    df_main = pd.read_csv(DATASET_PATH)
    print(f"Loaded primary rows: {len(df_main)}")

    df = df_main.copy()

    # --- CRITICAL FIX 1: Ensure primary DF text columns are strings ---
    # Solves the "can only concatenate str (not "int") to str" error if it happens on df_main
    df['email_text'] = df['email_text'].astype(str)
    if 'subject' in df.columns:
        df['subject'] = df['subject'].astype(str)

    if os.path.exists(PHISHING_DATASET_PATH):
        df_phish = pd.read_csv(PHISHING_DATASET_PATH)
        print(f"Loaded rich phishing rows: {len(df_phish)}")

        # Rename columns in the rich dataset
        df_phish.rename(columns=COLUMN_MAPPING, inplace=True)

        # --- CRITICAL FIX 2: Ensure rich DF text columns are strings ---
        # Solves the "can only concatenate str (not "int") to str" error if it happens on df_phish
        df_phish['email_text'] = df_phish['email_text'].astype(str)
        if 'subject' in df_phish.columns:
            df_phish['subject'] = df_phish['subject'].astype(str)

        # Canonical list of columns expected in the final merged DataFrame
        canonical_cols = [
            'email_text', 'subject', 'label', 'links_count', 'email_length_csv',
            'special_chars_csv', 'subject_length_csv', 'email_count',
            'keyword_count', 'misspelled_words_count', 'subject_keyword_count'
        ]

        # Ensure both DFs have all canonical columns (fill missing in one dataset with NaN)
        for col in canonical_cols:
            if col not in df_main.columns:
                df_main[col] = np.nan
            if col not in df_phish.columns:
                df_phish[col] = np.nan

        # Concatenate Datasets
        df = pd.concat([df_main[canonical_cols], df_phish[canonical_cols]], ignore_index=True)
        df.drop_duplicates(subset=['email_text'], inplace=True)

    # Final Preprocessing
    df['label'] = df['label'].astype(str).str.lower().apply(lambda x: 1 if 'phishing' in x else 0)

    # Feature Engineering (handles numeric cleansing internally)
    df = extract_additional_features(df)

    # Final cleanup before returning
    df.dropna(subset=['email_text', 'label'], inplace=True)

    print(f"Total rows after merge and cleaning: {len(df)}")
    return df


def train_model_sync():
    """Synchronous function to perform the training process."""
    global clf, tfidf_vectorizer, scaler, training_in_progress, global_numeric_cols

    print("\n--- Starting Model Training ---")
    try:
        df = load_and_prepare_dataset()

        # 80/10/10 split
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

        print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

        # TF-IDF (Enhanced with Bigrams)
        tfidf_vectorizer = TfidfVectorizer(
            max_features=20000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        X_train_text = tfidf_vectorizer.fit_transform(train_df['email_text'].astype(str).str[:MAX_EMAIL_LENGTH])
        X_val_text = tfidf_vectorizer.transform(val_df['email_text'].astype(str).str[:MAX_EMAIL_LENGTH])
        X_test_text = tfidf_vectorizer.transform(test_df['email_text'].astype(str).str[:MAX_EMAIL_LENGTH])

        # Define all required numeric features
        required_numeric_cols = [
            'email_length', 'subject_length', 'link_density', 'special_chars',
            'html_tags', 'email_count', 'keyword_count', 'misspelled_words_count',
            'subject_keyword_count'
        ]

        global_numeric_cols = [col for col in required_numeric_cols if col in df.columns]
        print(f"Using numeric features: {global_numeric_cols}")

        X_train_numeric = train_df[global_numeric_cols].values
        X_val_numeric = val_df[global_numeric_cols].values
        X_test_numeric = test_df[global_numeric_cols].values

        # Scale numeric
        scaler = StandardScaler()
        X_train_numeric = scaler.fit_transform(X_train_numeric)
        X_val_numeric = scaler.transform(X_val_numeric)
        X_test_numeric = scaler.transform(X_test_numeric)

        # Combine sparse TF-IDF with numeric
        X_train = hstack([X_train_text, X_train_numeric])
        X_val = hstack([X_val_text, X_val_numeric])
        X_test = hstack([X_test_text, X_test_numeric])

        # Classifier (Adjusted Hyperparameters)
        clf = GradientBoostingClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=7,
            random_state=42,
            subsample=0.8,
            max_features='sqrt'
        )
        clf.fit(X_train, train_df['label'])
        print("✅ Model trained successfully")

        # Evaluate
        for name, X_set, y_set in [('Validation', X_val, val_df['label']), ('Test', X_test, test_df['label'])]:
            y_pred = clf.predict(X_set)
            print(f"\n{name} Accuracy: {accuracy_score(y_set, y_pred):.4f}")
            print(f"{name} F1 Score: {f1_score(y_set, y_pred):.4f}")
            print(
                f"{name} Classification Report:\n{classification_report(y_set, y_pred, target_names=['legitimate', 'phishing'])}")

        # Save model
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(
            {'model': clf, 'vectorizer': tfidf_vectorizer, 'scaler': scaler, 'numeric_cols': global_numeric_cols},
            MODEL_PATH, compress=3)
        print(f"Model saved as '{MODEL_PATH}'")

    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        training_in_progress = False
        print("--- Model Training Complete ---")


def load_model():
    """Loads a pre-trained model and components."""
    global clf, tfidf_vectorizer, scaler, global_numeric_cols
    if os.path.exists(MODEL_PATH):
        try:
            print(f"Attempting to load model from {MODEL_PATH}...")
            data = joblib.load(MODEL_PATH)
            clf = data['model']
            tfidf_vectorizer = data['vectorizer']
            scaler = data['scaler']
            global_numeric_cols = data.get('numeric_cols', [])
            print("✅ Pre-trained model loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    return False


def start_background_training():
    """Initiates model training in a non-blocking background thread."""
    global training_in_progress
    if load_model():
        print("Skipping background training: Pre-trained model found.")
        return

    if not training_in_progress:
        training_in_progress = True
        thread = threading.Thread(target=train_model_sync, daemon=True)
        thread.start()


def predict_email(text):
    """Predicts the label for a given email text."""
    if training_in_progress or clf is None or tfidf_vectorizer is None or scaler is None:
        return "Model is not fully initialized (training in progress or failed to load/train).", 0, 0

    try:
        # Prepare dummy data row for feature calculation
        data_row = {'email_text': text, 'subject': '', 'links_count': 0, 'email_length_csv': np.nan,
                    'special_chars_csv': np.nan, 'subject_length_csv': np.nan}

        # Add placeholders for new features
        for col in global_numeric_cols:
            if col not in ['email_length', 'subject_length', 'link_density', 'special_chars', 'html_tags']:
                data_row[col] = 0

        df = pd.DataFrame([data_row])

        # Feature Extraction
        df = extract_additional_features(df)

        # Text Transformation
        X_text = tfidf_vectorizer.transform([text[:MAX_EMAIL_LENGTH]])

        # Numeric Scaling
        X_numeric = scaler.transform(df[global_numeric_cols].values)

        # Combination
        X_combined = hstack([X_text, X_numeric])

        # Prediction
        pred = clf.predict(X_combined)[0]
        proba = clf.predict_proba(X_combined)[0]

        # Calculate probabilities (returning floats for bot.py to round)
        safe_prob = proba[list(clf.classes_).index(0)] * 100 if 0 in clf.classes_ else 0
        phishing_prob = proba[list(clf.classes_).index(1)] * 100 if 1 in clf.classes_ else 0

        return ('phishing' if pred == 1 else 'legitimate', phishing_prob, safe_prob)

    except Exception as e:
        print(f"Prediction error: {e}")
        return f"Prediction error: {e}", 0, 0