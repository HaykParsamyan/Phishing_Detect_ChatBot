# model.py (Modified)

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
import os
from scipy.sparse import hstack
import threading
import csv
import sys

# Allow huge email fields
csv.field_size_limit(2 ** 31 - 1)

# Global variables
clf = None
tfidf_vectorizer = None
scaler = None
training_in_progress = False
MAX_EMAIL_LENGTH = 2000  # truncate long emails

DATASET_PATH = "data/dataset.csv"  # your main dataset
MODEL_PATH = "models/phishing_detection_model.pkl"  # Model save path


# --- Feature Engineering ---

def extract_additional_features(df):
    """Extract engineered features"""
    df['email_length'] = df['email_text'].apply(lambda x: len(str(x)))
    df['subject_length'] = df['subject'].apply(lambda x: len(str(x)))
    df['link_density'] = df['links_count'].apply(lambda x: float(x) if pd.notnull(x) else 0) / (df['email_length'] + 1)
    df['special_chars'] = df['email_text'].apply(
        lambda x: len(re.findall(r'[!$%^&*()_+|~=`{}\[\]:";\'<>?,./]', str(x))))
    df['html_tags'] = df['email_text'].apply(lambda x: len(re.findall(r'<[^>]+>', str(x).lower())))
    return df


def load_and_prepare_dataset():
    """Loads and preprocesses the dataset."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Please create the file.")

    df = pd.read_csv(DATASET_PATH)
    # Ensure label is binary
    df['label'] = df['label'].apply(lambda x: 1 if str(x).lower() == 'phishing' else 0)
    df = extract_additional_features(df)
    df.dropna(subset=['email_text', 'label'], inplace=True)
    print(f"Loaded total rows: {len(df)}")
    return df


# --- Training and Prediction ---

def train_model_sync():
    """Synchronous function to perform the training process."""
    global clf, tfidf_vectorizer, scaler, training_in_progress

    print("\n--- Starting Model Training ---")
    try:
        df = load_and_prepare_dataset()

        # 80/10/10 split
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

        print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

        # TF-IDF
        tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X_train_text = tfidf_vectorizer.fit_transform(train_df['email_text'].astype(str).str[:MAX_EMAIL_LENGTH])
        X_val_text = tfidf_vectorizer.transform(val_df['email_text'].astype(str).str[:MAX_EMAIL_LENGTH])
        X_test_text = tfidf_vectorizer.transform(test_df['email_text'].astype(str).str[:MAX_EMAIL_LENGTH])

        # Numeric features
        numeric_cols = ['email_length', 'subject_length', 'link_density', 'special_chars', 'html_tags']
        X_train_numeric = train_df[numeric_cols].values
        X_val_numeric = val_df[numeric_cols].values
        X_test_numeric = test_df[numeric_cols].values

        # Scale numeric
        scaler = StandardScaler()
        X_train_numeric = scaler.fit_transform(X_train_numeric)
        X_val_numeric = scaler.transform(X_val_numeric)
        X_test_numeric = scaler.transform(X_test_numeric)

        # Combine sparse TF-IDF with numeric
        X_train = hstack([X_train_text, X_train_numeric])
        X_val = hstack([X_val_text, X_val_numeric])
        X_test = hstack([X_test_text, X_test_numeric])

        # Classifier
        clf = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
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
        joblib.dump({'model': clf, 'vectorizer': tfidf_vectorizer, 'scaler': scaler},
                    MODEL_PATH, compress=3)
        print(f"Model saved as '{MODEL_PATH}'")

    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        training_in_progress = False
        print("--- Model Training Complete ---")


def load_model():
    """Loads a pre-trained model and components."""
    global clf, tfidf_vectorizer, scaler
    if os.path.exists(MODEL_PATH):
        try:
            print(f"Attempting to load model from {MODEL_PATH}...")
            data = joblib.load(MODEL_PATH)
            clf = data['model']
            tfidf_vectorizer = data['vectorizer']
            scaler = data['scaler']
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
        return  # Skip training if model is found

    if not training_in_progress:
        training_in_progress = True
        thread = threading.Thread(target=train_model_sync, daemon=True)
        thread.start()
    else:
        print("Training is already in progress.")


def predict_email(text):
    """Predicts the label for a given email text."""
    # Check if the model is ready
    if training_in_progress or clf is None or tfidf_vectorizer is None or scaler is None:
        return "Model is not fully initialized (training in progress or failed to load/train).", 0, 0

    try:
        # 1. Feature Extraction
        data = [{'email_text': text, 'subject': '', 'links_count': 0}]  # Dummy row for feature engineering
        df = pd.DataFrame(data)
        df = extract_additional_features(df)

        # 2. Text Transformation
        X_text = tfidf_vectorizer.transform([text[:MAX_EMAIL_LENGTH]])

        # 3. Numeric Scaling
        numeric_cols = ['email_length', 'subject_length', 'link_density', 'special_chars', 'html_tags']
        X_numeric = scaler.transform(df[numeric_cols].values)

        # 4. Combination
        X_combined = hstack([X_text, X_numeric])

        # 5. Prediction
        pred = clf.predict(X_combined)[0]
        proba = clf.predict_proba(X_combined)[0]

        # Calculate probabilities
        safe_prob = round(proba[list(clf.classes_).index(0)] * 100, 2) if 0 in clf.classes_ else 0
        phishing_prob = round(proba[list(clf.classes_).index(1)] * 100, 2) if 1 in clf.classes_ else 0

        return ('phishing' if pred == 1 else 'legitimate', phishing_prob, safe_prob)

    except Exception as e:
        print(f"Prediction error: {e}")
        return f"Prediction error: {e}", 0, 0