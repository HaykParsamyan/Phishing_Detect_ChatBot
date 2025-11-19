# model.py (Refactored for single merged dataset)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from scipy.sparse import hstack
import joblib
import os
import threading

# Global variables
clf = None
tfidf_vectorizer = None
training_in_progress = False
MODEL_PATH = "models/phishing_detection_model.pkl"
MAX_EMAIL_LENGTH = 10000  # adjust if needed

# --- Dataset Loading ---
def load_merged_dataset(path="final_data/all_phishing_master_dataset.csv"):
    """Load merged dataset and combine subject + body for text analysis."""
    df = pd.read_csv(path)
    # Combine subject and body, handling nulls
    df['email_text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')
    df = df[['email_text', 'label']].copy()
    return df

# --- Training ---
def train_model_sync():
    global clf, tfidf_vectorizer, training_in_progress
    print("\n--- Starting Model Training ---")
    try:
        df = load_merged_dataset()

        # Split data: 80% train, 10% val, 10% test
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
        print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

        # TF-IDF Vectorization
        tfidf_vectorizer = TfidfVectorizer(max_features=40000, stop_words='english', ngram_range=(1,3))
        X_train = tfidf_vectorizer.fit_transform(train_df['email_text'].astype(str).str[:MAX_EMAIL_LENGTH])
        X_val = tfidf_vectorizer.transform(val_df['email_text'].astype(str).str[:MAX_EMAIL_LENGTH])
        X_test = tfidf_vectorizer.transform(test_df['email_text'].astype(str).str[:MAX_EMAIL_LENGTH])

        # Model training
        clf = XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=7, eval_metric='logloss', random_state=42
        )
        clf.fit(X_train, train_df['label'])
        print("✅ Model trained successfully")

        # Simple evaluation
        val_acc = clf.score(X_val, val_df['label'])
        test_acc = clf.score(X_test, test_df['label'])
        print(f"Validation Accuracy: {val_acc:.4f}, Test Accuracy: {test_acc:.4f}")

        # Save model
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump({'model': clf, 'vectorizer': tfidf_vectorizer}, MODEL_PATH, compress=3)
        print(f"Model saved as '{MODEL_PATH}'")

    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        training_in_progress = False
        print("--- Model Training Complete ---")

# --- Load Model ---
def load_model():
    global clf, tfidf_vectorizer
    if os.path.exists(MODEL_PATH):
        try:
            print(f"Loading model from {MODEL_PATH}...")
            data = joblib.load(MODEL_PATH)
            clf = data['model']
            tfidf_vectorizer = data['vectorizer']
            print("✅ Model loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    return False

# --- Background Training ---
def start_background_training():
    global training_in_progress
    if load_model():
        print("Skipping training: model already exists.")
        return
    if not training_in_progress:
        training_in_progress = True
        threading.Thread(target=train_model_sync, daemon=True).start()

# --- Prediction ---
def predict_email(text, custom_threshold=0.40):
    """
    Predict phishing probability for a single email text.
    """
    if training_in_progress or clf is None or tfidf_vectorizer is None:
        return "Model is not ready (training or loading in progress)", 0, 0
    try:
        combined_text = text if text else ""
        X = tfidf_vectorizer.transform([combined_text[:MAX_EMAIL_LENGTH]])
        proba = clf.predict_proba(X)[0]
        phishing_prob = proba[list(clf.classes_).index(1)] * 100
        safe_prob = proba[list(clf.classes_).index(0)] * 100
        pred = 1 if phishing_prob >= custom_threshold*100 else 0
        return ('phishing' if pred==1 else 'legitimate', phishing_prob, safe_prob)
    except Exception as e:
        print(f"Prediction error: {e}")
        return f"Prediction error: {e}", 0, 0
