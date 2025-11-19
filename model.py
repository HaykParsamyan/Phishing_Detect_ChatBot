import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from scipy.sparse import hstack
import joblib
import os
import threading

from features import load_and_prepare_dataset, extract_additional_features, MAX_EMAIL_LENGTH

# --- Global variables ---
clf = None
tfidf_vectorizer = None
training_in_progress = False
MODEL_PATH = "models/phishing_detection_model_full.pkl"

# --- Training function ---

def train_model_sync():
    global clf, tfidf_vectorizer, training_in_progress

    print("\n--- Starting Full Model Training ---")
    training_in_progress = True
    try:
        df = load_and_prepare_dataset()
        df = extract_additional_features(df)

        print(f"⚡ Full dataset: {len(df)} rows")

        # Split data: 80% train, 10% val, 10% test
        train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        # TF-IDF
        tfidf_vectorizer = TfidfVectorizer(max_features=20000, stop_words='english', ngram_range=(1, 3))
        X_train_text = tfidf_vectorizer.fit_transform(train_df['email_text'].str[:MAX_EMAIL_LENGTH])
        X_val_text = tfidf_vectorizer.transform(val_df['email_text'].str[:MAX_EMAIL_LENGTH])
        X_test_text = tfidf_vectorizer.transform(test_df['email_text'].str[:MAX_EMAIL_LENGTH])

        # Only text features
        X_train = X_train_text
        X_val = X_val_text
        X_test = X_test_text

        # XGBoost
        clf = XGBClassifier(
            n_estimators=200,  # full training trees
            learning_rate=0.05,
            max_depth=7,
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42
        )
        clf.fit(X_train, train_df['label'])
        print("✅ Full model trained successfully")

        # Optional evaluation
        val_pred = clf.predict(X_val)
        test_pred = clf.predict(X_test)
        print(f"Validation sample size: {len(val_pred)}, Test sample size: {len(test_pred)}")

        # Save model
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump({'model': clf, 'vectorizer': tfidf_vectorizer}, MODEL_PATH, compress=3)
        print(f"Model saved as '{MODEL_PATH}'")

    except Exception as e:
        print(f"Error during full training: {e}")
    finally:
        training_in_progress = False
        print("--- Full Training Complete ---")


def load_model():
    global clf, tfidf_vectorizer
    if os.path.exists(MODEL_PATH):
        try:
            data = joblib.load(MODEL_PATH)
            clf = data['model']
            tfidf_vectorizer = data['vectorizer']
            print("✅ Full model loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading full model: {e}")
            return False
    return False


def start_background_training():
    global training_in_progress
    if load_model():
        print("Skipping full training: pre-trained model found.")
        return
    if not training_in_progress:
        thread = threading.Thread(target=train_model_sync, daemon=True)
        thread.start()


def predict_email(text, custom_threshold=0.4):
    if training_in_progress or clf is None or tfidf_vectorizer is None:
        return "Model not ready", 0, 0
    try:
        X_text = tfidf_vectorizer.transform([text[:MAX_EMAIL_LENGTH]])
        proba = clf.predict_proba(X_text)[0]
        phishing_prob = proba[list(clf.classes_).index(1)] * 100
        safe_prob = proba[list(clf.classes_).index(0)] * 100
        pred = 1 if phishing_prob >= custom_threshold * 100 else 0
        return ('phishing' if pred == 1 else 'legitimate', phishing_prob, safe_prob)
    except Exception as e:
        return f"Prediction error: {e}", 0, 0
