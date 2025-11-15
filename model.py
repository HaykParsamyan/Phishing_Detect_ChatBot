# model.py (The Training and Prediction Orchestrator)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from scipy.sparse import hstack
import joblib
import os
import threading

# Import functions and configurations from the new modules
from features import load_and_prepare_dataset, extract_additional_features, MAX_EMAIL_LENGTH, GLOBAL_NUMERIC_COLS
from metrics import evaluate_model

# Global variables
clf = None
tfidf_vectorizer = None
scaler = None
training_in_progress = False
MODEL_PATH = "models/phishing_detection_model.pkl"


# --- Training and Prediction Core Logic ---

def train_model_sync():
    """Synchronous function to perform the entire ML training pipeline."""
    global clf, tfidf_vectorizer, scaler, training_in_progress

    print("\n--- Starting Model Training ---")
    try:
        df = load_and_prepare_dataset()

        # Data Split: 80% Train, 10% Validation, 10% Test
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

        print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

        # 1. TF-IDF Text Transformation
        tfidf_vectorizer = TfidfVectorizer(max_features=40000, stop_words='english', ngram_range=(1, 3))
        X_train_text = tfidf_vectorizer.fit_transform(train_df['email_text'].astype(str).str[:MAX_EMAIL_LENGTH])
        X_val_text = tfidf_vectorizer.transform(val_df['email_text'].astype(str).str[:MAX_EMAIL_LENGTH])
        X_test_text = tfidf_vectorizer.transform(test_df['email_text'].astype(str).str[:MAX_EMAIL_LENGTH])

        # 2. Scaling Numeric Features
        X_train_numeric = train_df[GLOBAL_NUMERIC_COLS].values
        X_val_numeric = val_df[GLOBAL_NUMERIC_COLS].values
        X_test_numeric = test_df[GLOBAL_NUMERIC_COLS].values

        scaler = StandardScaler()
        X_train_numeric = scaler.fit_transform(X_train_numeric)
        X_val_numeric = scaler.transform(X_val_numeric)
        X_test_numeric = scaler.transform(X_test_numeric)

        # 3. Combine Features
        X_train = hstack([X_train_text, X_train_numeric])
        X_val = hstack([X_val_text, X_val_numeric])
        X_test = hstack([X_test_text, X_test_numeric])

        # 4. Model Training (XGBoost)
        clf = XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=7, eval_metric='logloss', random_state=42
        )
        clf.fit(X_train, train_df['label'])
        print("✅ Model trained successfully")

        # 5. Evaluation
        evaluate_model(X_val, val_df['label'], clf, 'Validation')
        evaluate_model(X_test, test_df['label'], clf, 'Test')

        # 6. Save model components
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(
            {'model': clf, 'vectorizer': tfidf_vectorizer, 'scaler': scaler, 'numeric_cols': GLOBAL_NUMERIC_COLS},
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
        return

    if not training_in_progress:
        training_in_progress = True
        thread = threading.Thread(target=train_model_sync, daemon=True)
        thread.start()


def predict_email(text, custom_threshold=0.5):
    """
    Predicts the label for a given email text using a custom threshold.
    The default threshold (0.35) is set to favor Recall (detecting more threats).
    """
    if training_in_progress or clf is None or tfidf_vectorizer is None or scaler is None:
        return "Model is not fully initialized (training in progress or failed to load/train).", 0, 0

    try:
        # 1. Prepare data row and Feature Extraction
        data_row = {'email_text': text, 'subject': '', 'links_count': 0, 'email_length_csv': np.nan,
                    'special_chars_csv': np.nan, 'subject_length_csv': np.nan}
        for col in GLOBAL_NUMERIC_COLS:
            if col not in data_row:
                data_row[col] = 0
        df = pd.DataFrame([data_row])
        df = extract_additional_features(df)

        # 2. Transformation
        X_text = tfidf_vectorizer.transform([text[:MAX_EMAIL_LENGTH]])
        X_numeric = scaler.transform(df[GLOBAL_NUMERIC_COLS].values)
        X_combined = hstack([X_text, X_numeric])

        # 3. Prediction and Custom Threshold Application
        proba = clf.predict_proba(X_combined)[0]
        phishing_class_index = list(clf.classes_).index(1)
        phishing_probability = proba[phishing_class_index]

        # Apply the custom threshold to classify (Recall-focused)
        pred = 1 if phishing_probability >= custom_threshold else 0

        # 4. Calculate probabilities (for display)
        safe_prob = proba[list(clf.classes_).index(0)] * 100 if 0 in clf.classes_ else 0
        phishing_prob = phishing_probability * 100

        return ('phishing' if pred == 1 else 'legitimate', phishing_prob, safe_prob)

    except Exception as e:
        print(f"Prediction error: {e}")
        return f"Prediction error: {e}", 0, 0