# model.py (The Training and Prediction Orchestrator)

import pandas as pd
import numpy as np
import xgboost as xgb  # Use native XGBoost API
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import joblib
import os
import threading

# --- ONNX Imports ---
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx
# --------------------

# Import functions and configurations from the new modules
# NOTE: Ensure your features.py and metrics.py files are available
from features import load_and_prepare_dataset, extract_additional_features, MAX_EMAIL_LENGTH, GLOBAL_NUMERIC_COLS
from metrics import evaluate_model

# Global variables
# clf is now an XGBoost Booster object, not an XGBClassifier
clf = None
tfidf_vectorizer = None
scaler = None
training_in_progress = False
MODEL_PATH = "models/phishing_detection_model.pkl"
ONNX_MODEL_PATH = "models/phishing_detector.onnx"


# --- Training and Prediction Core Logic ---

def train_model_sync():
    """Synchronous function to perform the entire ML training pipeline."""
    global clf, tfidf_vectorizer, scaler, training_in_progress

    print("\n--- Starting Model Training ---")
    try:
        # Load and prepare data (CPU-bound)
        df = load_and_prepare_dataset()

        # Data Split: 80% Train, 10% Validation, 10% Test
        print(f"Total rows after merge and cleaning: {len(df)}")
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

        print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

        # ------------------------------------------------------------------------
        # 1. TF-IDF Text Transformation (CPU-bound)
        print("\n⏳ **STATUS: Starting TF-IDF Vectorization and Transformation...**")
        tfidf_vectorizer = TfidfVectorizer(max_features=40000, stop_words='english', ngram_range=(1, 3))
        X_train_text = tfidf_vectorizer.fit_transform(train_df['email_text'].astype(str).str[:MAX_EMAIL_LENGTH])
        X_val_text = tfidf_vectorizer.transform(val_df['email_text'].astype(str).str[:MAX_EMAIL_LENGTH])
        X_test_text = tfidf_vectorizer.transform(test_df['email_text'].astype(str).str[:MAX_EMAIL_LENGTH])
        print("✅ TF-IDF Transformation Complete.")

        # 2. Scaling Numeric Features (CPU-bound)
        print("⏳ **STATUS: Starting Numeric Feature Scaling...**")
        X_train_numeric = train_df[GLOBAL_NUMERIC_COLS].values
        X_val_numeric = val_df[GLOBAL_NUMERIC_COLS].values
        X_test_numeric = test_df[GLOBAL_NUMERIC_COLS].values

        scaler = StandardScaler()
        X_train_numeric = scaler.fit_transform(X_train_numeric)
        X_val_numeric = scaler.transform(X_val_numeric)
        X_test_numeric = scaler.transform(X_test_numeric)
        print("✅ Numeric Feature Scaling Complete.")

        # 3. Combine Features (CPU-bound)
        print("⏳ **STATUS: Combining Text and Numeric Features...**")
        X_train = hstack([X_train_text, X_train_numeric])
        X_val = hstack([X_val_text, X_val_numeric])
        X_test = hstack([X_test_text, X_test_numeric])
        print("✅ Feature Combination Complete.")

        # ------------------------------------------------------------------------
        # 4. Convert to DMatrix for Optimal GPU Transfer (CRITICAL STEP)
        # ------------------------------------------------------------------------
        print("⏳ **STATUS: Converting to XGBoost DMatrix (Optimizing Sparse Data Transfer to GPU)...**")
        dtrain = xgb.DMatrix(X_train, label=train_df['label'])
        dval = xgb.DMatrix(X_val, label=val_df['label'])
        dtest = xgb.DMatrix(X_test, label=test_df['label'])
        print("✅ DMatrix Conversion Complete. Starting GPU Training.")

        # ------------------------------------------------------------------------
        # 5. Model Training (GPU-bound) - Using Native API
        # ------------------------------------------------------------------------

        # Hyperparameters passed as a dictionary for the native API
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'eta': 0.05,  # Learning Rate (was learning_rate in sklearn wrapper)
            'max_depth': 7,
            'random_state': 42,
            # CRUCIAL GPU ACCELERATION SETTINGS
            'tree_method': 'hist',
            'device': 'cuda'
        }

        # The native xgb.train function takes the DMatrix objects
        clf = xgb.train(
            params,
            dtrain,
            num_boost_round=500,  # Corresponds to n_estimators in sklearn wrapper
            evals=[(dval, 'validation')],
            verbose_eval=50
        )
        print("✅ Model trained successfully")

        # ----------------------------------------
        # 6. Saving Model Components
        # ----------------------------------------

        # Save the native XGBoost Booster object (official recommendation)
        clf.save_model("models/phishing_detector_booster.json")
        print(f"Native XGBoost Booster saved to models/phishing_detector_booster.json")

        # Save the preprocessor components using joblib
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(
            {'vectorizer': tfidf_vectorizer, 'scaler': scaler, 'numeric_cols': GLOBAL_NUMERIC_COLS},
            MODEL_PATH, compress=3)
        print(f"Pre-processor components saved as '{MODEL_PATH}'")

        # NOTE: ONNX conversion is more complex for native Booster objects
        # We will skip the ONNX conversion for the native Booster object for simplicity
        # and rely on the JSON/Joblib saving for now.

        # 7. Evaluation
        evaluate_model(dval, val_df['label'], clf, 'Validation', native_api=True)
        evaluate_model(dtest, test_df['label'], clf, 'Test', native_api=True)


    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        training_in_progress = False
        print("--- Model Training Complete ---")


def load_model():
    """Loads a pre-trained model and components."""
    global clf, tfidf_vectorizer, scaler

    # 1. Load Preprocessor Components
    if os.path.exists(MODEL_PATH):
        try:
            print(f"Attempting to load pre-processors from {MODEL_PATH}...")
            data = joblib.load(MODEL_PATH)
            tfidf_vectorizer = data['vectorizer']
            scaler = data['scaler']
            print("✅ Pre-processor components loaded successfully.")
        except Exception as e:
            print(f"Error loading pre-processors: {e}")
            return False
    else:
        return False

    # 2. Load Native XGBoost Booster
    booster_path = "models/phishing_detector_booster.json"
    if os.path.exists(booster_path):
        try:
            print(f"Attempting to load XGBoost Booster from {booster_path}...")
            # Initialize a new Booster object
            clf = xgb.Booster()
            # Load the model from the JSON file
            clf.load_model(booster_path)
            print("✅ XGBoost Booster loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading XGBoost Booster: {e}")
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


def predict_email(text, custom_threshold=0.40):
    """
    Predicts the label for a given email text using the loaded XGBoost Booster.
    """
    if training_in_progress or clf is None or tfidf_vectorizer is None or scaler is None:
        return "Model is not fully initialized (training in progress or failed to load/train).", 0, 0

    try:
        # 1. Feature Extraction (uses CPU components)
        data_row = {'email_text': text, 'subject': '', 'links_count': 0, 'email_length_csv': np.nan,
                    'special_chars_csv': np.nan, 'subject_length_csv': np.nan}
        for col in GLOBAL_NUMERIC_COLS:
            if col not in data_row:
                data_row[col] = 0

        df = pd.DataFrame([data_row])
        df = extract_additional_features(df)

        # 2. Transformation (uses CPU components)
        X_text = tfidf_vectorizer.transform([text[:MAX_EMAIL_LENGTH]])
        X_numeric = scaler.transform(df[GLOBAL_NUMERIC_COLS].values)
        X_combined = hstack([X_text, X_numeric])

        # 3. Prediction using DMatrix (Required for native Booster)
        dpredict = xgb.DMatrix(X_combined)

        # Native Booster predicts raw score by default; use output='prob' for probability
        # pred_proba returns probability for the positive class (1)
        phishing_probability = clf.predict(dpredict, output_margin=False)[0]

        # Apply the custom threshold
        pred = 1 if phishing_probability >= custom_threshold else 0

        # 4. Calculate probabilities (for display)
        phishing_prob = phishing_probability * 100
        safe_prob = (1.0 - phishing_probability) * 100

        return ('phishing' if pred == 1 else 'legitimate', phishing_prob, safe_prob)

    except Exception as e:
        print(f"Prediction error: {e}")
        return f"Prediction error: {e}", 0, 0