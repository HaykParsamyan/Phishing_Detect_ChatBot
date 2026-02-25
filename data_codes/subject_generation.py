import os
import pandas as pd
import requests
from tqdm import tqdm

# ===============================
# CONFIG
# ===============================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"
MAX_BODY_CHARS = 1200
SAVE_EVERY = 50

# ===============================
# PATH SETUP
# ===============================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

CSV_PATH = os.path.join(
    PROJECT_ROOT,
    "final_data",
    "all_phishing_master_dataset.csv"
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "final_data",
    "all_phishing_master_dataset_ai_filled.csv"
)

# ===============================
# LOAD DATA
# ===============================

df = pd.read_csv(CSV_PATH)

missing_mask = df["subject"].isna() | (df["subject"].astype(str).str.strip() == "")
missing_indices = df[missing_mask].index.tolist()

print("Total rows:", len(df))
print("Missing subjects:", len(missing_indices))

# ===============================
# CLEAN BODY
# ===============================

def clean_body(text):
    if pd.isna(text):
        return ""
    text = str(text).replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    return text[:MAX_BODY_CHARS]

# ===============================
# GENERATE SUBJECT USING OLLAMA
# ===============================

def generate_subject(body_text):
    prompt = f"""
Generate a realistic email subject line (max 10 words).
Do not invent new facts.
Keep tone consistent with the body.

Email Body:
{body_text}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"].strip()

# ===============================
# PROCESS LOOP
# ===============================

processed = 0

for idx in tqdm(missing_indices):
    try:
        body = clean_body(df.at[idx, "body"])

        if not body:
            df.at[idx, "subject"] = "No Subject"
            continue

        subject = generate_subject(body)
        df.at[idx, "subject"] = subject

        processed += 1

        if processed % SAVE_EVERY == 0:
            df.to_csv(OUTPUT_PATH, index=False)
            print(f"\nSaved progress at {processed}")

    except Exception as e:
        print(f"\nError at row {idx}: {e}")
        continue

df.to_csv(OUTPUT_PATH, index=False)

print("\nFinished.")
print("Saved to:", OUTPUT_PATH)