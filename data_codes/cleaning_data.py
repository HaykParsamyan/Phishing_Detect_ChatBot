import os
import pandas as pd

# ============================
# PATH SETUP
# ============================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

INPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "final_data",
    "all_phishing_master_dataset_ai_filled.csv"
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "final_data",
    "all_phishing_master_dataset_final.csv"
)

# ============================
# LOAD DATA
# ============================

df = pd.read_csv(INPUT_PATH)

print("Columns before:")
print(df.columns)

# ============================
# DROP COLUMN
# ============================

if "source_file" in df.columns:
    df = df.drop(columns=["source_file"])
    print("\n'source_file' column removed.")
else:
    print("\n'source_file' column not found.")

# ============================
# SAVE CLEAN DATASET
# ============================

df.to_csv(OUTPUT_PATH, index=False)

print("\nSaved cleaned dataset to:")
print(OUTPUT_PATH)