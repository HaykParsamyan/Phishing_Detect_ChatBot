import os
import pandas as pd

# =========================
# PATH SETUP
# =========================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "final_data",
    "all_phishing_master_dataset_final.csv"
)

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(DATA_PATH)

print("\n========== DATASET OVERVIEW ==========")
print("Total rows:", len(df))
print("Columns:", list(df.columns))


# =========================
# CHECK FUNCTION
# =========================

def analyze_column(column_name):
    if column_name not in df.columns:
        print(f"\nColumn '{column_name}' NOT FOUND.")
        return

    total = len(df)

    null_count = df[column_name].isna().sum()
    empty_count = (df[column_name].astype(str).str.strip() == "").sum()

    null_percent = (null_count / total) * 100
    empty_percent = (empty_count / total) * 100

    print(f"\n--- {column_name} ---")
    print(f"Null values: {null_count} ({null_percent:.2f}%)")
    print(f"Empty strings: {empty_count} ({empty_percent:.2f}%)")


# =========================
# RUN CHECKS
# =========================

analyze_column("body")
analyze_column("subject")
analyze_column("label")

# =========================
# LABEL DISTRIBUTION
# =========================

if "label" in df.columns:
    print("\n========== LABEL DISTRIBUTION ==========")
    print(df["label"].value_counts(dropna=False))