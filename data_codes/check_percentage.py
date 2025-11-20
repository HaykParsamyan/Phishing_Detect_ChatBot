import pandas as pd
import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # data_codes folder
DATA_PATH = os.path.join(BASE_DIR, "..", "final_data", "all_phishing_master_dataset.csv")

# --- Configuration ---
SAMPLE_FRAC = 0.5  # 50% of dataset

# --- Load dataset ---
print(f"Loading dataset from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH, low_memory=False)

# --- Take 50% sample ---
sample_df = df.sample(frac=SAMPLE_FRAC, random_state=42)
print(f"Sampled {len(sample_df)} rows ({SAMPLE_FRAC*100}% of dataset)")

# --- Check label distribution ---
if 'label' not in sample_df.columns:
    raise ValueError("The dataset does not contain a 'label' column. Update the column name accordingly.")

label_counts = sample_df['label'].value_counts()
label_percent = sample_df['label'].value_counts(normalize=True) * 100

print("\n--- Label Counts ---")
for label, count in label_counts.items():
    print(f"{label}: {count} rows")

print("\n--- Label Percentages ---")
for label, perc in label_percent.items():
    print(f"{label}: {perc:.2f}%")

print("\n✅ Summary Complete ✅")
print(f"Total sample size: {len(sample_df)}")
print(f"Phishing emails: {label_counts.get(1, 0)} ({label_percent.get(1, 0):.2f}%)")
print(f"Legitimate emails: {label_counts.get(0, 0)} ({label_percent.get(0, 0):.2f}%)")
