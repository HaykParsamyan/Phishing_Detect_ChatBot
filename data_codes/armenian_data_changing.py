from pathlib import Path
import pandas as pd

# ================= CONFIG =================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

INPUT_PATH = PROJECT_DIR / "cleaned_data" / "dataset_tragmanutyun.csv"
OUTPUT_PATH = PROJECT_DIR / "cleaned_data" / "dataset_tragmanutyun_cleaned.csv"

KEEP_COLUMNS = ["label", "subject", "email_text"]
# ==========================================

def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"File not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    print("Original columns:", list(df.columns))

    # Check required columns
    for col in KEEP_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Keep only required columns
    df = df[KEEP_COLUMNS].copy()

    # Rename email_text -> body
    df = df.rename(columns={"email_text": "body"})

    # Clean label column
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    # Map labels
    label_map = {
        "phishing": 1,
        "legitimate": 0
    }

    df["label"] = df["label"].map(label_map)

    # Drop rows where label not phishing/legitimate
    before = len(df)
    df = df.dropna(subset=["label"]).copy()
    dropped = before - len(df)

    df["label"] = df["label"].astype(int)

    print("\nRows removed (invalid labels):", dropped)
    print("\nFinal label distribution:")
    print(df["label"].value_counts())

    print("\nFinal columns:", list(df.columns))
    print("Final shape:", df.shape)

    df.to_csv(OUTPUT_PATH, index=False)
    print("\nSaved cleaned dataset to:")
    print(OUTPUT_PATH)

if __name__ == "__main__":
    main()