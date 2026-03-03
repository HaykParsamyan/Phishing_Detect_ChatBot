from pathlib import Path
import pandas as pd

# ================= CONFIG =================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

FILE_PATH = PROJECT_DIR / "cleaned_data" / "merged_dataset.csv"   # change if needed
OUTPUT_PATH = PROJECT_DIR / "cleaned_data" / "merged_dataset_no_null_labels.csv"

LABEL_COL = "label"
# ==========================================

def main():
    if not FILE_PATH.exists():
        raise FileNotFoundError(f"File not found: {FILE_PATH}")

    df = pd.read_csv(FILE_PATH)

    if LABEL_COL not in df.columns:
        raise ValueError(f"'label' column not found. Columns: {list(df.columns)}")

    print("Before cleaning:")
    print("Total rows:", len(df))
    print(df[LABEL_COL].value_counts(dropna=False))

    # Convert to numeric safely (invalid → NaN)
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")

    # Drop rows where label is NaN
    before = len(df)
    df = df.dropna(subset=[LABEL_COL]).copy()
    dropped_nan = before - len(df)

    # Keep only 0 or 1
    before2 = len(df)
    df = df[df[LABEL_COL].isin([0, 1])].copy()
    dropped_invalid = before2 - len(df)

    # Convert to integer
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    print("\nRemoved rows:")
    print("Null/NaN labels removed:", dropped_nan)
    print("Invalid labels removed:", dropped_invalid)

    print("\nAfter cleaning:")
    print("Total rows:", len(df))
    print(df[LABEL_COL].value_counts())
    print("dtype:", df[LABEL_COL].dtype)

    df.to_csv(OUTPUT_PATH, index=False)
    print("\nSaved cleaned dataset to:")
    print(OUTPUT_PATH)

if __name__ == "__main__":
    main()