from pathlib import Path
import pandas as pd

# ================= CONFIG =================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

FILE_1 = PROJECT_DIR / "cleaned_data" / "CEAS_08_minimal_processed.csv"
FILE_2 = PROJECT_DIR / "cleaned_data" / "dataset.csv"

OUTPUT_FILE = PROJECT_DIR / "cleaned_data" / "merged_dataset.csv"
# ==========================================

def main():
    if not FILE_1.exists():
        raise FileNotFoundError(f"Missing file: {FILE_1}")
    if not FILE_2.exists():
        raise FileNotFoundError(f"Missing file: {FILE_2}")

    print("Loading datasets...")
    df1 = pd.read_csv(FILE_1)
    df2 = pd.read_csv(FILE_2)

    print("File 1 shape:", df1.shape)
    print("File 2 shape:", df2.shape)

    # Check if columns match
    if list(df1.columns) != list(df2.columns):
        print("\n⚠ WARNING: Column mismatch detected.")
        print("File 1 columns:", list(df1.columns))
        print("File 2 columns:", list(df2.columns))
        raise ValueError("Columns do not match. Fix column names before merging.")

    print("\nMerging...")
    merged = pd.concat([df1, df2], ignore_index=True)

    print("Before duplicate removal:", merged.shape)

    merged = merged.drop_duplicates()

    print("After duplicate removal:", merged.shape)

    merged.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved merged dataset to:")
    print(OUTPUT_FILE)

if __name__ == "__main__":
    main()