from pathlib import Path
import pandas as pd

# ================= CONFIG =================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

FILE_PATH = PROJECT_DIR / "cleaned_data" / "dataset.csv"  # change if needed
LABEL_COL = "label"
# ==========================================

def main():
    if not FILE_PATH.exists():
        raise FileNotFoundError(f"File not found: {FILE_PATH}")

    df = pd.read_csv(FILE_PATH)

    if LABEL_COL not in df.columns:
        raise ValueError(f"'label' column not found. Columns: {list(df.columns)}")

    total = len(df)
    null_count = df[LABEL_COL].isna().sum()
    not_null_count = total - null_count

    print("===== LABEL NULL CHECK =====")
    print("Total rows:", total)
    print("Null labels:", null_count)
    print("Non-null labels:", not_null_count)

    if total > 0:
        print(f"Null percentage: {(null_count/total)*100:.4f}%")

    print("\n===== FULL LABEL DISTRIBUTION (including NaN) =====")
    print(df[LABEL_COL].value_counts(dropna=False))

if __name__ == "__main__":
    main()