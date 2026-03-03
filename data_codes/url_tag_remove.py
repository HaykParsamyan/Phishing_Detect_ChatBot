from pathlib import Path
import pandas as pd

# ================= CONFIG =================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

INPUT_PATH = PROJECT_DIR / "final_data" / "url_plus_email_148k_balanced.csv"
OUTPUT_PATH = PROJECT_DIR / "final_data" / "url_plus_email_148k_cleaned.csv"

TEXT_COL = "body"  # change if needed
# ==========================================

def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"File not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    if TEXT_COL not in df.columns:
        raise ValueError(f"Column '{TEXT_COL}' not found. Columns: {list(df.columns)}")

    print("Rows before:", len(df))

    # Remove only [URL] and [/URL] tags
    df[TEXT_COL] = (
        df[TEXT_COL]
        .astype(str)
        .str.replace(r"\[URL\]", "", regex=True)
        .str.replace(r"\[/URL\]", "", regex=True)
        .str.strip()
    )

    print("Rows after:", len(df))

    df.to_csv(OUTPUT_PATH, index=False)
    print("Saved cleaned dataset to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()