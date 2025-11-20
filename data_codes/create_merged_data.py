import pandas as pd
import os
import glob
import sys

# --- CONFIGURATION ---
CLEANED_DATA_DIR = '../cleaned_data'
FINAL_MERGED_FILE = os.path.join('../final_data', 'all_phishing_master_dataset.csv')

# --- CORE COLUMNS FOR ALIGNMENT ---
# All files must have these three columns
REQUIRED_COLUMNS = ['body', 'label', 'subject']


def merge_all_datasets():
    """
    Reads all CSV files from the cleaned_data directory, aligns their columns,
    merges them into a single DataFrame, handles duplicates, and saves the final result.
    """
    print("--- Starting Final Data Merge ---")

    # 1. Find all cleaned CSV files
    search_path = os.path.join(CLEANED_DATA_DIR, '*.csv')
    file_list = glob.glob(search_path)

    if not file_list:
        print(f"🛑 Error: No CSV files found in the directory: {CLEANED_DATA_DIR}")
        print("Please ensure your previous processing scripts ran successfully.")
        sys.exit(1)

    print(f"Found {len(file_list)} files to merge:")
    for f in file_list:
        print(f"  - {os.path.basename(f)}")

    all_dataframes = []

    # 2. Load and Prepare Each File
    for file_path in file_list:
        file_name = os.path.basename(file_path)

        try:
            # Read the file
            df = pd.read_csv(file_path, low_memory=False, encoding='utf-8')

            # Ensure the required columns exist, adding them with NaN if missing
            for col in REQUIRED_COLUMNS:
                if col not in df.columns:
                    df[col] = pd.NA

            # Standardize label column to the correct integer type
            df['label'] = pd.to_numeric(df['label'], errors='coerce').astype('Int64')

            # Ensure body is a string
            df['body'] = df['body'].astype(str)

            # Add a source column for tracking (optional, but helpful)
            df['source_file'] = file_name

            all_dataframes.append(df)
            print(f"  Loaded {len(df)} rows from {file_name}")

        except Exception as e:
            print(f"🛑 Error loading/preparing {file_name}: {e}. Skipping this file.")

    if not all_dataframes:
        print("🛑 No DataFrames were successfully loaded. Merge aborted.")
        return

    # 3. Concatenate all DataFrames
    # pd.concat handles the alignment of columns automatically, filling missing ones with NaN
    df_merged = pd.concat(all_dataframes, ignore_index=True)
    initial_rows = len(df_merged)

    # 4. Final Cleanup: Drop Duplicates and Nulls

    # Drop rows where the 'body' (email/url text) is duplicated or null
    df_merged.dropna(subset=['body', 'label'], inplace=True)
    df_merged.drop_duplicates(subset=['body'], keep='first', inplace=True)

    final_rows = len(df_merged)

    print(f"\n--- Merge Summary ---")
    print(f"Initial combined rows: {initial_rows}")
    print(f"Rows dropped (Duplicates/Null Body/Label): {initial_rows - final_rows}")
    print(f"Final rows for training: {final_rows}")

    # 5. Save the Final Master Dataset

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(FINAL_MERGED_FILE), exist_ok=True)

    # Select the core columns first for easy access
    core_cols = REQUIRED_COLUMNS + [col for col in df_merged.columns if col not in REQUIRED_COLUMNS]
    df_merged = df_merged[core_cols]

    df_merged.to_csv(FINAL_MERGED_FILE, index=False, encoding='utf-8')

    print(f"\n✅ Merge Complete! Master dataset saved to: {FINAL_MERGED_FILE}")


if __name__ == '__main__':
    merge_all_datasets()