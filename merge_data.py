import pandas as pd
import os
import sys

# --- CONFIGURATION ---
# Assumes the merged file is saved here from the previous step
FINAL_MERGED_FILE = os.path.join('final_data', 'all_phishing_master_dataset.csv')


def check_phishing_percentage(file_path):
    """
    Reads the final dataset and calculates the percentage of phishing (1) 
    and legitimate (0) samples.
    """
    print(f"--- Checking data balance for: {file_path} ---")

    # 1. Read the final merged CSV file
    try:
        df = pd.read_csv(file_path, low_memory=False, encoding='utf-8')
    except FileNotFoundError:
        print(f"🛑 Error: Final merged file not found at {file_path}.")
        print("Please ensure you ran 'merge_all_cleaned_data.py' successfully.")
        sys.exit(1)
    except Exception as e:
        print(f"🛑 Error reading file: {e}")
        sys.exit(1)

    total_rows = len(df)
    if total_rows == 0:
        print("⚠️ Warning: Dataset is empty. Cannot calculate percentage.")
        return

    # 2. Count the labels
    # We ensure the label column is treated as categorical/integer
    label_counts = df['label'].value_counts(dropna=False)

    # Define counts for required categories
    phishing_count = label_counts.get(1, 0)
    legitimate_count = label_counts.get(0, 0)

    # Check for any unmapped labels (NaN/others)
    unmapped_count = total_rows - (phishing_count + legitimate_count)

    # 3. Calculate Percentages
    phishing_percent = (phishing_count / total_rows) * 100
    legitimate_percent = (legitimate_count / total_rows) * 100

    # 4. Display Results
    print(f"\nTotal Samples in Dataset: {total_rows}")
    print("\n## ⚖️ Phishing/Legitimate Data Balance")
    print("-------------------------------------------------------")

    print(f"**Phishing (Label 1):** {phishing_count:10,} rows | **{phishing_percent:.2f}%**")
    print(f"**Legitimate (Label 0):** {legitimate_count:8,} rows | **{legitimate_percent:.2f}%**")

    if unmapped_count > 0:
        unmapped_percent = (unmapped_count / total_rows) * 100
        print(f"\n**⚠️ Unmapped/Null Labels:** {unmapped_count:8,} rows | **{unmapped_percent:.2f}%**")
        print("   (These rows should be reviewed if the percentage is significant.)")

    # 5. Provide a conclusion
    if 45 <= phishing_percent <= 55:
        print("\nConclusion: The dataset is **well-balanced**. Great job!")
    elif phishing_percent < 45 or phishing_percent > 55:
        print("\nConclusion: The dataset shows some **class imbalance**.")
        print(
            "   Consider techniques like oversampling the minority class or using class weights during model training.")


if __name__ == '__main__':
    check_phishing_percentage(FINAL_MERGED_FILE)