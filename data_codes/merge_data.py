import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NEW_DATA = os.path.join(BASE_DIR, "cleaned_data", "dataset.csv")
FINAL_DATA = os.path.join(BASE_DIR, "final_data", "merged_email_url_dataset.csv")

def main():
    print("Loading cleaned dataset...")
    new_df = pd.read_csv(NEW_DATA, low_memory=False)

    print("Loading final dataset...")
    final_df = pd.read_csv(FINAL_DATA, low_memory=False)

    # Required structure
    required_cols = ["body", "subject", "label"]

    for col in required_cols:
        if col not in new_df.columns:
            raise ValueError(f"New dataset missing column: {col}")
        if col not in final_df.columns:
            raise ValueError(f"Final dataset missing column: {col}")

    # Keep only correct columns
    new_df = new_df[required_cols].copy()
    final_df = final_df[required_cols].copy()

    # Clean text
    for col in ["body", "subject"]:
        new_df[col] = new_df[col].fillna("").astype(str).str.strip()
        final_df[col] = final_df[col].fillna("").astype(str).str.strip()

    # Clean labels
    new_df["label"] = pd.to_numeric(new_df["label"], errors="coerce")
    final_df["label"] = pd.to_numeric(final_df["label"], errors="coerce")

    new_df = new_df.dropna(subset=["label"])
    final_df = final_df.dropna(subset=["label"])

    new_df["label"] = new_df["label"].astype(int)
    final_df["label"] = final_df["label"].astype(int)

    print("Before merge size:", len(final_df))

    # Merge
    combined = pd.concat([final_df, new_df], ignore_index=True)

    # Remove duplicates (based on text + label)
    combined["dedup_key"] = (
        combined["subject"].astype(str) + " " +
        combined["body"].astype(str)
    ).str.strip()

    combined = combined.drop_duplicates(subset=["dedup_key", "label"])
    combined = combined.drop(columns=["dedup_key"])

    print("After merge size:", len(combined))
    print("Label distribution:\n", combined["label"].value_counts())

    # Save
    combined.to_csv(FINAL_DATA, index=False)
    print("Saved updated final dataset to:", FINAL_DATA)


if __name__ == "__main__":
    main()