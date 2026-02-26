import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "final_data", "merged_email_url_dataset_ai.csv")

def main():
    df = pd.read_csv(DATA_PATH, low_memory=False)

    if "label" not in df.columns:
        raise ValueError("Dataset does not contain 'label' column")

    # Convert safely
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    counts = df["label"].value_counts().sort_index()

    phishing = counts.get(1, 0)
    legitimate = counts.get(0, 0)

    total = len(df)

    print("\n===== LABEL DISTRIBUTION =====")
    print(f"Total samples:      {total}")
    print(f"Legitimate (0):     {legitimate}")
    print(f"Phishing   (1):     {phishing}")
    print("\nPercentages:")
    print(f"Legitimate: {legitimate / total * 100:.2f}%")
    print(f"Phishing:   {phishing / total * 100:.2f}%")

if __name__ == "__main__":
    main()