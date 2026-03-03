from pathlib import Path
import pandas as pd

# ================= CONFIG =================
FILE_PATH = Path("../final_data/url_plus_email_148k_balanced.csv")
LABEL_COL = "label"
TEXT_COL = "body"
URL_TAG = "[URL]"
# ==========================================

def main():
    if not FILE_PATH.exists():
        raise FileNotFoundError(f"File not found: {FILE_PATH}")

    df = pd.read_csv(FILE_PATH)

    if LABEL_COL not in df.columns or TEXT_COL not in df.columns:
        raise ValueError(f"Columns found: {list(df.columns)}")

    # Detect URL vs Email using tag
    df["data_type"] = df[TEXT_COL].apply(
        lambda x: "url" if isinstance(x, str) and URL_TAG in x else "email"
    )

    total = len(df)

    # Overall label distribution
    label_counts = df[LABEL_COL].value_counts().sort_index()

    # Overall type distribution
    type_counts = df["data_type"].value_counts()

    print("===== TOTAL DATA =====")
    print("Total samples:", total)
    print()

    print("===== LABEL DISTRIBUTION =====")
    print("Legitimate (0):", label_counts.get(0, 0))
    print("Phishing   (1):", label_counts.get(1, 0))
    print()

    print("===== TYPE DISTRIBUTION =====")
    print("URL:", type_counts.get("url", 0))
    print("Email:", type_counts.get("email", 0))
    print()

    print("===== LABEL + TYPE BREAKDOWN =====")
    breakdown = df.groupby([LABEL_COL, "data_type"]).size()
    print(breakdown)

    print()
    print("===== PERCENTAGES =====")

    legit = label_counts.get(0, 0)
    phish = label_counts.get(1, 0)

    if total > 0:
        print(f"Legit %: {(legit/total)*100:.2f}%")
        print(f"Phish %: {(phish/total)*100:.2f}%")

    for label in [0, 1]:
        subset = df[df[LABEL_COL] == label]
        if len(subset) > 0:
            url_count = (subset["data_type"] == "url").sum()
            email_count = (subset["data_type"] == "email").sum()

            print()
            print(f"Label {label} details:")
            print(f"  URL:   {url_count} ({url_count/len(subset)*100:.2f}%)")
            print(f"  Email: {email_count} ({email_count/len(subset)*100:.2f}%)")

if __name__ == "__main__":
    main()