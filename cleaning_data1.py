import pandas as pd
import os

# Get folder where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the CSV in the 'data' folder
csv_path = os.path.join(script_dir, "data", "phishing_emails.csv")

# Load CSV
df = pd.read_csv(csv_path)

# Drop the unwanted columns
columns_to_drop = ["Sender Email", "URLs", "Sender"]
df = df.drop(columns=columns_to_drop)

# Save cleaned CSV in the same 'data' folder
output_path = os.path.join(script_dir, "data", "phishing_emails_cleaned.csv")
df.to_csv(output_path, index=False)

print(f"Columns deleted and cleaned CSV saved at: {output_path}")
