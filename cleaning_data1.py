import re
import sys

FILE_PATH = "data/malicious_phishing_dataset.csv"

# Regex patterns for AWS secrets
aws_access_key_pattern = re.compile(r'AKIA[0-9A-Z]{16}')
aws_secret_key_pattern = re.compile(r'(?<![A-Z0-9])[A-Za-z0-9/+=]{40}(?![A-Z0-9])')

# If you want to add more protections, include:
additional_patterns = [
    re.compile(r'ASIA[0-9A-Z]{16}'),   # Temporary access keys
    re.compile(r'(?i)aws_secret_access_key.*'),
    re.compile(r'(?i)aws_access_key_id.*')
]

def clean_line(line):
    # Replace AWS access keys
    line = aws_access_key_pattern.sub("REDACTED", line)

    # Replace AWS secret keys
    line = aws_secret_key_pattern.sub("REDACTED", line)

    # Replace additional patterns
    for pat in additional_patterns:
        line = pat.sub("REDACTED", line)

    return line


def clean_file(path):
    print(f"Cleaning secrets in: {path}")

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        cleaned = []
        for i, line in enumerate(lines, 1):
            new_line = clean_line(line)
            cleaned.append(new_line)

        with open(path, "w", encoding="utf-8") as f:
            f.writelines(cleaned)

        print("DONE. All AWS keys replaced with 'REDACTED'.")

    except Exception as e:
        print("Error:", e)
        sys.exit(1)


if __name__ == "__main__":
    clean_file(FILE_PATH)
