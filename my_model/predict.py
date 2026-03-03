import re
import torch
from urllib.parse import urlparse

from my_model import model_manager
from my_model.config import DEVICE, MAX_LEN

# ----------------------------
# 1) Domain allowlist (edit this)
# ----------------------------
TRUSTED_ROOTS = {
    "google.com",
    "microsoft.com",
    "microsoftonline.com",
    "live.com",
    "office.com",
    "apple.com",
    "icloud.com",
    "paypal.com",
    "amazon.com",
    "github.com",
    "telegram.org",
}

TRUSTED_EXACT = {
    "accounts.google.com",
    "mail.google.com",
    "login.microsoftonline.com",
    "outlook.live.com",
}

URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "is.gd", "cutt.ly", "rebrand.ly", "shorturl.at"
}

SUSPICIOUS_TLDS = {".zip", ".top", ".xyz", ".click", ".icu", ".cfd", ".rest", ".cam", ".mom", ".work"}

SUSPICIOUS_PATH_WORDS = {
    "login", "signin", "sign-in", "verify", "verification", "password", "update",
    "secure", "unlock", "confirm", "account", "billing", "invoice", "payment"
}

# Thresholds (tune if needed)
MODEL_PHISH_THRESHOLD = 0.50     # normal model threshold
MODEL_URL_ASSIST = 0.15          # if URL looks risky and model is even slightly suspicious
URL_RISK_FORCE = 4               # URL-only or URL-present: force phishing if risk >= this
URL_RISK_STRONG = 2              # medium risk


# ----------------------------
# 2) URL extraction + normalization
# ----------------------------
def _normalize_url(raw: str) -> str:
    raw = (raw or "").strip().strip(".,)];\"'")
    if not raw:
        return ""
    low = raw.lower()

    # fix common user typos: https;// -> https://
    low = low.replace("https;//", "https://").replace("http;//", "http://")

    # if it's a bare domain like google.com or google.com/path
    if re.match(r"^[a-z0-9][a-z0-9.-]+\.[a-z]{2,}(/|$)", low) and not low.startswith(("http://", "https://")):
        low = "https://" + low
    return low


def extract_urls(text: str):
    t = (text or "").strip()

    # 1) explicit links
    urls = re.findall(r"(https?://[^\s]+|www\.[^\s]+)", t, flags=re.IGNORECASE)

    # 2) bare domains (safe-ish pattern)
    urls += re.findall(r"\b([a-z0-9][a-z0-9.-]+\.[a-z]{2,}(?:/[^\s]*)?)\b", t, flags=re.IGNORECASE)

    out = []
    for u in urls:
        nu = _normalize_url(u)
        if nu and nu not in out:
            out.append(nu)
    return out


def get_domain(url: str) -> str:
    u = _normalize_url(url)
    if not u:
        return ""
    p = urlparse(u)
    host = (p.netloc or "").lower()
    if not host:
        # fallback for malformed urls
        host = (p.path.split("/")[0] if p.path else "").lower()
    host = host.split(":")[0]
    # strip leading www.
    if host.startswith("www."):
        host = host[4:]
    return host


def _root_domain(domain: str) -> str:
    """
    Cheap root extraction (not public suffix accurate, but good enough for allowlist).
    Example: mail.google.com -> google.com
    """
    parts = domain.split(".")
    if len(parts) < 2:
        return domain
    return ".".join(parts[-2:])


def is_trusted_domain(domain: str) -> bool:
    if not domain:
        return False
    if domain in TRUSTED_EXACT:
        return True
    root = _root_domain(domain)
    return root in TRUSTED_ROOTS


def _is_only_url_message(text: str, urls: list[str]) -> bool:
    """
    True if user basically sent only one URL/domain (no extra text).
    """
    if len(urls) != 1:
        return False
    raw = (text or "").strip().lower()
    u = urls[0].lower()
    d = get_domain(u)

    # accept: "google.com", "https://google.com", "google.com/path"
    return raw == u or raw == d or raw.startswith(d + "/")


# ----------------------------
# 3) URL risk scoring (rules)
# ----------------------------
def url_risk_score(url: str) -> int:
    u = _normalize_url(url)
    d = get_domain(u)

    # trusted domains: do NOT mark risky just because path has "login"
    if is_trusted_domain(d):
        return 0

    score = 0

    # shorteners hide destination
    if d in URL_SHORTENERS:
        score += 3

    # punycode look-alike
    if "xn--" in d:
        score += 3

    # IP-based URL
    if re.search(r"https?://\d{1,3}(\.\d{1,3}){3}", u):
        score += 3

    # '@' trick
    if "@" in u:
        score += 3

    # too many subdomains
    if d.count(".") >= 4:
        score += 2

    # suspicious TLD
    if any(d.endswith(tld) for tld in SUSPICIOUS_TLDS):
        score += 2

    # suspicious path words (only for non-trusted domains)
    path = urlparse(u).path.lower()
    if any(w in path for w in SUSPICIOUS_PATH_WORDS):
        score += 2

    return score


# ----------------------------
# 4) Model inference
# ----------------------------
def _model_probs(text: str) -> tuple[float, float]:
    if model_manager.model is None or model_manager.tokenizer is None:
        raise ValueError("Model/tokenizer not loaded. Run load_trained_model() first.")

    inputs = model_manager.tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    model_manager.model.eval()
    with torch.no_grad():
        logits = model_manager.model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)

    legit_prob = float(probs[0].item())
    phish_prob = float(probs[1].item())
    return phish_prob, legit_prob


# ----------------------------
# 5) Final decision
# ----------------------------
def predict_email(text: str):
    """
    Returns: (label, phishing_prob, legitimate_prob)
    label is always one of: 'phishing' or 'legitimate'
    """

    text = (text or "").strip()
    urls = extract_urls(text)

    # HARD allowlist: if user sent ONLY a trusted domain/link, force legitimate.
    if _is_only_url_message(text, urls):
        d = get_domain(urls[0])
        if is_trusted_domain(d):
            # Force legit; keep probabilities sane for UI
            return "legitimate", 0.01, 0.99

    phish_prob, legit_prob = _model_probs(text)

    # URL-based guardrails
    total_risk = sum(url_risk_score(u) for u in urls)

    # If URLs exist and risk is high -> force phishing even if model says "legit"
    if urls and total_risk >= URL_RISK_FORCE:
        return "phishing", phish_prob, legit_prob

    # Medium URL risk + model slight suspicion -> phishing
    if urls and total_risk >= URL_RISK_STRONG and phish_prob >= MODEL_URL_ASSIST:
        return "phishing", phish_prob, legit_prob

    # Otherwise normal model threshold
    label = "phishing" if phish_prob >= MODEL_PHISH_THRESHOLD else "legitimate"
    return label, phish_prob, legit_prob