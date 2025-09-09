
"""
Phishing URL Detector (Mini)
- Teaches feature engineering + TF-IDF + LogisticRegression for security URLs.
- Written with instructor-style comments so reviewers see your reasoning.
"""

import argparse
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from scipy.sparse import hstack

# --- Helper: handcrafted numeric features for URLs ---
def url_numeric_features(urls):
    """
    For each URL, compute simple security-relevant features:
    - length of URL
    - count of digits
    - count of special characters (.,-,_,@,?,=,&)
    - presence of suspicious tokens (login, verify, update, secure, account, billing, password)
    The goal isn't perfection; it's to show the thought process and create useful signals.
    """
    rows = []
    suspicious = ("login", "verify", "update", "secure", "account", "billing", "password")
    for u in urls:
        s = str(u)
        rows.append([
            len(s),
            sum(ch.isdigit() for ch in s),
            sum(ch in ".-_/@?=&" for ch in s),
            int(any(tok in s.lower() for tok in suspicious)),
        ])
    return pd.DataFrame(rows, columns=["len", "digits", "specials", "has_suspicious"])

def parse_args():
    p = argparse.ArgumentParser(description="Phishing URL Detector (Mini)")
    p.add_argument("--data", default="data/urls_small.csv", help="CSV with columns url,label")
    p.add_argument("--output", default="outputs", help="Output directory for artifacts")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # 1) Load data
    df = pd.read_csv(args.data)
    X_text = df["url"].astype(str)
    y = df["label"].astype(int)

    # 2) Split train/test (stratify to keep class balance)
    Xtr_text, Xte_text, ytr, yte = train_test_split(X_text, y, test_size=0.3, random_state=42, stratify=y)

    # 3) TF-IDF on character n-grams (3..5). This captures subword patterns like ".secure-" or "/login"
    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1)
    Xtr_tfidf = tfidf.fit_transform(Xtr_text)
    Xte_tfidf = tfidf.transform(Xte_text)

    # 4) Numeric features
    Xtr_num = url_numeric_features(Xtr_text)
    Xte_num = url_numeric_features(Xte_text)

    # 5) Combine sparse tfidf with dense numeric features
    Xtr = hstack([Xtr_tfidf, Xtr_num.values])
    Xte = hstack([Xte_tfidf, Xte_num.values])

    # 6) Train a simple Logistic Regression (interpretable baseline)
    clf = LogisticRegression(max_iter=200)
    clf.fit(Xtr, ytr)

    # 7) Evaluate
    ypred = clf.predict(Xte)
    report = classification_report(yte, ypred, output_dict=True)
    with open(os.path.join(args.output, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("Classification report saved -> outputs/classification_report.json")

    # 8) Confusion matrix plot
    disp = ConfusionMatrixDisplay.from_predictions(yte, ypred)
    plt.title("Phishing URL Detector — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, "confusion_matrix.png"), dpi=150)
    print("Confusion matrix saved -> outputs/confusion_matrix.png")

if __name__ == "__main__":
    main()
