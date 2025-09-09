
"""
Security Alert Triage NLP (Mini)
- Multi-class classification for alert prioritization (Low/Medium/High)
- TF-IDF + LinearSVC baseline, with clear comments for interview readability.
"""

import argparse
import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

def parse_args():
    p = argparse.ArgumentParser(description="Security Alert Triage NLP (Mini)")
    p.add_argument("--data", default="data/alerts_small.csv", help="CSV with columns text,priority")
    p.add_argument("--output", default="outputs", help="Output directory")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # 1) Load dataset
    df = pd.read_csv(args.data)
    X = df["text"].astype(str)
    y = df["priority"].astype("category")

    # 2) Train/test split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # 3) TF-IDF vectorization (unigrams + bigrams) to capture phrases like "failed logins"
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, stop_words="english")
    Xtr_vec = vec.fit_transform(Xtr)
    Xte_vec = vec.transform(Xte)

    # 4) Linear SVM is a strong baseline for text classification
    clf = LinearSVC()
    clf.fit(Xtr_vec, ytr)

    # 5) Evaluate
    ypred = clf.predict(Xte_vec)
    report = classification_report(yte, ypred, output_dict=True)
    with open(os.path.join(args.output, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("Classification report saved -> outputs/classification_report.json")

    # 6) Confusion matrix
    disp = ConfusionMatrixDisplay.from_predictions(yte, ypred)
    plt.title("Alert Triage — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, "confusion_matrix.png"), dpi=150)

    # 7) Persist model + vectorizer for reuse
    joblib.dump(clf, os.path.join(args.output, "model_linear_svc.joblib"))
    joblib.dump(vec, os.path.join(args.output, "vectorizer_tfidf.joblib"))
    print("Saved model and vectorizer -> outputs/*.joblib")

if __name__ == "__main__":
    main()
