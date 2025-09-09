
"""
Wikipedia Scraper -> CSV + TF-IDF Keyword Extraction

This script intentionally includes detailed comments/log messages so a reviewer
can assess your understanding of scraping, cleaning, and simple NLP.
"""

import re
import os
import argparse
from urllib.parse import quote
import requests
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

WIKI_BASE = "https://en.wikipedia.org/wiki/"

def parse_args():
    p = argparse.ArgumentParser(description="Wikipedia -> CSV + TF-IDF keywords & summary")
    p.add_argument("--topic", help="Wikipedia topic title (e.g., 'Data engineering')")
    p.add_argument("--url", help="Full Wikipedia article URL (overrides --topic)")
    p.add_argument("--output-dir", default="outputs", help="Where to save CSV/plots")
    p.add_argument("--topk", type=int, default=15, help="Top K keywords to extract")
    return p.parse_args()

def fetch_article_html(topic: str = None, url: str = None) -> str:
    """
    Fetch the HTML of a Wikipedia page either by topic or full URL.
    We use the regular article path; for deep production work use the API.
    """
    if url:
        target = url
    elif topic:
        # Wikipedia replaces spaces with underscores in URLs.
        target = WIKI_BASE + quote(topic.replace(" ", "_"))
    else:
        raise ValueError("Provide either --topic or --url")
    print(f"[Extract] GET {target}")
    resp = requests.get(target, timeout=20)
    resp.raise_for_status()
    return resp.text

def clean_text(text: str) -> str:
    """
    Light cleanup to remove reference markers like [1], [2], etc. and collapse spaces.
    """
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def parse_article(html: str) -> pd.DataFrame:
    """
    Parse the article structure:
    - Title
    - Headings (h2/h3) and paragraphs beneath
    Returns a DataFrame with columns: section, paragraph.
    """
    soup = BeautifulSoup(html, "html.parser")
    # Main content is inside <div id="mw-content-text">
    content = soup.find("div", {"id": "mw-content-text"})
    if content is None:
        raise RuntimeError("Could not locate main content on the page.")

    sections = []
    current_section = "Introduction"
    # Iterate over children to maintain document order
    for elem in content.descendants:
        if elem.name in ["h2", "h3"]:
            # Section headings contain a span with class mw-headline
            headline = elem.find("span", {"class": "mw-headline"})
            if headline and headline.text:
                current_section = headline.text.strip()
        elif elem.name == "p":
            paragraph = clean_text(elem.get_text())
            if paragraph and len(paragraph.split()) > 10:  # keep only substantial paragraphs
                sections.append({"section": current_section, "paragraph": paragraph})

    df = pd.DataFrame(sections)
    print(f"[Transform] Parsed paragraphs: {len(df):,}")
    return df

def save_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[Load] Saved CSV -> {path}")

def tfidf_keywords(df: pd.DataFrame, topk: int):
    """
    Fit a TF-IDF vectorizer on paragraphs; compute global top terms by mean TF-IDF.
    """
    texts = df["paragraph"].tolist()
    vectorizer = TfidfVectorizer(
        max_df=0.9, min_df=2, ngram_range=(1,2), stop_words="english"
    )
    X = vectorizer.fit_transform(texts)
    vocab = np.array(vectorizer.get_feature_names_out())
    # Mean TF-IDF across documents to rank global importance
    means = X.mean(axis=0).A1
    idx = np.argsort(means)[::-1][:topk]
    top_terms = vocab[idx]
    scores = means[idx]
    keywords = pd.DataFrame({"term": top_terms, "score": scores})
    return keywords

def plot_keywords(keywords: pd.DataFrame, out_path: str):
    plt.figure()
    plt.barh(keywords["term"][::-1], keywords["score"][::-1])
    plt.title("Top TF-IDF Terms")
    plt.xlabel("Score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[NLP] Saved keyword plot -> {out_path}")

def naive_summary(df: pd.DataFrame, keywords: pd.DataFrame, k_sentences: int = 7) -> str:
    """
    Naive extractive summary:
    - Score sentences by presence of top TF-IDF terms
    - Return top-K sentences in original order
    This is intentionally simple (no heavy models) but showcases the NLP idea.
    """
    top_terms = set(keywords["term"].tolist())
    sentences = []
    for _, row in df.iterrows():
        for sent in re.split(r"(?<=[.!?])\s+", row["paragraph"]):
            if len(sent.split()) < 6:  # skip very short
                continue
            sentences.append(sent.strip())

    def score_sentence(s: str) -> int:
        toks = re.findall(r"[A-Za-z0-9_'-]+", s.lower())
        return sum(1 for t in toks if t in top_terms)

    scored = [(i, s, score_sentence(s)) for i, s in enumerate(sentences)]
    # Pick top by score, then re-sort by original order
    top = sorted(sorted(scored, key=lambda x: x[2], reverse=True)[:k_sentences], key=lambda x: x[0])
    summary = "\n".join(s for _, s, _ in top)
    return summary

def main():
    args = parse_args()
    html = fetch_article_html(topic=args.topic, url=args.url)
    df = parse_article(html)

    # Load to CSV (structured)
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "article.csv")
    save_csv(df, csv_path)

    # NLP: keywords
    keywords = tfidf_keywords(df, args.topk)
    kw_csv = os.path.join(out_dir, "top_keywords.csv")
    keywords.to_csv(kw_csv, index=False)
    print(f"[NLP] Saved top keywords -> {kw_csv}")

    # Plot keywords
    kw_png = os.path.join(out_dir, "top_keywords.png")
    plot_keywords(keywords, kw_png)

    # Naive extractive summary
    summary_text = naive_summary(df, keywords, k_sentences=7)
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"[NLP] Saved summary -> {summary_path}")

    print("[Done] Scrape → Clean → CSV → TF-IDF → Summary complete.")

if __name__ == "__main__":
    main()
