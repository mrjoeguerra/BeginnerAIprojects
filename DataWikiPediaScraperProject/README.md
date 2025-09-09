
# Wikipedia Scraper → CSV + TF‑IDF Keyword Extraction (NLP)

This project shows a simple **data collection + lightweight NLP** workflow:
1) **Extract**: Download a Wikipedia article by topic.
2) **Parse/Transform**: Clean text from HTML (remove citations, boilerplate).
3) **Load**: Save structured content to CSV.
4) **NLP**: Use **TF‑IDF** to extract top keywords and generate a naive summary.

> Instructive comments are included to demonstrate reasoning and tradeoffs.

## Tech Stack
- Python: requests, beautifulsoup4, pandas, scikit-learn, matplotlib (for a quick keyword bar chart).

## Quickstart
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Scrape + analyze a topic (spaces allowed)
python wiki_scrape_nlp.py --topic "Data engineering"

# Or specify a full URL explicitly
python wiki_scrape_nlp.py --url "https://en.wikipedia.org/wiki/Machine_learning"
```

### Outputs
- `outputs/article.csv` — cleaned paragraphs with section labels.
- `outputs/top_keywords.csv` — TF‑IDF top terms.
- `outputs/summary.txt` — naive extractive summary.
- `outputs/top_keywords.png` — quick bar chart for top terms.

### Why this is portfolio‑worthy
- Shows pragmatic **web scraping** with respect for HTML structure.
- Clean text processing pipeline + export to structured CSV.
- **NLP** feature extraction (TF‑IDF) and basic summarization.
- Clear CLI and logging make it reproducible and reviewable.
