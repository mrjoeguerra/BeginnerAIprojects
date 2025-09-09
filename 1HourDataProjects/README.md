
# Data Engineering Portfolio

This portfolio showcases **three compact, 1-hour projects** that demonstrate data engineering,
ETL pipelines, analytics, and lightweight ML/NLP. Each project is designed to be instructive,
well-commented, and reproducible.

---

## 📊 COVID-19 Mini Data Pipeline
**Tech:** Python, pandas, matplotlib, SQLite  
**Highlights:**
- ETL pipeline: Extract CSV → Transform (clean/filter) → Load SQLite.
- SQL query for daily cases.
- Chart output with matplotlib.
- Reproducible CLI with arguments.

---

## 🚕 NYC Taxi: Parquet → DuckDB → SQL + Tiny ML
**Tech:** DuckDB, pandas, scikit-learn, matplotlib  
**Highlights:**
- Ingest Parquet over HTTP into DuckDB (columnar analytics).
- Data cleaning, feature engineering (trip duration, hour-of-day).
- SQL analytics for QA and insights.
- Tiny Linear Regression to predict trip duration.
- Output: QA CSVs, charts, model metrics JSON.

---

## 📚 Wikipedia Scraper → CSV + TF-IDF Keywords/Summary
**Tech:** requests, BeautifulSoup, pandas, scikit-learn, matplotlib  
**Highlights:**
- Web scraping of Wikipedia article HTML.
- Clean structured CSV export of sections/paragraphs.
- TF-IDF keyword extraction with bar chart.
- Naive extractive text summary.
- Clear CLI and modular design.

---

## Why This Portfolio Works
- Each repo has **clean structure**: README, requirements, .gitignore, outputs folder.
- **Heavily commented code** explains decisions and tradeoffs.
- Covers **SQL, ETL, ML, NLP, scraping, visualization**.
- Recruiters and technical reviewers can see both **breadth and depth**.

---

## Next Steps
- Add Dockerfiles for containerized runs.
- Expand ML models beyond baselines.
- Deploy one pipeline to a cloud service for demonstration.

---
Prepared by **Joe Guerra**  
M.Ed., CASP+, CCSP
