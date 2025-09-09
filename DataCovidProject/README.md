
# COVID-19 Mini Data Pipeline (CSV → SQLite → Analysis)

A tiny, portfolio-ready **ETL data engineering** project you can run end-to-end:
- **Extract**: Download COVID-19 data from Our World in Data (OWID).
- **Transform**: Clean and filter by country and date.
- **Load**: Persist the result into an **SQLite** database.
- **Analyze**: Query and **visualize** daily new cases.

> Designed to be completed in ~1 hour and look great in your GitHub portfolio.

---

## Tech Stack
- Python 3.9+
- pandas
- matplotlib
- SQLite (via Python's built-in `sqlite3`)

---

## Quickstart

```bash
# 1) Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the ETL + analysis (defaults to United States since 2022-01-01)
python covid_pipeline.py

# Optional: pick a different country and start date
python covid_pipeline.py --country "Canada" --since "2022-01-01"

# Optional: change output folder / db name
python covid_pipeline.py --output-dir outputs --db covid_data.db
```

Outputs:
- `outputs/us_daily_cases.png` (or `<country>_daily_cases.png`) — line chart of daily new cases.
- `outputs/daily_cases.csv` — query result exported to CSV.
- `covid_data.db` — SQLite database with the table of filtered rows.

---

## What this shows in your portfolio
- Data ingestion from a public source (CSV).
- Data cleaning and transformation with pandas.
- Loading into a relational store (SQLite).
- Basic analytics with SQL and matplotlib.
- CLI parameters and clean, reproducible scripts.

---

## Notes
- Source data: Our World in Data COVID-19 dataset: https://covid.ourworldindata.org/data/owid-covid-data.csv
- The script includes basic exception handling and sensible defaults.
- Internet access is required to fetch the CSV at runtime.
