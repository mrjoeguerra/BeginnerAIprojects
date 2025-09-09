
import argparse
import os
import sys
import sqlite3
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_URL = "https://covid.ourworldindata.org/data/owid-covid-data.csv"

def parse_args():
    p = argparse.ArgumentParser(description="Mini ETL: COVID-19 CSV -> SQLite -> Analysis chart/CSV")
    p.add_argument("--url", default=DEFAULT_URL, help="CSV source URL (OWID)")
    p.add_argument("--country", default="United States", help="Country name to filter on")
    p.add_argument("--since", default="2022-01-01", help="Start date (YYYY-MM-DD) for query/plot")
    p.add_argument("--db", default="covid_data.db", help="SQLite database file name")
    p.add_argument("--table", default="covid_country", help="Table name for loaded data")
    p.add_argument("--output-dir", default="outputs", help="Directory for outputs (chart, CSV)")
    return p.parse_args()

def extract(url: str) -> pd.DataFrame:
    print(f"[Extract] Downloading CSV from: {url}")
    df = pd.read_csv(url)
    print(f"[Extract] Rows: {len(df):,}, Columns: {len(df.columns)}")
    return df

def transform(df: pd.DataFrame, country: str) -> pd.DataFrame:
    print(f"[Transform] Filtering for country: {country}")
    subset = df[df["location"] == country].copy()
    cols = ["date", "location", "new_cases", "total_cases"]
    subset = subset[cols]
    # Ensure proper types
    subset["date"] = pd.to_datetime(subset["date"], errors="coerce")
    subset["new_cases"] = subset["new_cases"].fillna(0).astype(float)
    subset["total_cases"] = subset["total_cases"].fillna(method="ffill").fillna(0).astype(float)
    print(f"[Transform] Filtered rows: {len(subset):,}")
    return subset

def load_to_sqlite(df: pd.DataFrame, db_path: str, table: str):
    print(f"[Load] Writing to SQLite: {db_path} (table: {table})")
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql(table, conn, if_exists="replace", index=False)
        conn.commit()
    finally:
        conn.close()
    print("[Load] Done.")

def query_daily_cases(db_path: str, table: str, since: str) -> pd.DataFrame:
    print(f"[Query] Since: {since}")
    conn = sqlite3.connect(db_path)
    try:
        q = f"""
        SELECT date, new_cases
        FROM {table}
        WHERE date >= ?
        ORDER BY date;
        """
        res = pd.read_sql(q, conn, params=[since], parse_dates=["date"])
        print(f"[Query] Returned rows: {len(res):,}")
        return res
    finally:
        conn.close()

def save_outputs(daily_cases: pd.DataFrame, output_dir: str, country: str):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "daily_cases.csv")
    daily_cases.to_csv(csv_path, index=False)
    print(f"[Output] CSV saved -> {csv_path}")

    # Plot
    plt.figure()
    plt.plot(daily_cases["date"], daily_cases["new_cases"])
    plt.title(f"{country} Daily COVID-19 New Cases")
    plt.xlabel("Date")
    plt.ylabel("New Cases")
    plt.xticks(rotation=45)
    plt.tight_layout()
    chart_name = f"{country.lower().replace(' ', '_')}_daily_cases.png"
    chart_path = os.path.join(output_dir, chart_name)
    plt.savefig(chart_path, dpi=150)
    print(f"[Output] Chart saved -> {chart_path}")

def main():
    args = parse_args()

    try:
        df = extract(args.url)
    except Exception as e:
        print(f"[Error] Failed to download or read CSV: {e}", file=sys.stderr)
        sys.exit(1)

    if "location" not in df.columns or "date" not in df.columns:
        print("[Error] Unexpected CSV schema: missing required columns.", file=sys.stderr)
        sys.exit(2)

    try:
        subset = transform(df, args.country)
        if subset.empty:
            print(f"[Warn] No rows found for country '{args.country}'. "
                  f"Try a different --country value.", file=sys.stderr)
            sys.exit(3)
    except Exception as e:
        print(f"[Error] Transform step failed: {e}", file=sys.stderr)
        sys.exit(4)

    try:
        load_to_sqlite(subset, args.db, args.table)
    except Exception as e:
        print(f"[Error] Load step failed: {e}", file=sys.stderr)
        sys.exit(5)

    try:
        daily_cases = query_daily_cases(args.db, args.table, args.since)
        if daily_cases.empty:
            print(f"[Warn] Query returned no rows since {args.since}.", file=sys.stderr)
    except Exception as e:
        print(f"[Error] Query step failed: {e}", file=sys.stderr)
        sys.exit(6)

    try:
        save_outputs(daily_cases, args.output_dir, args.country)
    except Exception as e:
        print(f"[Error] Output step failed: {e}", file=sys.stderr)
        sys.exit(7)

    print("[Done] ETL + analysis complete.")

if __name__ == "__main__":
    main()
