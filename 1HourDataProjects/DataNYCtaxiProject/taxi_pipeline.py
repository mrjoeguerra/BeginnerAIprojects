
"""
NYC Taxi Pipeline: Parquet -> DuckDB -> SQL Analytics -> Tiny ML

This script is **deliberately verbose** with comments and logs to show
engineering thinking, assumptions, and tradeoffs to reviewers.
"""

import os
import json
import argparse
from datetime import datetime
import duckdb  # fast embedded analytics DB; great for Parquet + SQL
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# TLC provides monthly Parquet files via a CloudFront CDN. Using DuckDB's httpfs
# we can read them directly without downloading first.
URL_TEMPLATE = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"

def parse_args():
    p = argparse.ArgumentParser(description="NYC Taxi: Parquet → DuckDB → SQL + Tiny ML")
    p.add_argument("--year", type=int, default=2023, help="Year of the parquet to read")
    p.add_argument("--month", type=int, default=1, help="Month of the parquet to read")
    p.add_argument("--limit", type=int, default=100_000, help="Row limit for speed in demos")
    p.add_argument("--db", default="taxi.duckdb", help="DuckDB file name")
    p.add_argument("--table", default="trips_clean", help="Target table name")
    p.add_argument("--output-dir", default="outputs", help="Where to save charts/CSVs")
    return p.parse_args()

def ensure_httpfs(conn):
    """
    Enable DuckDB's HTTPFS so we can read from https:// directly.
    """
    conn.execute("INSTALL httpfs;")
    conn.execute("LOAD httpfs;")

def create_or_connect(db_path: str) -> duckdb.DuckDBPyConnection:
    """
    Connect to a DuckDB database file (created if missing).
    """
    return duckdb.connect(db_path)

def ingest_parquet(conn, url: str, limit: int) -> pd.DataFrame:
    """
    Use DuckDB to read Parquet over HTTP, then limit rows for demo speed.
    We immediately do some column selection to keep memory reasonable.
    """
    print(f"[Extract] Reading Parquet from {url} with LIMIT {limit:,}")
    query = f"""
        SELECT 
            tpep_pickup_datetime,
            tpep_dropoff_datetime,
            passenger_count,
            trip_distance,
            PULocationID,
            DOLocationID,
            fare_amount,
            total_amount,
            payment_type
        FROM read_parquet('{url}')
        LIMIT {limit}
    """
    return conn.execute(query).fetch_df()

def clean_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic data quality cleanup and feature engineering.
    The goal is not to be perfect but to show reasonable hygiene.
    """
    print("[Transform] Cleaning and engineering features")
    # Parse timestamps
    for col in ["tpep_pickup_datetime", "tpep_dropoff_datetime"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    # Drop rows with invalid datetimes
    df = df.dropna(subset=["tpep_pickup_datetime", "tpep_dropoff_datetime"]).copy()
    # Compute trip duration in minutes
    df["trip_minutes"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60.0
    # Filter outliers: negative or too long durations, zero/negative distance
    df = df[(df["trip_minutes"] > 0) & (df["trip_minutes"] <= 180)]
    df = df[df["trip_distance"] > 0]

    # Derive hour-of-day (useful signal for traffic)
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    # Payment type as category -> numeric codes for modeling
    df["payment_type"] = df["payment_type"].fillna(0).astype(int)

    # Reasonable fare checks (non-negative)
    for col in ["fare_amount", "total_amount"]:
        df[col] = df[col].fillna(0)
        df = df[df[col] >= 0]

    print(f"[Transform] Rows after cleaning: {len(df):,}")
    return df

def load_to_duckdb(conn, df: pd.DataFrame, table: str):
    """
    Replace target table with our cleaned data. DuckDB can ingest a pandas DF directly.
    """
    print(f"[Load] Writing cleaned data to table '{table}'")
    conn.execute(f"DROP TABLE IF EXISTS {table};")
    conn.register("df_view", df)
    conn.execute(f"CREATE TABLE {table} AS SELECT * FROM df_view;")
    conn.unregister("df_view")

def run_analytics(conn, table: str, output_dir: str):
    """
    A couple of compact SQLs that demonstrate analytics and QA mindset.
    """
    os.makedirs(output_dir, exist_ok=True)

    # QA metrics (sanity checks)
    qa_df = conn.execute(f"""
        SELECT
            COUNT(*) AS rows,
            AVG(trip_distance) AS avg_distance,
            AVG(trip_minutes) AS avg_minutes,
            AVG(total_amount) AS avg_total_amount
        FROM {table};
    """).fetch_df()
    qa_path = os.path.join(output_dir, "qa_metrics.csv")
    qa_df.to_csv(qa_path, index=False)
    print(f"[Analyze] Saved QA metrics -> {qa_path}")

    # Example analytic: top pickup hours by average duration
    top_hours = conn.execute(f"""
        SELECT pickup_hour, AVG(trip_minutes) AS avg_minutes, COUNT(*) AS n
        FROM {table}
        GROUP BY 1
        ORDER BY avg_minutes DESC
        LIMIT 24;
    """).fetch_df()
    top_hours_path = os.path.join(output_dir, "top_hours.csv")
    top_hours.to_csv(top_hours_path, index=False)
    print(f"[Analyze] Saved top hours -> {top_hours_path}")

def tiny_model(df: pd.DataFrame, output_dir: str):
    """
    Tiny ML: predict trip_minutes from trip_distance + pickup_hour + passenger_count.
    Intent: show modeling workflow, not state-of-the-art accuracy.
    """
    feat_cols = ["trip_distance", "pickup_hour", "passenger_count"]
    X = df[feat_cols].fillna(0)
    y = df["trip_minutes"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    os.makedirs(output_dir, exist_ok=True)
    metrics = {"r2": float(r2), "mae_minutes": float(mae), "n_train": int(len(X_train)), "n_test": int(len(X_test))}
    with open(os.path.join(output_dir, "model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Model] R^2={r2:.3f}, MAE={mae:.2f} min -> outputs/model_metrics.json")

    # Quick diagnostic scatter plot
    sample = df.sample(min(5000, len(df)), random_state=42)[["trip_distance", "trip_minutes"]]
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(sample["trip_distance"], sample["trip_minutes"], s=5, alpha=0.5)
    plt.xlabel("Trip distance (miles)")
    plt.ylabel("Trip duration (minutes)")
    plt.title("NYC Taxi: Duration vs Distance (sample)")
    plt.tight_layout()
    chart_path = os.path.join(output_dir, "duration_vs_distance.png")
    plt.savefig(chart_path, dpi=150)
    print(f"[Model] Saved chart -> {chart_path}")

def main():
    args = parse_args()
    url = URL_TEMPLATE.format(year=args.year, month=args.month)

    # Connect to DuckDB and enable HTTP reading
    conn = create_or_connect(args.db)
    ensure_httpfs(conn)

    # Extract
    raw_df = ingest_parquet(conn, url, args.limit)

    # Transform
    clean_df = clean_transform(raw_df)

    # Load
    load_to_duckdb(conn, clean_df, args.table)

    # Analyze
    run_analytics(conn, args.table, args.output_dir)

    # Model
    tiny_model(clean_df, args.output_dir)

    print("[Done] Ingestion → Warehouse → Analytics → ML complete.")

if __name__ == "__main__":
    main()
