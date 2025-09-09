
"""
SSH Auth Log Anomaly Detection (Mini)
- Builds synthetic auth logs, engineers features, applies IsolationForest,
  and uses simple rules (burst failures, off-hours) to flag anomalies.
- Heavily commented for interview-readability.
"""

import argparse
import os
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

USERS = ["alice", "bob", "carol", "dave", "eve", "service"]
IPS = [f"192.168.1.{i}" for i in range(10, 80)] + [f"10.0.0.{i}" for i in range(3, 30)]
STATUSES = ["FAILED", "FAILED", "FAILED", "SUCCESS"]  # more failures than successes

def synthesize_logs(n_rows: int, seed: int = 42):
    """
    Create a simple synthetic SSH auth log with plausible patterns:
    - Mostly normal behavior with a few injected anomalies:
      * brute-force bursts from a single IP
      * odd-hour successes
    """
    rng = random.Random(seed)
    start = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0) - timedelta(days=2)
    rows = []
    for i in range(n_rows):
        ts = start + timedelta(minutes=i * rng.randint(1, 3))
        user = rng.choice(USERS)
        ip = rng.choice(IPS)
        status = rng.choice(STATUSES)
        rows.append({"ts": ts, "user": user, "src_ip": ip, "status": status})

    df = pd.DataFrame(rows)

    # Inject a brute-force pattern: many failures from one IP
    bf_ip = "203.0.113.45"
    for j in range(50):
        ts = start + timedelta(hours=10, minutes=j)  # tight burst
        df.loc[len(df)] = {"ts": ts, "user": rng.choice(USERS), "src_ip": bf_ip, "status": "FAILED"}

    # Inject odd-hour successful login from rare IP
    df.loc[len(df)] = {"ts": start.replace(hour=3, minute=13), "user": "alice", "src_ip": "198.51.100.99", "status": "SUCCESS"}

    df = df.sort_values("ts").reset_index(drop=True)
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering at event-level to feed the detector:
    - hour of day
    - rolling failures per IP (last ~50 events as a proxy for 60 minutes)
    - expanding unique users per IP (proxy for credential stuffing variety)
    - encode status
    """
    df = df.copy()
    df["hour"] = df["ts"].dt.hour
    df["is_failed"] = (df["status"] == "FAILED").astype(int)
    df["is_success"] = (df["status"] == "SUCCESS").astype(int)

    # Rolling failure count per IP in trailing 'window' events (simple proxy for time)
    df = df.sort_values(["src_ip", "ts"]).reset_index(drop=True)
    df["fail_rolling_60m"] = (
        df.groupby("src_ip")["is_failed"]
          .rolling(window=50, min_periods=1)
          .sum().reset_index(level=0, drop=True)
    )

    # Expanding unique users per IP (proxy for breadth of credential attempts)
    df["unique_users_seen"] = (
        df.groupby("src_ip")["user"]
          .transform(lambda s: s.expanding().apply(lambda x: len(set(x)), raw=False))
    )

    return df

def detect_anomalies(df_feat: pd.DataFrame, contamination: float = 0.02) -> pd.DataFrame:
    """
    Unsupervised anomaly detection using IsolationForest on selected features.
    """
    feat_cols = ["hour", "fail_rolling_60m", "unique_users_seen", "is_failed", "is_success"]
    X = df_feat[feat_cols].values
    iso = IsolationForest(random_state=42, contamination=contamination)
    iso.fit(X)
    scores = iso.decision_function(X)
    preds = iso.predict(X)  # -1 anomaly, 1 normal
    df_out = df_feat.copy()
    df_out["iforest_score"] = scores
    df_out["iforest_pred"] = preds
    return df_out

def apply_rules(df: pd.DataFrame):
    """
    Simple deterministic rules that a SOC might use:
    - burst failures from single IP (>= 20 in short span)
    - success login at odd hours (1-4 AM)
    """
    rule_burst = df["fail_rolling_60m"] >= 20
    rule_odd_success = (df["is_success"] == 1) & (df["hour"].between(1,4))
    return rule_burst | rule_odd_success

def visualize(df: pd.DataFrame, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    hourly = df.set_index("ts").resample("1H")["is_failed"].sum()
    plt.figure()
    hourly.plot()
    plt.title("Failed Logins per Hour")
    plt.xlabel("Time")
    plt.ylabel("Failures")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "failed_per_hour.png"), dpi=150)

def parse_args():
    p = argparse.ArgumentParser(description="SSH Log Anomaly Detection (Mini)")
    p.add_argument("--rows", type=int, default=3000, help="Number of synthetic rows to generate")
    p.add_argument("--output", default="outputs", help="Output folder")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # 1) Synthesize logs
    df = synthesize_logs(args.rows)
    df.to_csv(os.path.join(args.output, "auth_log.csv"), index=False)

    # 2) Engineer features
    df_feat = engineer_features(df)

    # 3) IsolationForest
    df_if = detect_anomalies(df_feat, contamination=0.02)

    # 4) Rules
    df_if["rule_flag"] = apply_rules(df_if)

    # 5) Combined risk (either IF anomaly or rule flag)
    df_if["flagged"] = (df_if["iforest_pred"] == -1) | (df_if["rule_flag"] == True)

    # 6) Save flagged events
    flagged = df_if[df_if["flagged"]].copy()
    flagged.to_csv(os.path.join(args.output, "flagged_events.csv"), index=False)

    # 7) Visualize
    visualize(df_if, args.output)

    print(f"Done. Rows={len(df_if)}, Flagged={len(flagged)}")
    print("Artifacts: outputs/auth_log.csv, outputs/flagged_events.csv, outputs/failed_per_hour.png")

if __name__ == "__main__":
    main()
