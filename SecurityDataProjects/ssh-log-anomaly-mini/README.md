
# SSH Auth Log Anomaly Detection (Mini)

Generate a synthetic auth log, engineer features, detect anomalies with **IsolationForest**,
and layer **rule-based** checks (burst failures, off-hours successes).

## Quickstart
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python ssh_anomaly.py --rows 3000 --output outputs
```
Artifacts: `outputs/auth_log.csv`, `outputs/flagged_events.csv`, `outputs/failed_per_hour.png`.
