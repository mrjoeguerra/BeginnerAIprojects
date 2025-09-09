
# Security Alert Triage NLP (Mini)

Classify short security alerts into **Low / Medium / High** priority using **TF‑IDF + LinearSVC**.

## Quickstart
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python train_triage.py --data data/alerts_small.csv --output outputs
```
Artifacts: `outputs/classification_report.json`, `outputs/confusion_matrix.png`, `outputs/*.joblib`.
