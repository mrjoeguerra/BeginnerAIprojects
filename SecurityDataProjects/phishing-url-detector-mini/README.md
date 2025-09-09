
# Phishing URL Detector (Mini)

A compact, fully commented **security + ML** project that classifies URLs as **phishing (1)** or **benign (0)**.

## What this demonstrates
- Pragmatic **feature engineering** for URLs (length, digits, suspicious tokens).
- **TF‑IDF** on character n‑grams (URLs behave like character streams).
- **Logistic Regression** baseline classifier, easy to explain.
- Clear metrics (classification report) and a confusion matrix chart.

## Quickstart
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python train_phishing.py --data data/urls_small.csv --output outputs
```
Artifacts will appear in `outputs/`.
