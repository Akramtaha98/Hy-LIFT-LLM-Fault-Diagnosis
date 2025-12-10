# Minimal Fault Diagnosis API (5G/6G-style logs)

This is a **simple and well-commented** starter implementation for automated fault diagnosis
using a rule-based baseline and a clean FastAPI endpoint. You can run it locally and push it to GitHub.

## 1) Install & Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Run the API
uvicorn app:app --reload --port 8000
```

Visit docs at: http://127.0.0.1:8000/docs

### Test with curl
```bash
curl -X POST http://127.0.0.1:8000/diagnose -H "Content-Type: application/json" -d '{
  "logs": ["[ERROR] gNB-23: Packet loss increased 80% (uplink)", "[WARNING] gNB-7: UPF latency 120ms"]
}'
```

## 2) Generate Sample Data (optional)
```bash
python data_gen.py --out sample_logs.csv --n 50
```

## 3) Upload to GitHub

1. Create a new empty repo on GitHub (e.g., `llm-fault-diagnosis-simple`).
2. In this folder run:
```bash
git init
git add .
git commit -m "feat: minimal, well-commented fault diagnosis API"
git branch -M main
git remote add origin https://github.com/<YOUR-USER>/llm-fault-diagnosis-simple.git
git push -u origin main
```

## What to extend next
- Replace or complement the rule-based method in `classify.py` with a real LLM call.
- Log latency per request and aggregate accuracy on labeled data.
- Add a confusion matrix plot and a short evaluation script.
