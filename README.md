# ML-Model-IS455-PredictFraud

Fraud scoring for IS455 Chapter 17: trained sklearn pipeline, CRISP-DM notebook, FastAPI scoring service (Supabase), and SQLite migration helper.

## Contents

- `Chapter_17_Part2_Fraud_CRISP_DM.ipynb` — full CRISP-DM workflow
- `fraud_model.joblib` — serialized `Pipeline` (preprocess + classifier)
- `score_api.py` — FastAPI: `GET /health`, `POST /run-scoring` (Supabase)
- `migrate_schema.py` — optional SQLite migration for nullable `is_fraud`
- `requirements.txt` — Python dependencies
- `shop.db` / `shop.db.zip` — sample operational database

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # add SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY
uvicorn score_api:app --reload --port 8000
```

## Remote

[https://github.com/lafebenn/ML-Model-IS455-PredictFraud](https://github.com/lafebenn/ML-Model-IS455-PredictFraud)
