# Razorpay Merchant Risk Desk — Fraud Triage Service

## Overview
This is a local, offline fraud-triage capstone built on fictional synthetic payment data.
Part A audits and cleans the export, then engineers risk features.
Part B trains an imbalanced-class classifier and selects an alert threshold using validation data.
Part C retrieves a relevant policy and validates structured case notes from the supplied offline mock language model.
Part D exposes triage and model-scoring routes through FastAPI, with input validation, logs, metrics, and a smoke test.
The project is educational and is not affiliated with or endorsed by Razorpay.
It does not use real customer data, external model APIs, paid services, or cloud deployment.
A hold decision means human review; the model does not independently establish fraud.

## Reproduce
Run every command from the repository root using Python 3.11 or newer. The supplied `generate_data.py` and `mock_llm.py` are used unmodified.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```

Then run these commands in order:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python generate_data.py
python prepare.py
python train.py
python retrieval.py
python casenote.py
python smoke_test.py
```

`generate_data.py` writes `transactions_raw.csv`, `fraud_policies.json`, `retrieval_queries.json`, and `case_scenarios.json`. The model training step creates `fraud_model.pkl` and `alert_threshold.txt`; these are generated artifacts and are intentionally ignored by Git. `smoke_test.py` starts a local server on `127.0.0.1:8765`, sends eleven requests, prints results and updates `service.log`. Stop any other process using that port before running the test.

To run the service manually after preparation and training:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

Routes: `GET /health`, `GET /metrics`, `POST /triage`, and `POST /score`. `/triage` expects `txn_id`, `fraud_probability`, `amount_inr`, and `question`. `/score` accepts transaction features and returns a model score and decision. The interactive API schema is available at `http://127.0.0.1:8000/docs` while the server is running.

## Part A results
`prepare.py` audited 6,090 raw rows, found 6,000 unique transaction IDs, removed 90 duplicates, identified 76 missing amounts and 5 invalid amounts, and produced 5,919 clean rows with 366 fraud rows (6.18%). Median imputation left zero missing distance values. Feature engineering adds amount-to-average ratio, log-transformed amount/distance values, night-time, high-velocity, and new-account flags. The model excludes `chargeback_filed` because it can be a post-payment outcome and create target leakage.

## Part B results
`train.py` used a stratified 60/20/20 split with random state `2605`: 3,551 train rows (220 fraud), 1,184 validation rows (73 fraud), and 1,184 test rows (73 fraud). The scaled, one-hot-encoded balanced logistic regression selected an alert threshold of 0.5500. Test average precision was 0.6177, ROC-AUC 0.9249, precision 0.2817, recall 0.8219, and F1 0.4196; confusion counts were TN=958, FP=153, FN=13, TP=60. The threshold maximizes validation recall among tested thresholds whose precision is at least 0.30; the final persisted model is refit on train plus validation. The validation precision floor is an explicit operating choice, not a universal business rule.

## Part C results
`retrieval.py` matched all 10 retrieval-query checkpoints (10/10) using TF-IDF/cosine similarity and `MIN_SCORE=0.16`; the two unrelated queries returned no policy. All 10 case scenarios produced valid held notes. The retry handler needed up to four attempts for `T900005`, and handled malformed JSON, missing fields, invalid values, invalid teams/actions, and policy mismatch. It accepts only a complete JSON object with exactly the required fields, approved enum values, a matching policy ID, and a non-empty bounded summary; persistent failure is sent to manual fallback, and a simulated model outage fails closed.

## Part D results
`app.py` exposes health, metrics, model scoring, and policy-grounded triage endpoints with Pydantic request validation. The smoke test passed 11/11 requests: the ten supplied scenarios returned HTTP 200 and the simulated `T900099` outage returned HTTP 503. The final smoke-run metrics recorded 13 requests including health and metrics checks, 10 held triage cases, one model outage, zero service errors, and zero validation errors. Middleware logs request method, path, status, and duration to the committed `service.log`; the service binds to localhost and is not deployed.

## Limitations
The dataset is synthetic and does not represent real-world payment fraud, merchant behaviour, or production class prevalence.
The model is a simple logistic regression and has not been calibrated or independently validated for live financial decisions.
The threshold is selected from a small fixed grid and a chosen precision floor, rather than an approved operational cost model.
TF-IDF retrieval is lexical and can miss relevant policy meaning when a query uses unfamiliar wording.
The supplied mock language model is deterministic and does not represent the reliability or capabilities of a production language model.
The service uses in-process counters and a local log file, so metrics are not durable or shared across multiple workers.
Authentication, authorization, rate limiting, persistent case storage, analyst feedback, and production monitoring are not implemented.
A human risk analyst must review held transactions before any real settlement decision.
