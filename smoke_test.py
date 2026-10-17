"""Part D smoke test: starts the local service and exercises eleven requests."""
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

BASE_URL = "http://127.0.0.1:8765"
ROOT = Path(__file__).resolve().parent


def main():
    if not (ROOT / "fraud_model.pkl").exists():
        raise SystemExit("Missing fraud_model.pkl; run `python prepare.py` and `python train.py` first.")
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8765", "--log-level", "warning"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rows = []
    try:
        for _ in range(60):
            try:
                if requests.get(BASE_URL + "/health", timeout=0.5).status_code == 200:
                    break
            except requests.RequestException:
                time.sleep(0.25)
        else:
            raise RuntimeError("service did not start")

        scenarios = json.loads((ROOT / "case_scenarios.json").read_text(encoding="utf-8"))
        # Ten provided scenarios cover the deterministic mock's injected defects.
        for item in scenarios:
            try:
                response = requests.post(BASE_URL + "/triage", json=item, timeout=5)
                try:
                    body = response.json()
                except ValueError:
                    body = {"error": "non_json_response"}
                rows.append({"request": item["txn_id"], "http": response.status_code,
                             "status": body.get("status", "error"), "decision": body.get("decision", "-"),
                             "attempts": body.get("attempts", "-"), "pass": response.status_code in (200, 503)})
            except requests.RequestException as exc:
                rows.append({"request": item["txn_id"], "http": "ERR", "status": str(exc), "decision": "-", "attempts": "-", "pass": False})
        # Eleventh request exercises the mock model outage and fail-closed response.
        outage = {"txn_id": "T900099", "fraud_probability": 0.99, "amount_inr": 12000, "question": "card payment above INR 50000 from a new device"}
        response = requests.post(BASE_URL + "/triage", json=outage, timeout=5)
        body = response.json()
        rows.append({"request": outage["txn_id"], "http": response.status_code, "status": body.get("status", "error"),
                     "decision": body.get("decision", "-"), "attempts": body.get("attempts", "-"),
                     "pass": response.status_code == 503 and body.get("status") == "unavailable"})

        print("PART D SMOKE TEST — 11 REQUESTS")
        print("request | http | status | decision | attempts | pass")
        for row in rows:
            print(f"{row['request']} | {row['http']} | {row['status']} | {row['decision']} | {row['attempts']} | {row['pass']}")
        metrics = requests.get(BASE_URL + "/metrics", timeout=3).json()
        print("metrics=" + json.dumps(metrics, sort_keys=True))
        print(f"smoke_passed={sum(bool(r['pass']) for r in rows)}/{len(rows)}")
        if not all(row["pass"] for row in rows):
            raise SystemExit(1)
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)


if __name__ == "__main__":
    main()
