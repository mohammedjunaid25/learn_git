"""Part C: policy-grounded structured case notes with retries and validation."""
import json
import re
from pathlib import Path
from mock_llm import generate
from retrieval import retrieve

SCENARIO_PATH = Path("case_scenarios.json")
MAX_ATTEMPTS = 4
ALLOWED_PRIORITIES = {"low", "medium", "high"}
ALLOWED_TEAMS = {"card_risk", "upi_risk", "customer_care", "merchant_ops"}
ALLOWED_ACTIONS = {"hold_for_review", "block_and_notify"}
REQUIRED_FIELDS = {"summary", "priority", "recommended_team", "recommended_action", "policy_id"}


def build_prompt(txn_id, fraud_probability, amount_inr, policy):
    """Build a grounded prompt with exact machine-readable labels for mock_llm."""
    policy_id = policy["policy_id"]
    return (
        "You are a payment-risk case-note assistant. Use only the policy below and supplied transaction facts. "
        "Return only a JSON object with fields summary, priority, recommended_team, recommended_action, policy_id. "
        "priority must be low, medium, or high; recommended_team must be card_risk, upi_risk, customer_care, or merchant_ops; "
        "recommended_action must be hold_for_review or block_and_notify. Never approve a flagged payment.\n"
        f"POLICY_TITLE: {policy['title']}\nPOLICY_TEXT: {policy['text']}\n"
        f"TXN_ID: {txn_id}\nFRAUD_PROBABILITY: {float(fraud_probability):.4f}\n"
        f"AMOUNT_INR: {float(amount_inr):.2f}\nPOLICY_ID: {policy_id}\n"
    )


def validate(text, expected_policy_id):
    """Parse and validate a model response. Returns (note, error); never trusts raw output."""
    if not isinstance(text, str):
        return None, "response_not_text"
    candidate = text.strip()
    # Reject explanatory prose and accept only a complete JSON object.
    if not candidate.startswith("{") or not candidate.endswith("}"):
        return None, "malformed_json_envelope"
    try:
        note = json.loads(candidate)
    except json.JSONDecodeError:
        return None, "invalid_json"
    if not isinstance(note, dict):
        return None, "not_object"
    if set(note.keys()) != REQUIRED_FIELDS:
        return None, "field_set_mismatch"
    if not isinstance(note["summary"], str) or not note["summary"].strip() or len(note["summary"]) > 500:
        return None, "invalid_summary"
    if note["priority"] not in ALLOWED_PRIORITIES:
        return None, "invalid_priority"
    if note["recommended_team"] not in ALLOWED_TEAMS:
        return None, "invalid_team"
    if note["recommended_action"] not in ALLOWED_ACTIONS:
        return None, "invalid_action"
    if note["policy_id"] != expected_policy_id:
        return None, "policy_mismatch"
    return note, None


def run_case(txn_id, fraud_probability, amount_inr, question):
    """Retrieve a policy and retry defective LLM output; fail closed on outages/errors."""
    if not isinstance(txn_id, str) or not re.fullmatch(r"T\d{6}", txn_id):
        return {"txn_id": str(txn_id), "status": "rejected", "attempts": 0, "policy_id": None, "case_note": None, "error": "invalid_txn_id"}
    try:
        prob, amount = float(fraud_probability), float(amount_inr)
    except (TypeError, ValueError):
        return {"txn_id": txn_id, "status": "rejected", "attempts": 0, "policy_id": None, "case_note": None, "error": "invalid_numeric_input"}
    if not 0 <= prob <= 1 or not 0 < amount <= 1_000_000:
        return {"txn_id": txn_id, "status": "rejected", "attempts": 0, "policy_id": None, "case_note": None, "error": "numeric_input_out_of_range"}

    policy = retrieve(question)
    if policy is None:
        return {"txn_id": txn_id, "status": "no_policy", "attempts": 0, "policy_id": None, "case_note": None, "error": "no_relevant_policy"}
    prompt = build_prompt(txn_id, prob, amount, policy)
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = generate(prompt)
        except ConnectionError:
            return {"txn_id": txn_id, "status": "unavailable", "attempts": attempt, "policy_id": policy["policy_id"], "case_note": None, "error": "llm_unavailable"}
        except Exception:
            last_error = "llm_generation_error"
            continue
        note, error = validate(response, policy["policy_id"])
        if error is None:
            return {"txn_id": txn_id, "status": "held", "attempts": attempt, "policy_id": policy["policy_id"], "policy_score": round(policy["score"], 4), "case_note": note, "error": None}
        last_error = error
    return {"txn_id": txn_id, "status": "needs_manual_fallback", "attempts": MAX_ATTEMPTS, "policy_id": policy["policy_id"], "policy_score": round(policy["score"], 4), "case_note": None, "error": last_error or "validation_failed"}


def main():
    scenarios = json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))
    print("PART C CASE NOTE SCENARIOS")
    print("txn_id | policy | status | attempts | error | summary")
    counts = {}
    for row in scenarios:
        result = run_case(row["txn_id"], row["fraud_probability"], row["amount_inr"], row["question"])
        counts[result["status"]] = counts.get(result["status"], 0) + 1
        summary = result["case_note"]["summary"] if result["case_note"] else "-"
        print(f"{row['txn_id']} | {result.get('policy_id')} | {result['status']} | {result['attempts']} | {result['error']} | {summary}")
    print("status_counts=" + json.dumps(counts, sort_keys=True))
    print(f"max_attempts={MAX_ATTEMPTS}")


if __name__ == "__main__":
    main()
