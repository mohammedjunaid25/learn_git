"""Part D: FastAPI service for fraud triage and policy-grounded case notes."""
import json
import logging
import pickle
import time
from pathlib import Path
from threading import Lock
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict

from casenote import run_case
from train import FEATURES, MODEL_PATH

LOG_PATH = Path("service.log")
logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fraud_triage")
app = FastAPI(title="Merchant Risk Desk Fraud Triage", version="1.0.0")
COUNTERS = {"requests_total": 0, "triage_held": 0, "triage_approved": 0, "validation_errors": 0, "service_errors": 0, "llm_unavailable": 0}
COUNTER_LOCK = Lock()
STARTED_AT = time.time()


class TriageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    txn_id: str = Field(pattern=r"^T\d{6}$")
    fraud_probability: float = Field(ge=0, le=1)
    amount_inr: float = Field(gt=0, le=1_000_000)
    question: str = Field(min_length=5, max_length=500)


class ModelTriageRequest(BaseModel):
    """Optional raw-feature route for scoring cleaned transaction-like records."""
    model_config = ConfigDict(extra="forbid")
    txn_id: str = Field(pattern=r"^T\d{6}$")
    merchant_category: str = Field(min_length=1, max_length=50)
    city: str = Field(min_length=1, max_length=50)
    payment_mode: Literal["UPI", "Card", "Netbanking", "Wallet"]
    amount_inr: float = Field(gt=0, le=1_000_000)
    hour_of_day: int = Field(ge=0, le=23)
    txn_count_last_24h: int = Field(ge=0, le=100)
    avg_amount_30d_inr: float = Field(gt=0, le=1_000_000)
    distance_from_home_km: float = Field(ge=0, le=100_000)
    is_new_device: int = Field(ge=0, le=1)
    account_age_days: int = Field(ge=0, le=100_000)


def _increment(key):
    with COUNTER_LOCK:
        COUNTERS[key] += 1


def _load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError("fraud_model.pkl is missing; run `python prepare.py` and `python train.py` first")
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


@app.middleware("http")
async def request_logging(request: Request, call_next):
    start = time.perf_counter()
    _increment("requests_total")
    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info("method=%s path=%s status=%s duration_ms=%.2f", request.method, request.url.path, response.status_code, duration_ms)
        return response
    except Exception:
        _increment("service_errors")
        logger.exception("unhandled request error path=%s", request.url.path)
        raise


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    _increment("validation_errors")
    logger.warning("validation_error path=%s details=%s", request.url.path, json.dumps(exc.errors(), default=str))
    return JSONResponse(status_code=422, content={"detail": exc.errors(), "error": "request_validation_failed"})


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": MODEL_PATH.exists(), "uptime_seconds": round(time.time() - STARTED_AT, 2)}


@app.get("/metrics")
def metrics():
    with COUNTER_LOCK:
        result = dict(COUNTERS)
    result["uptime_seconds"] = round(time.time() - STARTED_AT, 2)
    return result


@app.post("/triage")
def triage(payload: TriageRequest):
    try:
        result = run_case(payload.txn_id, payload.fraud_probability, payload.amount_inr, payload.question)
        if result["status"] == "unavailable":
            _increment("llm_unavailable")
            logger.error("llm_unavailable txn_id=%s", payload.txn_id)
            return JSONResponse(status_code=503, content=result)
        if result["status"] in {"rejected", "no_policy", "needs_manual_fallback"}:
            _increment("triage_held")
            return JSONResponse(status_code=200, content={**result, "decision": "hold_for_human"})
        _increment("triage_held")
        return {**result, "decision": "hold_for_human"}
    except Exception as exc:
        _increment("service_errors")
        logger.exception("triage_error txn_id=%s", payload.txn_id)
        raise HTTPException(status_code=500, detail="internal_triage_error") from exc


@app.post("/score")
def score(payload: ModelTriageRequest):
    """Score an individual transaction using the trained classifier and alert threshold."""
    try:
        artifact = _load_model()
        row = payload.model_dump()
        amount = float(row["amount_inr"])
        average = float(row["avg_amount_30d_inr"])
        distance = float(row["distance_from_home_km"])
        row.update({
            "amount_to_avg_ratio": amount / max(average, 1.0),
            "amount_log1p": __import__("math").log1p(amount),
            "avg_amount_log1p": __import__("math").log1p(average),
            "distance_log1p": __import__("math").log1p(distance),
            "is_night": int(0 <= row["hour_of_day"] <= 5),
            "high_velocity": int(row["txn_count_last_24h"] > 6),
            "new_account": int(row["account_age_days"] < 30),
        })
        row["merchant_category"] = row["merchant_category"].strip().title()
        row["city"] = row["city"].strip().title()
        frame = pd.DataFrame([{col: row[col] for col in FEATURES}])
        probability = float(artifact["model"].predict_proba(frame)[0, 1])
        decision = "hold_for_human" if probability >= artifact["threshold"] else "approve"
        _increment("triage_held" if decision == "hold_for_human" else "triage_approved")
        return {"txn_id": payload.txn_id, "fraud_probability": round(probability, 6), "alert_threshold": artifact["threshold"], "decision": decision}
    except RuntimeError as exc:
        _increment("service_errors")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        _increment("service_errors")
        logger.exception("score_error txn_id=%s", payload.txn_id)
        raise HTTPException(status_code=500, detail="internal_scoring_error") from exc
