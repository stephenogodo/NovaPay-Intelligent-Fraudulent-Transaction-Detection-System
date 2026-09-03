"""NovaPay Fraud Detection — real-time scoring microservice.

Endpoints:
    GET  /health            -- liveness/readiness probe
    POST /score              -- score a single transaction, with SHAP reasons
    POST /score/batch        -- score up to 500 transactions in one call
    GET  /model/metadata     -- model version, threshold, feature list

Run locally:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from novapay_fraud import config, explain, features
from .schemas import (
    BatchScoreRequest, BatchScoreResponse, HealthResponse,
    Reason, ScoreResponse, TransactionRequest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_STATE: dict = {"pipeline": None, "explainer": None, "metadata": None}


def _load_artifacts():
    if not config.MODEL_PATH.exists():
        raise RuntimeError(
            f"Model artifact not found at {config.MODEL_PATH}. "
            "Run `python -m novapay_fraud.train` first."
        )
    pipeline = joblib.load(config.MODEL_PATH)
    explainer = joblib.load(config.SHAP_EXPLAINER_PATH)
    with open(config.METADATA_PATH) as f:
        metadata = json.load(f)
    return pipeline, explainer, metadata


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model artifacts...")
    pipeline, explainer, metadata = _load_artifacts()
    MODEL_STATE["pipeline"] = pipeline
    MODEL_STATE["explainer"] = explainer
    MODEL_STATE["metadata"] = metadata
    logger.info("Loaded model: %s (threshold=%.2f)",
                metadata["best_model"], metadata["decision_threshold"])
    yield
    MODEL_STATE.clear()


app = FastAPI(
    title="NovaPay Fraud Detection API",
    description="Real-time transaction fraud scoring with SHAP explainability.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.2f}"
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error scoring request")
    return JSONResponse(status_code=500, content={"detail": "Internal scoring error"})


def _request_to_row(req: TransactionRequest) -> pd.DataFrame:
    raw = pd.DataFrame([req.model_dump()])
    raw[config.TIMESTAMP_COL] = pd.to_datetime(raw[config.TIMESTAMP_COL], utc=True)
    engineered = features.engineer_features(raw)
    X = features.get_feature_matrix(engineered)
    return X


def _score_row(req: TransactionRequest) -> ScoreResponse:
    pipeline = MODEL_STATE["pipeline"]
    explainer = MODEL_STATE["explainer"]
    metadata = MODEL_STATE["metadata"]

    X = _request_to_row(req)
    proba = float(pipeline.predict_proba(X)[:, 1][0])
    threshold = metadata["decision_threshold"]

    preprocessor = pipeline.named_steps["preprocessor"]
    Xt = preprocessor.transform(X)  # already a named DataFrame (set_output="pandas")
    reasons_raw = explain.explain_instance(explainer, Xt, top_n=5, raw_row=X)

    return ScoreResponse(
        transaction_id=req.transaction_id,
        fraud_probability=round(proba, 4),
        is_flagged=proba >= threshold,
        decision_threshold=threshold,
        model_version=metadata["best_model"],
        reasons=[Reason(**r) for r in reasons_raw],
        scored_at=datetime.now(timezone.utc),
    )


@app.get("/health", response_model=HealthResponse)
def health():
    loaded = MODEL_STATE.get("pipeline") is not None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_version=MODEL_STATE["metadata"]["best_model"] if loaded else None,
    )


@app.get("/model/metadata")
def model_metadata():
    if MODEL_STATE.get("metadata") is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return MODEL_STATE["metadata"]


@app.post("/score", response_model=ScoreResponse)
def score_transaction(req: TransactionRequest):
    if MODEL_STATE.get("pipeline") is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        return _score_row(req)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to score transaction %s", req.transaction_id)
        raise HTTPException(status_code=422, detail=f"Could not score transaction: {exc}")


@app.post("/score/batch", response_model=BatchScoreResponse)
def score_batch(req: BatchScoreRequest):
    if MODEL_STATE.get("pipeline") is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    results = [_score_row(txn) for txn in req.transactions]
    return BatchScoreResponse(results=results)
