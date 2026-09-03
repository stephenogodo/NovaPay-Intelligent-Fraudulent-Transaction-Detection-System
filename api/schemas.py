"""Request/response schemas for the fraud-scoring API."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TransactionRequest(BaseModel):
    """Raw transaction payload — mirrors the source system's fields.

    Engineered features (time-of-day, risk flags) are computed server-side
    so callers never need to know the model's internal feature engineering.
    """

    transaction_id: str = Field(..., description="Unique transaction identifier")
    timestamp: datetime = Field(..., description="Transaction timestamp, ISO 8601")

    home_country: Literal["US", "CA", "UK"]
    source_currency: str
    dest_currency: str
    channel: Literal["ATM", "WEB", "MOBILE"]
    kyc_tier: Literal["LOW", "STANDARD", "ENHANCED"]
    ip_country: Literal["US", "CA", "UK"]

    amount_src: float = Field(..., ge=0)
    amount_usd: float = Field(..., ge=0)
    fee: float = Field(..., ge=0)
    exchange_rate_src_to_dest: float = Field(..., gt=0)

    new_device: bool
    location_mismatch: bool

    ip_risk_score: float = Field(..., ge=0, le=1)
    account_age_days: int = Field(..., ge=0)
    device_trust_score: float = Field(..., ge=0, le=1)
    chargeback_history_count: int = Field(..., ge=0)
    risk_score_internal: float = Field(..., ge=0, le=1)
    txn_velocity_1h: int = Field(..., ge=0)
    txn_velocity_24h: int = Field(..., ge=0)
    corridor_risk: float = Field(..., ge=0, le=1)

    @field_validator("amount_usd")
    @classmethod
    def sanity_check_amount(cls, v, info):
        if v > 1_000_000:
            raise ValueError("amount_usd exceeds sane upper bound for this service")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "transaction_id": "txn_12345",
                "timestamp": "2026-09-03T02:14:00Z",
                "home_country": "US",
                "source_currency": "USD",
                "dest_currency": "MXN",
                "channel": "WEB",
                "kyc_tier": "LOW",
                "ip_country": "CA",
                "amount_src": 3200.0,
                "amount_usd": 3200.0,
                "fee": 12.5,
                "exchange_rate_src_to_dest": 17.1,
                "new_device": True,
                "location_mismatch": True,
                "ip_risk_score": 0.82,
                "account_age_days": 12,
                "device_trust_score": 0.21,
                "chargeback_history_count": 1,
                "risk_score_internal": 0.7,
                "txn_velocity_1h": 4,
                "txn_velocity_24h": 9,
                "corridor_risk": 0.55,
            }
        }
    }


class Reason(BaseModel):
    feature: str
    value: float | int | bool | str | None
    shap_contribution: float
    direction: Literal["increases_fraud_risk", "decreases_fraud_risk"]


class ScoreResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_flagged: bool
    decision_threshold: float
    model_version: str
    reasons: list[Reason]
    scored_at: datetime


class BatchScoreRequest(BaseModel):
    transactions: list[TransactionRequest] = Field(..., min_length=1, max_length=500)


class BatchScoreResponse(BaseModel):
    results: list[ScoreResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
