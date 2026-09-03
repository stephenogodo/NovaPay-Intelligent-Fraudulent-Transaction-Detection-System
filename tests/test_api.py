import pytest
from fastapi.testclient import TestClient

from api.main import app

SAMPLE_TXN = {
    "transaction_id": "txn_test_1",
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


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_metadata(client):
    resp = client.get("/model/metadata")
    assert resp.status_code == 200
    body = resp.json()
    assert "best_model" in body
    assert "decision_threshold" in body


def test_score_single_transaction(client):
    resp = client.post("/score", json=SAMPLE_TXN)
    assert resp.status_code == 200
    body = resp.json()
    assert body["transaction_id"] == "txn_test_1"
    assert 0 <= body["fraud_probability"] <= 1
    assert isinstance(body["is_flagged"], bool)
    assert len(body["reasons"]) == 5
    for r in body["reasons"]:
        assert r["direction"] in ("increases_fraud_risk", "decreases_fraud_risk")


def test_score_rejects_invalid_payload(client):
    bad_txn = dict(SAMPLE_TXN)
    bad_txn["ip_risk_score"] = 5.0  # out of [0,1] bound
    resp = client.post("/score", json=bad_txn)
    assert resp.status_code == 422


def test_batch_scoring(client):
    resp = client.post("/score/batch", json={"transactions": [SAMPLE_TXN, SAMPLE_TXN]})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 2


def test_high_risk_transaction_scores_higher_than_clean_one(client):
    clean_txn = dict(SAMPLE_TXN)
    clean_txn.update({
        "transaction_id": "txn_clean",
        "ip_risk_score": 0.05,
        "device_trust_score": 0.95,
        "location_mismatch": False,
        "new_device": False,
        "chargeback_history_count": 0,
        "account_age_days": 900,
        "amount_usd": 60.0,
        "amount_src": 60.0,
        "txn_velocity_1h": 0,
        "txn_velocity_24h": 1,
        "corridor_risk": 0.05,
        "risk_score_internal": 0.05,
    })
    risky_resp = client.post("/score", json=SAMPLE_TXN).json()
    clean_resp = client.post("/score", json=clean_txn).json()
    assert risky_resp["fraud_probability"] > clean_resp["fraud_probability"]
