import pandas as pd
import pytest

from novapay_fraud import config
from novapay_fraud.monitoring.drift_check import check_drift


@pytest.fixture
def reference():
    return pd.read_parquet(config.REFERENCE_DATA_PATH)


def test_no_drift_alerts_on_a_resampled_reference_window(reference):
    """Sampling from the SAME distribution should not trigger alerts --
    guards against an overly-sensitive detector that cries wolf on normal
    day-to-day sampling variation.
    """
    current = reference.sample(min(300, len(reference)), random_state=42)
    result = check_drift(reference, current)
    assert result.alerts == []


def test_severe_synthetic_drift_triggers_alerts(reference):
    current = reference.sample(min(300, len(reference)), random_state=3).copy()
    for col in ["amount_usd", "amount_src", "fee", "ip_risk_score",
                "device_trust_score", "account_age_days", "txn_velocity_1h",
                "txn_velocity_24h", "corridor_risk", "risk_score_internal",
                "exchange_rate_src_to_dest", "chargeback_history_count"]:
        if col in current.columns:
            current[col] = current[col] * 6.0
    current["fraud_probability"] = (current["fraud_probability"] * 4).clip(0, 1)

    result = check_drift(reference, current)
    assert len(result.alerts) > 0
    assert result.drifted_feature_count >= 5
    # at severe, broad drift we expect at least the prediction-drift or
    # feature-drift alert to fire (exact boundary crossing for the 30%
    # feature-share alert can vary slightly with the random sample, but a
    # shift this large must trip at least one detector)
    assert any(kind in a for a in result.alerts for kind in ("FEATURE_DRIFT", "PREDICTION_DRIFT", "FLAG_RATE_SHIFT"))


def test_drift_result_serializes_cleanly(reference):
    current = reference.sample(min(50, len(reference)), random_state=1)
    result = check_drift(reference, current)
    d = result.to_dict()
    assert d["n_reference"] == len(reference)
    assert d["n_current"] == len(current)
    assert isinstance(d["alerts"], list)
