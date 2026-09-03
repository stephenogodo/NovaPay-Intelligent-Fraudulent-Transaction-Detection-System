import pandas as pd

from novapay_fraud import baseline


def test_baseline_flags_high_risk_transactions():
    df = pd.DataFrame({
        "ip_risk_score": [0.95, 0.1, 0.1],
        "amount_usd": [100, 6000, 100],
        "chargeback_history_count": [0, 0, 3],
    })
    out = baseline.score(df)
    assert list(out) == [1, 1, 1]


def test_baseline_does_not_flag_clean_transaction():
    df = pd.DataFrame({
        "ip_risk_score": [0.1],
        "amount_usd": [50],
        "chargeback_history_count": [0],
    })
    out = baseline.score(df)
    assert list(out) == [0]
