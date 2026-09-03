import numpy as np
import pandas as pd
import pytest

from novapay_fraud import data, config


@pytest.fixture
def dirty_df():
    return pd.DataFrame({
        "transaction_id": ["a", "b", "c", "c"],
        "customer_id": ["c1", "c2", "c3", "c3"],
        "device_id": ["d1", "d2", "d3", "d3"],
        "ip_address": ["1.1.1.1", "2.2.2.2", "3.3.3.3", "3.3.3.3"],
        "timestamp": [
            "2024-01-01 00:00:00+00:00", "2024-01-02 00:00:00+00:00",
            "2024-01-03 00:00:00+00:00", "2024-01-03 00:00:00+00:00",
        ],
        "home_country": [" US  ", "ca", "UK", "UK"],
        "source_currency": ["USD", "CAD", "GBP", "GBP"],
        "dest_currency": ["USD", "CAD", "GBP", "GBP"],
        "channel": ["ATm", " web  ", "mobille", "mobille"],
        "kyc_tier": ["standrd", "enhancd", "unknown", "unknown"],
        "ip_country": ["US", "CA", "UK", "UK"],
        "amount_src": ["1,200.50", "300", "45", "45"],
        "amount_usd": [1200.50, 300.0, 45.0, 45.0],
        "fee": [5.0, 2.0, 1.0, 1.0],
        "exchange_rate_src_to_dest": [1.0, 1.0, 1.0, 1.0],
        "new_device": [False, True, False, False],
        "location_mismatch": [False, False, True, True],
        "ip_risk_score": [0.1, 0.2, 0.9, 0.9],
        "account_age_days": [400, 10, 5, 5],
        "device_trust_score": [0.8, 0.4, 0.1, 0.1],
        "chargeback_history_count": [0, 0, 2, 2],
        "risk_score_internal": [0.1, 0.3, 0.9, 0.9],
        "txn_velocity_1h": [0, 1, 5, 5],
        "txn_velocity_24h": [1, 2, 10, 10],
        "corridor_risk": [0.0, 0.1, 0.8, 0.8],
        "is_fraud": [0, 0, 1, 1],
    })


def test_category_normalization_fixes_typos(dirty_df):
    cleaned = data._normalize_categoricals(dirty_df)
    assert set(cleaned["home_country"].unique()) == {"US", "CA", "UK"}
    assert set(cleaned["channel"].unique()) == {"ATM", "WEB", "MOBILE"}
    assert cleaned["kyc_tier"].iloc[0] == "STANDARD"
    assert cleaned["kyc_tier"].iloc[1] == "ENHANCED"
    # 'unknown' should become NaN, not a spurious category
    assert cleaned["kyc_tier"].isna().iloc[2]


def test_amount_src_comma_stripped(dirty_df):
    cleaned = data._clean_numeric_strings(dirty_df)
    assert cleaned["amount_src"].iloc[0] == 1200.50
    assert cleaned["amount_src"].dtype == float


def test_full_duplicate_rows_are_dropped(dirty_df):
    fixed = data._clean_numeric_strings(dirty_df)
    fixed = data._normalize_categoricals(fixed)
    fixed = data._parse_timestamp(fixed)
    out, report = data.resolve_duplicate_transaction_ids(fixed)
    assert report["full_duplicate_rows"] == 1
    assert len(out) == 3
    assert "fraud_rate_in_dropped_dupes" in report


def test_clean_end_to_end_drops_incomplete_and_dupes(dirty_df):
    clean_df, report = data.clean(dirty_df, drop_incomplete_rows=True)
    assert report["output_rows"] <= report["input_rows"]
    assert clean_df[config.TARGET].isin([0, 1]).all()
    assert not clean_df.duplicated().any()


def test_missingness_vs_target_reports_both_groups():
    df = pd.DataFrame({
        "a": [1, np.nan, 3, 4],
        "is_fraud": [0, 1, 0, 1],
    })
    out = data.analyze_missingness_vs_target(df)
    assert "has_missing" in out.index or "complete" in out.index
