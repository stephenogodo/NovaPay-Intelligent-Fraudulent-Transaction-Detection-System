import pandas as pd

from novapay_fraud import features, config


def _base_df():
    return pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2024-01-06 03:00:00+00:00",  # Saturday, night hour
            "2024-01-08 14:00:00+00:00",  # Monday, daytime
        ], utc=True),
        "account_age_days": [10, 200],
        "txn_velocity_1h": [5, 0],
        "amount_usd": [3000, 50],
        "ip_risk_score": [0.9, 0.1],
        "device_trust_score": [0.2, 0.9],
        "corridor_risk": [0.6, 0.1],
        "source_currency": ["USD", "USD"],
        "dest_currency": ["MXN", "USD"],
        "chargeback_history_count": [2, 0],
    })


def test_time_features():
    df = features.add_time_features(_base_df())
    assert df["hour_of_day"].tolist() == [3, 14]
    assert df["is_weekend"].tolist() == [1, 0]


def test_risk_flags_thresholds():
    df = features.add_time_features(_base_df())
    df = features.add_risk_flags(df)
    row0 = df.iloc[0]
    row1 = df.iloc[1]

    assert row0["night_hour"] == 1
    assert row0["account_very_new"] == 1
    assert row0["velocity_burst"] == 1
    assert row0["amount_high"] == 1
    assert row0["ip_high_risk"] == 1
    assert row0["device_low_trust"] == 1
    assert row0["cross_border"] == 1
    assert row0["high_chargeback_history"] == 1

    assert row1["night_hour"] == 0
    assert row1["account_very_new"] == 0
    assert row1["velocity_burst"] == 0
    assert row1["amount_high"] == 0
    assert row1["cross_border"] == 0


def test_get_feature_matrix_selects_expected_columns():
    df = _base_df()
    df = features.engineer_features(df)
    for col in ["home_country", "channel", "kyc_tier", "ip_country",
                "new_device", "location_mismatch", "exchange_rate_src_to_dest",
                "fee", "amount_src", "risk_score_internal", "txn_velocity_24h"]:
        df[col] = 0
    X = features.get_feature_matrix(df)
    assert list(X.columns) == config.ALL_FEATURES
    assert X.shape[0] == 2


def test_preprocessor_shapes_are_consistent():
    df = _base_df()
    df = features.engineer_features(df)
    for col in ["home_country", "channel", "kyc_tier", "ip_country",
                "new_device", "location_mismatch", "exchange_rate_src_to_dest",
                "fee", "amount_src", "risk_score_internal", "txn_velocity_24h"]:
        df[col] = "A" if col in config.CATEGORICAL_FEATURES else (
            False if col in config.BOOLEAN_FEATURES else 0.0
        )
    X = features.get_feature_matrix(df)
    pre = features.build_preprocessor()
    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == 2
    assert Xt.shape[1] == len(pre.get_feature_names_out())
