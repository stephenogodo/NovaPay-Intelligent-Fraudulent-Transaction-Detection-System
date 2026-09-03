"""Feature engineering for NovaPay fraud detection.

All thresholds used below were derived from EDA (see notebooks/01_eda.ipynb)
and are centralised as named constants so they can be tuned or A/B tested
without touching pipeline code.
"""
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config

# EDA-derived thresholds
NIGHT_HOUR_START, NIGHT_HOUR_END = 0, 7
ACCOUNT_VERY_NEW_DAYS = 30
ACCOUNT_NEW_DAYS = 90
VELOCITY_BURST_THRESHOLD = 3
AMOUNT_HIGH_USD = 2000
IP_HIGH_RISK_THRESHOLD = 0.7
DEVICE_LOW_TRUST_THRESHOLD = 0.5
CORRIDOR_HIGH_RISK_THRESHOLD = 0.5
HIGH_CHARGEBACK_COUNT = 1


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = df[config.TIMESTAMP_COL]
    df["hour_of_day"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = ts.dt.month
    return df


def add_risk_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Threshold-based binary fraud-signal flags, derived from EDA."""
    df = df.copy()
    hour = df["hour_of_day"]
    df["night_hour"] = ((hour >= NIGHT_HOUR_START) & (hour <= NIGHT_HOUR_END)).astype(int)
    df["account_very_new"] = (df["account_age_days"] < ACCOUNT_VERY_NEW_DAYS).astype(int)
    df["account_new"] = (
        (df["account_age_days"] >= ACCOUNT_VERY_NEW_DAYS)
        & (df["account_age_days"] <= ACCOUNT_NEW_DAYS)
    ).astype(int)
    df["velocity_burst"] = (df["txn_velocity_1h"] >= VELOCITY_BURST_THRESHOLD).astype(int)
    df["amount_high"] = (df["amount_usd"] >= AMOUNT_HIGH_USD).astype(int)
    df["ip_high_risk"] = (df["ip_risk_score"] >= IP_HIGH_RISK_THRESHOLD).astype(int)
    df["device_low_trust"] = (df["device_trust_score"] < DEVICE_LOW_TRUST_THRESHOLD).astype(int)
    df["corridor_high_risk"] = (df["corridor_risk"] >= CORRIDOR_HIGH_RISK_THRESHOLD).astype(int)
    df["cross_border"] = (df["source_currency"] != df["dest_currency"]).astype(int)
    df["high_chargeback_history"] = (
        df["chargeback_history_count"] >= HIGH_CHARGEBACK_COUNT
    ).astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full feature-engineering chain in one call."""
    df = add_time_features(df)
    df = add_risk_flags(df)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Column transformer: OHE for categoricals, passthrough/scale for numerics.

    Fit ONLY on training data at call-site to avoid leakage; this factory
    just defines the shape of the transform.
    """
    ct = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False),
                config.CATEGORICAL_FEATURES,
            ),
            (
                "bool",
                "passthrough",
                config.BOOLEAN_FEATURES,
            ),
            (
                "num",
                StandardScaler(),
                config.NUMERIC_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    # Keep pandas DataFrames (with real column names) flowing all the way
    # through the sklearn Pipeline, so the fitted model always sees named
    # columns -- both to silence sklearn's "no feature names" warning and,
    # more importantly, so SHAP explanations line up with the right labels
    # at inference time without a separate manual reconstruction step.
    ct.set_output(transform="pandas")
    return ct


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Select the final modeling feature set (booleans coerced to int)."""
    X = df[config.ALL_FEATURES].copy()
    for col in config.BOOLEAN_FEATURES:
        X[col] = X[col].astype(int)
    return X
