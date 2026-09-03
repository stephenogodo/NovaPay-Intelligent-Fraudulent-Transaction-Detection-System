"""Central configuration for the NovaPay fraud detection pipeline.

Keeping every path, column list, and hyperparameter default in one place
means the notebook, training scripts, API, and tests can never silently
drift out of sync with each other (the failure mode this project is
explicitly designed to avoid).
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
RAW_DATA_PATH = DATA_DIR / "nova_pay_combined.csv"

MODEL_PATH = ARTIFACTS_DIR / "fraud_model.joblib"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
SHAP_EXPLAINER_PATH = ARTIFACTS_DIR / "shap_explainer.joblib"
REFERENCE_DATA_PATH = ARTIFACTS_DIR / "reference_data.parquet"
FEATURE_SCHEMA_PATH = ARTIFACTS_DIR / "feature_schema.json"

# --------------------------------------------------------------------------
# Target & identifiers
# --------------------------------------------------------------------------
TARGET = "is_fraud"
ID_COLS = ["transaction_id", "customer_id", "device_id", "ip_address"]
TIMESTAMP_COL = "timestamp"

# --------------------------------------------------------------------------
# Raw schema (as it exists in the source CSV)
# --------------------------------------------------------------------------
RAW_CATEGORICAL = [
    "home_country", "source_currency", "dest_currency",
    "channel", "kyc_tier", "ip_country",
]
RAW_BOOLEAN = ["new_device", "location_mismatch"]
RAW_NUMERIC = [
    "amount_src", "amount_usd", "fee", "exchange_rate_src_to_dest",
    "ip_risk_score", "account_age_days", "device_trust_score",
    "chargeback_history_count", "risk_score_internal",
    "txn_velocity_1h", "txn_velocity_24h", "corridor_risk",
]

# Category-normalisation maps for known dirty-data variants (case/whitespace
# typos observed during profiling). Centralised so cleaning is deterministic
# and testable rather than embedded as one-off notebook cells.
CATEGORY_NORMALIZATION = {
    "channel": {
        "atm": "ATM", "web": "WEB", "weeb": "WEB", "mobile": "MOBILE",
        "mobille": "MOBILE", "atm ": "ATM",
    },
    "kyc_tier": {
        "standard": "STANDARD", "standrd": "STANDARD",
        "enhanced": "ENHANCED", "enhancd": "ENHANCED",
        "low": "LOW",
    },
    "home_country": {"us": "US", "ca": "CA", "uk": "UK"},
    "ip_country": {"us": "US", "ca": "CA", "uk": "UK"},
}
UNKNOWN_TOKENS = {"unknown", "nan", "none", "null", "", "n/a"}

# --------------------------------------------------------------------------
# Engineered features
# --------------------------------------------------------------------------
ENGINEERED_BINARY_FLAGS = [
    "night_hour", "account_very_new", "account_new", "velocity_burst",
    "amount_high", "ip_high_risk", "device_low_trust",
    "corridor_high_risk", "cross_border", "high_chargeback_history",
]
TIME_FEATURES = ["hour_of_day", "day_of_week", "is_weekend", "month"]

CATEGORICAL_FEATURES = RAW_CATEGORICAL
NUMERIC_FEATURES = RAW_NUMERIC + TIME_FEATURES + ENGINEERED_BINARY_FLAGS
BOOLEAN_FEATURES = RAW_BOOLEAN

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES + BOOLEAN_FEATURES

# --------------------------------------------------------------------------
# Modeling
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALID_SIZE = 0.15  # carved out of the training window for early stopping / tuning

# Business-driven decision threshold sweep (fraud review is expensive, but
# missed fraud is worse) -- default operating point is picked on validation
# data, not hard-coded. Coarse 0.05 steps across the low/mid range (where
# fine resolution rarely changes the F1-maximizing choice) plus fine 0.01
# steps from 0.70 to 0.99 -- verified by direct sweep that this model's
# actual F1 peak sits at 0.91, well past where an earlier, narrower grid
# (capped at 0.70) had artificially clipped the search at its own boundary.
THRESHOLD_GRID = (
    [round(x, 2) for x in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                            0.55, 0.6, 0.65]]
    + [round(0.70 + 0.01 * i, 2) for i in range(30)]  # 0.70, 0.71, ..., 0.99
)

# Minimum recall the business requires the deployed model to hit on the
# held-out test set relative to a naive rules baseline (see baseline.py).
MIN_RECALL_UPLIFT_PCT = 15.0

MODEL_REGISTRY = {
    "logistic_regression": {
        "class_weight": "balanced",
        "max_iter": 2000,
        "random_state": RANDOM_STATE,
    },
    "random_forest": {
        "n_estimators": 400,
        "class_weight": "balanced_subsample",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
    "xgboost": {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "aucpr",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
    "lightgbm": {
        "n_estimators": 400,
        "max_depth": -1,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    },
}
