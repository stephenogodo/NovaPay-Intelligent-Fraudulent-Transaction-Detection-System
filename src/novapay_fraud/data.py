"""Data loading, profiling, and cleaning for the NovaPay fraud dataset.

Design principle: every cleaning decision is a function with a docstring
explaining *why*, and every function is unit-testable in isolation
(see tests/test_data.py). Nothing here is a notebook cell copy-paste.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


def load_raw(path=None) -> pd.DataFrame:
    """Load the raw NovaPay CSV exactly as delivered (no cleaning)."""
    path = path or config.RAW_DATA_PATH
    df = pd.read_csv(path)
    logger.info("Loaded raw data: %s rows, %s cols", *df.shape)
    return df


def profile(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact data-quality profile: dtype, missingness, cardinality."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(),
    }).sort_values("missing_pct", ascending=False)


def _normalize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Fix whitespace/case typos and collapse unknown-tokens to real NaN.

    Source data contains variants like ' web  ', 'ATm', 'enhancd', 'unknown'
    etc. We normalise on a lower-cased, stripped key so new typo variants of
    an already-known category are still caught, and route anything left
    unrecognised (e.g. genuinely unseen categories) to NaN rather than
    silently keeping a dirty string that a one-hot encoder would treat as
    its own spurious category.
    """
    df = df.copy()
    for col in config.RAW_CATEGORICAL:
        if col not in df.columns:
            continue
        cleaned = df[col].astype(str).str.strip().str.lower()
        cleaned = cleaned.where(~cleaned.isin(config.UNKNOWN_TOKENS), np.nan)
        norm_map = config.CATEGORY_NORMALIZATION.get(col, {})
        mapped = cleaned.map(norm_map)
        # keep already-clean values (e.g. properly-cased) if no mapping hit
        df[col] = mapped.fillna(cleaned.str.upper()).where(cleaned.notna(), np.nan)
    return df


def _clean_numeric_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Strip thousands-separators from numeric columns stored as text."""
    df = df.copy()
    if "amount_src" in df.columns:
        df["amount_src"] = (
            df["amount_src"].astype(str).str.replace(",", "", regex=False)
        )
        df["amount_src"] = pd.to_numeric(df["amount_src"], errors="coerce")
    return df


def _parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[config.TIMESTAMP_COL] = pd.to_datetime(
        df[config.TIMESTAMP_COL], errors="coerce", utc=True
    )
    return df


def resolve_duplicate_transaction_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Investigate and resolve rows sharing a transaction_id.

    A duplicate transaction_id is NOT automatically a duplicate row: it can
    also indicate a genuine retry/re-send with a different outcome. We only
    drop rows that are *fully* identical across every column (a true
    duplicate record); partial duplicates (same id, different payload) are
    kept and flagged, since silently dropping them could remove real fraud
    signal or create leakage the other way.
    """
    report = {}
    full_dupe_mask = df.duplicated(keep="first")
    report["full_duplicate_rows"] = int(full_dupe_mask.sum())

    id_dupe_mask = df.duplicated(subset=["transaction_id"], keep=False)
    report["rows_sharing_a_transaction_id"] = int(id_dupe_mask.sum())

    partial = id_dupe_mask & ~df.duplicated(keep=False)
    report["partial_duplicates_kept"] = int(partial.sum())

    if report["full_duplicate_rows"]:
        fraud_rate_in_dupes = df.loc[full_dupe_mask, config.TARGET].mean()
        fraud_rate_overall = df[config.TARGET].mean()
        report["fraud_rate_in_dropped_dupes"] = float(fraud_rate_in_dupes)
        report["fraud_rate_overall_before_drop"] = float(fraud_rate_overall)

    df_clean = df.loc[~full_dupe_mask].reset_index(drop=True)
    return df_clean, report


def analyze_missingness_vs_target(df: pd.DataFrame) -> pd.DataFrame:
    """Compare fraud rate for rows with vs without missing data.

    Dropping rows with missing values is only safe if missingness is not
    correlated with the target. This makes that assumption explicit and
    checkable instead of assumed.
    """
    has_missing = df.isna().any(axis=1)
    out = df.groupby(has_missing)[config.TARGET].agg(["mean", "count"])
    out.index = out.index.map({True: "has_missing", False: "complete"})
    out.columns = ["fraud_rate", "n_rows"]
    return out


def clean(df: pd.DataFrame, drop_incomplete_rows: bool = True) -> tuple[pd.DataFrame, dict]:
    """Full cleaning pipeline. Returns (clean_df, report) for auditability."""
    report: dict = {"input_rows": len(df)}

    df = _clean_numeric_strings(df)
    df = _normalize_categoricals(df)
    df = _parse_timestamp(df)

    missingness_check = analyze_missingness_vs_target(df)
    report["missingness_vs_target"] = missingness_check.to_dict()

    df, dupe_report = resolve_duplicate_transaction_ids(df)
    report["duplicates"] = dupe_report

    report["missing_before_drop"] = df.isna().sum().to_dict()
    if drop_incomplete_rows:
        before = len(df)
        df = df.dropna(subset=[c for c in df.columns if c != config.TARGET])
        report["rows_dropped_for_missing"] = before - len(df)
    df = df.reset_index(drop=True)

    report["output_rows"] = len(df)
    report["output_fraud_rate"] = float(df[config.TARGET].mean())
    return df, report
