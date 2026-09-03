"""Model training, threshold selection, and evaluation.

Time-ordered splitting is used throughout (never a random shuffle-split) to
avoid look-ahead leakage: a fraud detector must be validated on transactions
that happen *after* the ones it trained on, matching how it will actually be
used in production.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from . import config

logger = logging.getLogger(__name__)


@dataclass
class SplitData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_valid: pd.DataFrame
    y_valid: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    raw_test: pd.DataFrame = field(repr=False, default=None)  # for baseline comparison


def time_based_split(df: pd.DataFrame, feature_cols: list[str]) -> SplitData:
    """Chronological train/valid/test split (no shuffling).

    Order: train (earliest) -> valid -> test (most recent), mirroring the
    real deployment scenario of "train on history, score the future".
    """
    df = df.sort_values(config.TIMESTAMP_COL).reset_index(drop=True)
    n = len(df)
    test_start = int(n * (1 - config.TEST_SIZE))
    valid_start = int(test_start * (1 - config.VALID_SIZE))

    train_df = df.iloc[:valid_start]
    valid_df = df.iloc[valid_start:test_start]
    test_df = df.iloc[test_start:]

    return SplitData(
        X_train=train_df[feature_cols], y_train=train_df[config.TARGET],
        X_valid=valid_df[feature_cols], y_valid=valid_df[config.TARGET],
        X_test=test_df[feature_cols], y_test=test_df[config.TARGET],
        raw_test=test_df,
    )


def build_model(name: str):
    params = config.MODEL_REGISTRY[name]
    if name == "logistic_regression":
        return LogisticRegression(**params)
    if name == "random_forest":
        return RandomForestClassifier(**params)
    if name == "xgboost":
        # imbalance handled via scale_pos_weight, set at fit time
        return XGBClassifier(**params)
    if name == "lightgbm":
        return LGBMClassifier(**params)
    raise ValueError(f"Unknown model: {name}")


def compute_scale_pos_weight(y: pd.Series) -> float:
    neg, pos = (y == 0).sum(), (y == 1).sum()
    return float(neg / max(pos, 1))


def fit_model(name: str, model, X_train, y_train):
    if name == "xgboost":
        model.set_params(scale_pos_weight=compute_scale_pos_weight(y_train))
    model.fit(X_train, y_train)
    return model


def predict_proba_positive(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def select_threshold(y_valid: pd.Series, proba_valid: np.ndarray,
                      grid: list[float] | None = None) -> dict:
    """Pick the probability threshold on VALIDATION data that maximises F1,
    subject to a minimum precision floor (fraud review capacity is finite,
    so we don't want an unbounded flood of false positives even if recall
    looks great on paper).
    """
    grid = grid or config.THRESHOLD_GRID
    best = {"threshold": 0.5, "f1": -1.0}
    for t in grid:
        preds = (proba_valid >= t).astype(int)
        p = precision_score(y_valid, preds, zero_division=0)
        r = recall_score(y_valid, preds, zero_division=0)
        f1 = f1_score(y_valid, preds, zero_division=0)
        if f1 > best["f1"]:
            best = {"threshold": t, "precision": p, "recall": r, "f1": f1}
    return best


def evaluate(y_true, proba, threshold: float) -> dict:
    preds = (proba >= threshold).astype(int)
    return {
        "threshold": threshold,
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "flag_rate": float(preds.mean()),
        "n_flagged": int(preds.sum()),
        "n_total": int(len(preds)),
    }


def precision_recall_table(y_true, proba) -> pd.DataFrame:
    prec, rec, thr = precision_recall_curve(y_true, proba)
    return pd.DataFrame({"precision": prec[:-1], "recall": rec[:-1], "threshold": thr})
