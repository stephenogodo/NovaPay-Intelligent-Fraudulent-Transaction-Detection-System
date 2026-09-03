"""A simple static rules-based fraud flagger, standing in for NovaPay's
legacy system, so the ML models' recall improvement can be measured against
an actual baseline instead of an assumed one.

Rules are intentionally simple thresholds an ops team could hand-write and
maintain without ML — this is what "static rules-based approach" in the
project brief means in practice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def score(df: pd.DataFrame) -> np.ndarray:
    """Flag a transaction as fraud (1) if ANY hand-written rule fires.

    Deliberately kept to simple, single-signal, fixed-threshold checks --
    this mirrors what a real static rules engine looks like (hard limits an
    ops team configured and rarely revisits), NOT a hand-tuned combination
    of behavioral signals. Compound/velocity-aware logic is exactly the kind
    of adaptive detection the ML system is being built to add, so folding it
    into the "baseline" would understate the legacy system's real limitation
    and make the recall-uplift comparison meaningless.
      - very high IP risk score (single hard threshold)
      - large transfer amount (single hard threshold)
      - repeat chargeback history (single hard threshold)
    """
    r1 = df["ip_risk_score"] >= 0.9
    r2 = df["amount_usd"] >= 5000
    r3 = df["chargeback_history_count"] >= 2
    flagged = r1 | r2 | r3
    return flagged.astype(int).to_numpy()


def evaluate(df: pd.DataFrame, y_true: pd.Series) -> dict:
    from sklearn.metrics import precision_score, recall_score, f1_score

    y_pred = score(df)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "flag_rate": float(y_pred.mean()),
    }
