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
    raw_valid: pd.DataFrame = field(repr=False, default=None)  # for validation-time baseline comparison
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
        raw_valid=valid_df, raw_test=test_df,
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
                      grid: list[float] | None = None,
                      min_recall: float | None = None) -> dict:
    """Pick the probability threshold on VALIDATION data.

    If `min_recall` is given, this is a CONSTRAINED search: among grid
    thresholds whose recall clears `min_recall`, pick the one with the
    highest precision (ties broken by F1). Only if no threshold clears the
    floor does it fall back to the unconstrained best-F1 choice.

    This constraint exists because pure F1-maximization and the business's
    actual requirement are NOT the same objective, and treating them as
    interchangeable is a real bug this project shipped once already: an
    F1-only search pushed Logistic Regression's threshold to 0.91 (highest
    F1 of any threshold), which pushed precision to 99.6% but let recall
    drop from 93.8% to 92.2% -- just enough to fall from a 16.5% recall
    uplift over the rules baseline to 14.5%, missing the required >=15%
    floor. F1 does not know that this system's constraint is asymmetric
    (a missed fraud case is treated as worse than a false-positive review,
    up to the required recall floor) -- only a recall-constrained search
    does. `min_recall` should be the validation-window rules-baseline
    recall scaled up by the required uplift, so the SAME floor the model
    will later be checked against in `select_best_model` is respected
    at the point the threshold is actually chosen, not discovered to have
    been missed only after the fact.
    """
    grid = grid or config.THRESHOLD_GRID

    def _score(t: float) -> dict:
        preds = (proba_valid >= t).astype(int)
        return {
            "threshold": t,
            "precision": precision_score(y_valid, preds, zero_division=0),
            "recall": recall_score(y_valid, preds, zero_division=0),
            "f1": f1_score(y_valid, preds, zero_division=0),
        }

    scored = [_score(t) for t in grid]

    if min_recall is not None:
        passing = [s for s in scored if s["recall"] >= min_recall]
        if passing:
            best = max(passing, key=lambda s: (s["precision"], s["f1"]))
            best["recall_floor_met"] = True
            best["at_grid_boundary"] = best["threshold"] in (min(grid), max(grid))
            return best
        logger.warning(
            "select_threshold: no grid threshold reaches the required "
            "min_recall=%.4f on validation data (best available recall "
            "was %.4f) -- falling back to the unconstrained best-F1 "
            "threshold, which will NOT meet the recall requirement.",
            min_recall, max(s["recall"] for s in scored),
        )

    best = max(scored, key=lambda s: s["f1"])
    best["recall_floor_met"] = min_recall is None
    if best["threshold"] in (min(grid), max(grid)):
        logger.warning(
            "select_threshold: winning threshold %.2f sits on the search "
            "grid's boundary (range %.2f-%.2f) -- this usually means the "
            "true F1-maximizing threshold lies outside the searched range "
            "and the grid should be widened before trusting this result.",
            best["threshold"], min(grid), max(grid),
        )
        best["at_grid_boundary"] = True
    else:
        best["at_grid_boundary"] = False
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


@dataclass
class SelectionResult:
    selected_model: str
    reason: str
    passed_gate: list[str]
    failed_gate: list[str]
    recall_uplift_pct: dict  # model_name -> uplift percentage, for every candidate


def select_best_model(
    candidate_metrics: dict,
    baseline_recall: float,
    min_recall_uplift_pct: float,
    primary_metric: str = "pr_auc",
) -> SelectionResult:
    """Automatically select the best candidate model, gated on the actual
    business requirement rather than a single metric picked in isolation.

    Selecting purely by PR-AUC (or any single statistical metric) can pick
    a model that scores well in isolation but fails the requirement that
    actually matters here: recall must beat the rules-based baseline by at
    least `min_recall_uplift_pct`. This function makes that check part of
    selection itself, not something read off a report afterward:

      1. Compute each candidate's recall uplift over the rules baseline.
      2. Filter to candidates that meet the minimum uplift requirement.
      3. Among those that pass, select the one with the best `primary_metric`
         (PR-AUC by default -- the appropriate ranking metric for an
         imbalanced classification problem).
      4. If NO candidate passes the gate, fall back to the best
         `primary_metric` among all candidates, but flag this clearly in the
         returned reason -- an automatic pipeline should never silently
         promote a model that fails the stated requirement without saying so.

    candidate_metrics: {model_name: metrics_dict} where metrics_dict has at
        least "recall" and `primary_metric` keys (as produced by `evaluate`).
    """
    uplift = {
        name: (m["recall"] - baseline_recall) / max(baseline_recall, 1e-9) * 100
        for name, m in candidate_metrics.items()
    }
    passed = [name for name, u in uplift.items() if u >= min_recall_uplift_pct]
    failed = [name for name in candidate_metrics if name not in passed]

    if passed:
        selected = max(passed, key=lambda n: candidate_metrics[n][primary_metric])
        reason = (
            f"'{selected}' selected: highest {primary_metric.upper()} "
            f"({candidate_metrics[selected][primary_metric]:.3f}) among the "
            f"{len(passed)}/{len(candidate_metrics)} candidate(s) that met the "
            f"minimum {min_recall_uplift_pct:.0f}% recall-uplift requirement "
            f"(achieved {uplift[selected]:.1f}%)."
        )
    else:
        selected = max(candidate_metrics, key=lambda n: candidate_metrics[n][primary_metric])
        reason = (
            f"WARNING: no candidate met the minimum {min_recall_uplift_pct:.0f}% "
            f"recall-uplift requirement (best achieved was "
            f"{max(uplift.values()):.1f}%). Falling back to '{selected}', the "
            f"highest-{primary_metric.upper()} candidate overall "
            f"({candidate_metrics[selected][primary_metric]:.3f}), but this "
            f"selection does NOT meet the stated business requirement and "
            f"should not be promoted without review."
        )

    return SelectionResult(
        selected_model=selected,
        reason=reason,
        passed_gate=passed,
        failed_gate=failed,
        recall_uplift_pct=uplift,
    )
