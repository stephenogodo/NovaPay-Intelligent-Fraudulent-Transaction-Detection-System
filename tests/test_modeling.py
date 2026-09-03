import numpy as np
import pandas as pd
import pytest

from novapay_fraud import modeling, config


def test_time_based_split_is_chronological_and_non_overlapping():
    n = 1000
    df = pd.DataFrame({
        config.TIMESTAMP_COL: pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        config.TARGET: np.random.RandomState(0).binomial(1, 0.1, n),
        "f1": np.random.RandomState(1).rand(n),
    })
    split = modeling.time_based_split(df, ["f1"])

    assert len(split.X_train) + len(split.X_valid) + len(split.X_test) == n
    train_end = split.raw_test[config.TIMESTAMP_COL].min()
    # every test timestamp must be >= every train timestamp (chronological)
    assert split.raw_test[config.TIMESTAMP_COL].min() >= df.iloc[:len(split.X_train)][config.TIMESTAMP_COL].max()


def test_compute_scale_pos_weight():
    y = pd.Series([0, 0, 0, 0, 1])
    w = modeling.compute_scale_pos_weight(y)
    assert w == pytest.approx(4.0)


def test_select_threshold_returns_valid_grid_value():
    y = pd.Series([0, 1, 0, 1, 1, 0, 0, 1])
    proba = np.array([0.1, 0.9, 0.2, 0.8, 0.6, 0.3, 0.4, 0.95])
    result = modeling.select_threshold(y, proba)
    assert result["threshold"] in config.THRESHOLD_GRID
    assert 0 <= result["precision"] <= 1
    assert 0 <= result["recall"] <= 1


def test_evaluate_metrics_are_bounded():
    y = pd.Series([0, 1, 0, 1, 1, 0, 0, 1])
    proba = np.array([0.1, 0.9, 0.2, 0.8, 0.6, 0.3, 0.4, 0.95])
    m = modeling.evaluate(y, proba, threshold=0.5)
    for key in ["precision", "recall", "f1", "roc_auc", "pr_auc"]:
        assert 0 <= m[key] <= 1
    assert m["n_total"] == 8
