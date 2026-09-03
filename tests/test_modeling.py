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


def test_select_best_model_picks_highest_pr_auc_among_those_passing_gate():
    # model_a has the best PR-AUC but fails the recall-uplift gate;
    # model_b has a lower PR-AUC but is the only one that clears the bar.
    candidates = {
        "model_a": {"recall": 0.85, "pr_auc": 0.99},
        "model_b": {"recall": 0.95, "pr_auc": 0.90},
    }
    result = modeling.select_best_model(candidates, baseline_recall=0.80, min_recall_uplift_pct=15.0)
    assert result.selected_model == "model_b"
    assert "model_b" in result.passed_gate
    assert "model_a" in result.failed_gate
    assert result.recall_uplift_pct["model_a"] == pytest.approx(6.25, abs=0.01)
    assert result.recall_uplift_pct["model_b"] == pytest.approx(18.75, abs=0.01)


def test_select_best_model_ranks_by_pr_auc_among_multiple_passers():
    candidates = {
        "model_a": {"recall": 0.95, "pr_auc": 0.90},
        "model_b": {"recall": 0.96, "pr_auc": 0.95},
    }
    result = modeling.select_best_model(candidates, baseline_recall=0.80, min_recall_uplift_pct=15.0)
    assert result.selected_model == "model_b"
    assert set(result.passed_gate) == {"model_a", "model_b"}
    assert result.failed_gate == []


def test_select_best_model_falls_back_and_warns_when_nothing_passes_gate():
    candidates = {
        "model_a": {"recall": 0.81, "pr_auc": 0.99},
        "model_b": {"recall": 0.82, "pr_auc": 0.90},
    }
    result = modeling.select_best_model(candidates, baseline_recall=0.80, min_recall_uplift_pct=15.0)
    # neither candidate clears a 15% uplift over an 0.80 baseline recall
    assert result.passed_gate == []
    assert result.selected_model == "model_a"  # falls back to best PR-AUC
    assert "WARNING" in result.reason
    assert "does NOT meet" in result.reason
