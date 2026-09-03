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
    # raw_valid must exist and line up with X_valid/y_valid (needed to
    # compute the validation-window baseline recall for threshold gating)
    assert split.raw_valid is not None
    assert len(split.raw_valid) == len(split.X_valid) == len(split.y_valid)


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


def test_default_threshold_grid_extends_well_past_070():
    """Regression test for a real bug: THRESHOLD_GRID used to stop at 0.70,
    which forced Logistic Regression's selected threshold to land exactly
    on that boundary (0.70) by construction -- the search could never find
    the true F1-maximizing threshold (0.91, verified by direct sweep on
    real data) because the grid didn't extend far enough to contain it.
    The grid must reach at least 0.95 for this class of clipping to be
    structurally impossible going forward.
    """
    assert max(config.THRESHOLD_GRID) >= 0.95


def test_select_threshold_flags_a_boundary_result():
    """If every candidate threshold's F1 keeps improving right up to the
    edge of the search grid, that's the signature of a clipped search, not
    a genuine interior optimum -- select_threshold must say so rather than
    returning a boundary value that looks no different from a real one.
    """
    # Monotonically-separable probabilities: F1 keeps climbing as the
    # threshold rises across this whole narrow grid, so the winner is
    # forced to land on the grid's own upper edge.
    y = pd.Series([0, 0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.85, 0.9, 0.95])
    narrow_grid = [0.1, 0.2, 0.3, 0.4]
    result = modeling.select_threshold(y, proba, grid=narrow_grid)
    assert result["threshold"] == max(narrow_grid)
    assert result["at_grid_boundary"] is True


def test_select_threshold_does_not_flag_a_genuine_interior_optimum():
    y = pd.Series([0, 1, 0, 1, 1, 0, 0, 1])
    proba = np.array([0.1, 0.9, 0.2, 0.8, 0.6, 0.3, 0.4, 0.95])
    grid = [0.1, 0.3, 0.5, 0.7, 0.9]
    result = modeling.select_threshold(y, proba, grid=grid)
    if result["threshold"] not in (min(grid), max(grid)):
        assert result["at_grid_boundary"] is False


def test_select_threshold_respects_a_recall_floor_over_pure_f1():
    """Regression test for a real bug: pure F1-maximization and the
    business's actual requirement (a hard recall floor) are not the same
    objective. On this fixture, one true positive scores lower than
    several negatives, so the unconstrained F1-optimal threshold sacrifices
    it for better precision; a recall-constrained search must instead
    accept the precision hit needed to still catch it.
    """
    y = pd.Series([1, 0, 0, 0, 0, 1, 1, 1, 1])
    proba = np.array([0.05, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.92, 0.95])
    grid = [0.05, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.9]

    unconstrained = modeling.select_threshold(y, proba, grid=grid)
    # The unconstrained winner should NOT already satisfy a 100% recall
    # floor -- otherwise this fixture doesn't actually exercise the
    # trade-off the test is designed to catch.
    assert unconstrained["recall"] < 1.0
    assert unconstrained["threshold"] == 0.6  # verified F1-optimal point

    constrained = modeling.select_threshold(y, proba, grid=grid, min_recall=1.0)
    assert constrained["recall"] == 1.0
    assert constrained["recall_floor_met"] is True
    # Satisfying full recall must come at or below the unconstrained
    # threshold -- it cannot be pickier about what counts as fraud.
    assert constrained["threshold"] <= unconstrained["threshold"]


def test_select_threshold_flags_when_no_grid_value_meets_the_recall_floor():
    y = pd.Series([0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    grid = [0.1, 0.3, 0.5, 0.7]
    # No threshold in this grid can achieve 100% recall AND avoid flagging
    # everything, but more importantly: require a recall no threshold here
    # can reach at all (recall of 1.5 is impossible) to force the fallback.
    result = modeling.select_threshold(y, proba, grid=grid, min_recall=1.5)
    assert result["recall_floor_met"] is False


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
