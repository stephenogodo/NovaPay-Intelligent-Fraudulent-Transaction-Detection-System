"""SHAP-based explainability for the fraud model.

Supports two consumers:
  1. Offline/analyst use — global summary plots, top-feature ranking.
  2. Online/API use — per-transaction top-N reasons, cheap enough to compute
     at request time for a single row (required for the regulatory /
     analyst-facing "why was this flagged" use case).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap


_TREE_MODEL_TYPES = (
    "RandomForestClassifier", "XGBClassifier", "LGBMClassifier",
)


def build_explainer(model, background: pd.DataFrame | None = None):
    """Return the appropriate SHAP explainer for whichever model type won
    model selection, so explainability isn't silently broken if a linear
    model happens to win on a given retrain.

    - Tree ensembles (RF / XGBoost / LightGBM): shap.TreeExplainer — exact,
      fast enough for real-time per-transaction scoring, no background
      sample required.
    - Linear models (Logistic Regression): shap.LinearExplainer — exact for
      linear models, requires a background sample to estimate feature
      correlations/means.
    - Fallback: shap.Explainer (model-agnostic, slower) for anything else.
    """
    model_type = type(model).__name__
    if model_type in _TREE_MODEL_TYPES:
        return shap.TreeExplainer(model)
    if model_type == "LogisticRegression":
        if background is None:
            raise ValueError("LinearExplainer requires a background sample")
        return shap.LinearExplainer(model, background)
    return shap.Explainer(model, background)


# Backwards-compatible alias
def build_tree_explainer(model, background: pd.DataFrame | None = None):
    return build_explainer(model, background)


def global_feature_importance(explainer, X: pd.DataFrame, max_display: int = 15) -> pd.DataFrame:
    """Mean absolute SHAP value per feature, sorted descending."""
    shap_values = explainer.shap_values(X)
    values = shap_values[1] if isinstance(shap_values, list) else shap_values
    mean_abs = np.abs(values).mean(axis=0)
    out = (
        pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return out.head(max_display)


def explain_instance(explainer, x_row: pd.DataFrame, top_n: int = 5,
                      raw_row: pd.DataFrame | None = None) -> list[dict]:
    """Return the top-N features driving a single transaction's fraud score,
    each with its SHAP contribution and direction — this is what the API
    exposes as `reasons` in the /score response.

    SHAP is computed on the model's transformed input (`x_row`: one-hot
    encoded + standard-scaled), but a scaled value like "corridor_risk:
    6.01" is meaningless to an analyst or regulator. When `raw_row` (the
    pre-transform engineered feature row) is supplied, the human-readable
    original value is reported instead — numeric/boolean features map
    directly by name; one-hot encoded categorical columns (e.g.
    "channel_WEB") are resolved back to their source column's actual
    category.
    """
    shap_values = explainer.shap_values(x_row)
    values = shap_values[1] if isinstance(shap_values, list) else shap_values
    row_values = values[0]

    contributions = pd.Series(row_values, index=x_row.columns)
    top = contributions.reindex(contributions.abs().sort_values(ascending=False).index)
    top = top.head(top_n)

    reasons = []
    for feature, contribution in top.items():
        display_value = _resolve_display_value(feature, x_row, raw_row)
        reasons.append({
            "feature": feature,
            "value": display_value,
            "shap_contribution": round(float(contribution), 4),
            "direction": "increases_fraud_risk" if contribution > 0 else "decreases_fraud_risk",
        })
    return reasons


def _resolve_display_value(feature: str, x_row: pd.DataFrame, raw_row: pd.DataFrame | None):
    """Map a (possibly one-hot-encoded, possibly scaled) transformed
    feature name back to a human-readable value from the raw engineered row.
    Falls back to the transformed value if no raw row was supplied or the
    feature can't be resolved (keeps this robust rather than raising).
    """
    if raw_row is None:
        return _to_native(x_row.iloc[0][feature])

    if feature in raw_row.columns:
        return _to_native(raw_row.iloc[0][feature])

    # one-hot encoded column, e.g. "channel_WEB" -> source col "channel"
    for source_col in raw_row.columns:
        prefix = f"{source_col}_"
        if feature.startswith(prefix):
            category = feature[len(prefix):]
            actual = raw_row.iloc[0][source_col]
            return f"{source_col}={actual}" if str(actual) == category else f"{source_col}={actual} (not {category})"

    return _to_native(x_row.iloc[0][feature])


def _to_native(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
