"""Data & prediction drift monitoring for the deployed fraud model.

Compares a window of recently-scored production transactions against the
reference distribution captured at training time (artifacts/reference_data.parquet)
and flags:
  1. Feature drift  -- input distributions shifting (new fraud tactics,
     new markets, seasonal effects, upstream data-quality regressions).
  2. Prediction drift -- the model's output distribution shifting even if
     inputs look stable (a symptom of model decay).
  3. Flag-rate drift -- the review queue's *volume* changing, which is an
     operational signal fraud-ops needs regardless of statistical drift.

Usage:
    python -m novapay_fraud.monitoring.drift_check --current path/to/recent_scored.parquet
    python -m novapay_fraud.monitoring.drift_check --simulate   # demo mode, uses a
                                                                  # perturbed sample
                                                                  # of the reference set
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

from novapay_fraud import config

logger = logging.getLogger(__name__)

# Columns Evidently should compare distribution-for-distribution. Identifiers
# and the raw target are excluded on purpose -- 'actual' fraud labels lag
# behind in production (investigations take time), so label-based drift
# checks belong in the retraining evaluation step, not this real-time check.
DRIFT_FEATURE_COLUMNS = config.NUMERIC_FEATURES + config.BOOLEAN_FEATURES + config.CATEGORICAL_FEATURES
PREDICTION_COLUMN = "fraud_probability"

# Alerting thresholds -- deliberately conservative defaults; tune against
# real operational tolerance once the service has a production track record.
DRIFTED_COLUMN_SHARE_ALERT = 0.30   # >30% of features drifting -> investigate
PREDICTION_DRIFT_ALERT = 0.10       # Wasserstein distance (normed) on p(fraud)
FLAG_RATE_RELATIVE_CHANGE_ALERT = 0.50  # +/-50% change in review-queue volume


@dataclass
class DriftResult:
    n_reference: int
    n_current: int
    drifted_feature_count: int
    drifted_feature_share: float
    drifted_features: list[str]
    prediction_drift_score: float | None
    reference_flag_rate: float
    current_flag_rate: float
    flag_rate_relative_change: float
    alerts: list[str]
    checked_at: str

    def to_dict(self) -> dict:
        return {
            "n_reference": self.n_reference,
            "n_current": self.n_current,
            "drifted_feature_count": self.drifted_feature_count,
            "drifted_feature_share": round(self.drifted_feature_share, 4),
            "drifted_features": self.drifted_features,
            "prediction_drift_score": (
                round(self.prediction_drift_score, 4)
                if self.prediction_drift_score is not None else None
            ),
            "reference_flag_rate": round(self.reference_flag_rate, 4),
            "current_flag_rate": round(self.current_flag_rate, 4),
            "flag_rate_relative_change": round(self.flag_rate_relative_change, 4),
            "alerts": self.alerts,
            "checked_at": self.checked_at,
        }


def _extract_drift_values(snapshot_dict: dict) -> dict[str, float]:
    """Pull per-column drift scores out of an Evidently Report snapshot."""
    values = {}
    for metric in snapshot_dict.get("metrics", []):
        name = metric.get("metric_name", "")
        if name.startswith("ValueDrift(column="):
            col = name.split("column=")[1].split(",")[0]
            v = metric.get("value")
            if isinstance(v, dict):
                v = v.get("value")
            values[col] = v
    return values


def check_drift(reference: pd.DataFrame, current: pd.DataFrame,
                 decision_threshold: float | None = None) -> DriftResult:
    threshold = decision_threshold
    if threshold is None:
        with open(config.METADATA_PATH) as f:
            threshold = json.load(f)["decision_threshold"]

    feature_cols = [c for c in DRIFT_FEATURE_COLUMNS if c in reference.columns and c in current.columns]
    definition = DataDefinition()
    ref_ds = Dataset.from_pandas(reference[feature_cols], data_definition=definition)
    cur_ds = Dataset.from_pandas(current[feature_cols], data_definition=definition)

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(current_data=cur_ds, reference_data=ref_ds)
    snap_dict = snapshot.dict()

    per_column = _extract_drift_values(snap_dict)
    drifted_features = [c for c, v in per_column.items() if v is not None and v >= 0.1]

    pred_drift_score = None
    if PREDICTION_COLUMN in reference.columns and PREDICTION_COLUMN in current.columns:
        pred_definition = DataDefinition()
        pred_ref = Dataset.from_pandas(reference[[PREDICTION_COLUMN]], data_definition=pred_definition)
        pred_cur = Dataset.from_pandas(current[[PREDICTION_COLUMN]], data_definition=pred_definition)
        pred_report = Report(metrics=[DataDriftPreset()])
        pred_snapshot = pred_report.run(current_data=pred_cur, reference_data=pred_ref).dict()
        pred_values = _extract_drift_values(pred_snapshot)
        pred_drift_score = pred_values.get(PREDICTION_COLUMN)

    ref_flag_rate = float((reference[PREDICTION_COLUMN] >= threshold).mean()) if PREDICTION_COLUMN in reference else 0.0
    cur_flag_rate = float((current[PREDICTION_COLUMN] >= threshold).mean()) if PREDICTION_COLUMN in current else 0.0
    relative_change = (cur_flag_rate - ref_flag_rate) / max(ref_flag_rate, 1e-9)

    alerts = []
    drifted_share = len(drifted_features) / max(len(feature_cols), 1)
    if drifted_share >= DRIFTED_COLUMN_SHARE_ALERT:
        alerts.append(
            f"FEATURE_DRIFT: {len(drifted_features)}/{len(feature_cols)} "
            f"features drifted ({drifted_share:.0%}) -- investigate upstream "
            f"data sources and consider retraining."
        )
    if pred_drift_score is not None and pred_drift_score >= PREDICTION_DRIFT_ALERT:
        alerts.append(
            f"PREDICTION_DRIFT: model's output distribution shifted "
            f"(score={pred_drift_score:.3f}) -- possible model decay."
        )
    if abs(relative_change) >= FLAG_RATE_RELATIVE_CHANGE_ALERT:
        direction = "increased" if relative_change > 0 else "decreased"
        alerts.append(
            f"FLAG_RATE_SHIFT: review queue volume {direction} by "
            f"{abs(relative_change):.0%} vs reference -- check ops capacity "
            f"and confirm this isn't a broken feature pipeline."
        )

    return DriftResult(
        n_reference=len(reference),
        n_current=len(current),
        drifted_feature_count=len(drifted_features),
        drifted_feature_share=drifted_share,
        drifted_features=drifted_features,
        prediction_drift_score=pred_drift_score,
        reference_flag_rate=ref_flag_rate,
        current_flag_rate=cur_flag_rate,
        flag_rate_relative_change=relative_change,
        alerts=alerts,
        checked_at=datetime.now(timezone.utc).isoformat(),
    )


def _simulate_current_window(reference: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Demo/self-test helper: perturb a sample of the reference set so the
    CLI has something to check without a real production feed yet.
    """
    cur = reference.sample(min(300, len(reference)), random_state=seed).copy()
    cur["amount_usd"] = cur["amount_usd"] * 1.4
    cur["ip_risk_score"] = (cur["ip_risk_score"] * 1.2).clip(0, 1)
    return cur


def main():
    parser = argparse.ArgumentParser(description="Check the fraud model for data/prediction drift")
    parser.add_argument("--current", type=str, default=None,
                         help="Path to a parquet/csv of recently-scored transactions")
    parser.add_argument("--simulate", action="store_true",
                         help="Use a synthetically perturbed sample for a demo/self-test run")
    parser.add_argument("--out", type=str, default=None,
                         help="Optional path to write the JSON drift report")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    reference = pd.read_parquet(config.REFERENCE_DATA_PATH)

    if args.simulate:
        current = _simulate_current_window(reference)
        logger.info("Running in --simulate mode on a perturbed reference sample")
    elif args.current:
        path = Path(args.current)
        current = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    else:
        parser.error("Provide --current <path> or use --simulate")

    result = check_drift(reference, current)
    out = json.dumps(result.to_dict(), indent=2)
    print(out)

    if result.alerts:
        for a in result.alerts:
            logger.warning(a)

    if args.out:
        Path(args.out).write_text(out)
        logger.info("Report written to %s", args.out)


if __name__ == "__main__":
    main()
