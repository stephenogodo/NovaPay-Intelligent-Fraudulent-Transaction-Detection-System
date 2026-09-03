"""End-to-end training entrypoint.

Usage:
    python -m novapay_fraud.train

Produces (all under artifacts/):
    preprocessor.joblib   -- fitted ColumnTransformer
    fraud_model.joblib    -- fitted best model (full sklearn Pipeline)
    shap_explainer.joblib -- fitted SHAP TreeExplainer for the best model
    model_metadata.json   -- chosen model, threshold, feature list, versions
    metrics.json           -- full comparison table across all candidate models
    reference_data.parquet -- scored training-window sample, for drift monitoring
    feature_schema.json    -- expected input schema for the API / monitoring
"""
from __future__ import annotations

import json
import logging
import platform
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from . import baseline, config, data, explain, features, modeling

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run() -> dict:
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load & clean --------------------------------------------------
    logger.info("Step 1/7: loading and cleaning raw data")
    raw = data.load_raw()
    clean_df, clean_report = data.clean(raw, drop_incomplete_rows=True)
    logger.info(
        "Cleaned: %s -> %s rows (%.2f%% fraud)",
        clean_report["input_rows"], clean_report["output_rows"],
        clean_report["output_fraud_rate"] * 100,
    )

    # 2. Feature engineering --------------------------------------------
    logger.info("Step 2/7: feature engineering")
    fe_df = features.engineer_features(clean_df)
    X_full = features.get_feature_matrix(fe_df)
    fe_df = pd.concat(
        [fe_df[[config.TIMESTAMP_COL, config.TARGET]], X_full], axis=1
    )

    # 3. Time-ordered split ----------------------------------------------
    logger.info("Step 3/7: time-based train/valid/test split")
    split = modeling.time_based_split(fe_df, config.ALL_FEATURES)
    logger.info(
        "Train=%s (fraud=%.2f%%) Valid=%s (fraud=%.2f%%) Test=%s (fraud=%.2f%%)",
        len(split.X_train), split.y_train.mean() * 100,
        len(split.X_valid), split.y_valid.mean() * 100,
        len(split.X_test), split.y_test.mean() * 100,
    )

    # 4. Preprocessing (fit on train only) --------------------------------
    logger.info("Step 4/7: fitting preprocessor on training data only")
    preprocessor = features.build_preprocessor()
    Xt_train = preprocessor.fit_transform(split.X_train)
    Xt_valid = preprocessor.transform(split.X_valid)
    Xt_test = preprocessor.transform(split.X_test)
    feature_names = preprocessor.get_feature_names_out().tolist()
    Xt_train = pd.DataFrame(Xt_train, columns=feature_names, index=split.X_train.index)
    Xt_valid = pd.DataFrame(Xt_valid, columns=feature_names, index=split.X_valid.index)
    Xt_test = pd.DataFrame(Xt_test, columns=feature_names, index=split.X_test.index)

    # 5. Train & compare candidate models ---------------------------------
    logger.info("Step 5/7: training candidate models")

    # Compute the recall floor on the VALIDATION window (not test) so
    # threshold selection respects the same business requirement it will
    # later be checked against on test data -- rather than discovering
    # only after the fact, via select_best_model's gate, that an
    # F1-optimal threshold silently failed to meet it. See
    # modeling.select_threshold's docstring for why this must be enforced
    # at threshold-selection time, not just at model-selection time.
    baseline_valid_metrics = baseline.evaluate(split.raw_valid, split.y_valid)
    min_recall_floor = baseline_valid_metrics["recall"] * (1 + config.MIN_RECALL_UPLIFT_PCT / 100)
    logger.info(
        "    Validation-window rules baseline: recall=%.4f -> required "
        "recall floor for threshold selection: %.4f",
        baseline_valid_metrics["recall"], min_recall_floor,
    )

    results = {}
    fitted_models = {}
    for name in config.MODEL_REGISTRY:
        logger.info(" -> training %s", name)
        model = modeling.build_model(name)
        model = modeling.fit_model(name, model, Xt_train, split.y_train)
        proba_valid = modeling.predict_proba_positive(model, Xt_valid)
        thresh_choice = modeling.select_threshold(
            split.y_valid, proba_valid, min_recall=min_recall_floor,
        )

        proba_test = modeling.predict_proba_positive(model, Xt_test)
        test_metrics = modeling.evaluate(split.y_test, proba_test, thresh_choice["threshold"])

        fitted_models[name] = model
        results[name] = {
            "valid_threshold_selection": thresh_choice,
            "test_metrics": test_metrics,
        }
        logger.info(
            "    %s | thr=%.2f  P=%.3f R=%.3f F1=%.3f ROC-AUC=%.3f PR-AUC=%.3f "
            "(validation recall floor met: %s)",
            name, test_metrics["threshold"], test_metrics["precision"],
            test_metrics["recall"], test_metrics["f1"],
            test_metrics["roc_auc"], test_metrics["pr_auc"],
            thresh_choice.get("recall_floor_met"),
        )

    # rules-based baseline, for the required recall-uplift comparison
    baseline_metrics = baseline.evaluate(split.raw_test, split.y_test)
    logger.info(
        "    rules_baseline | P=%.3f R=%.3f F1=%.3f",
        baseline_metrics["precision"], baseline_metrics["recall"], baseline_metrics["f1"],
    )

    # 6. Model selection ---------------------------------------------------
    # Automatic, gated on the actual business requirement: filter candidates
    # to those meeting the minimum recall-uplift threshold, then rank the
    # survivors by PR-AUC. See modeling.select_best_model's docstring for why
    # this is not the same as simply picking whichever candidate has the
    # highest PR-AUC in isolation.
    candidate_test_metrics = {name: r["test_metrics"] for name, r in results.items()}
    selection = modeling.select_best_model(
        candidate_test_metrics,
        baseline_recall=baseline_metrics["recall"],
        min_recall_uplift_pct=config.MIN_RECALL_UPLIFT_PCT,
    )
    best_name = selection.selected_model
    best_model = fitted_models[best_name]
    best_metrics = results[best_name]["test_metrics"]
    recall_uplift_pct = selection.recall_uplift_pct[best_name]

    logger.info("Model selection: %s", selection.reason)
    logger.info(
        "Selected model: %s (PR-AUC=%.3f, recall uplift vs rules baseline = %.1f%%)",
        best_name, best_metrics["pr_auc"], recall_uplift_pct,
    )
    if selection.failed_gate:
        logger.info(
            "    (candidates that did NOT meet the %.0f%% recall-uplift "
            "requirement and were excluded from selection: %s)",
            config.MIN_RECALL_UPLIFT_PCT, ", ".join(selection.failed_gate),
        )

    # 7. Persist artifacts ---------------------------------------------------
    logger.info("Step 6/7: persisting model + explainer artifacts")
    full_pipeline = Pipeline([("preprocessor", preprocessor), ("model", best_model)])
    joblib.dump(full_pipeline, config.MODEL_PATH)
    joblib.dump(preprocessor, config.PREPROCESSOR_PATH)

    # background sample only used by non-tree explainers; harmless to pass always
    background_sample = Xt_train.sample(min(100, len(Xt_train)), random_state=config.RANDOM_STATE)
    explainer = explain.build_explainer(best_model, background=background_sample)
    joblib.dump(explainer, config.SHAP_EXPLAINER_PATH)

    global_importance = explain.global_feature_importance(explainer, Xt_test)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "best_model": best_name,
        "decision_threshold": best_metrics["threshold"],
        "feature_names_raw": config.ALL_FEATURES,
        "feature_names_transformed": feature_names,
        "recall_uplift_vs_rules_baseline_pct": recall_uplift_pct,
        "meets_min_recall_uplift_requirement": recall_uplift_pct >= config.MIN_RECALL_UPLIFT_PCT,
        "selection_reason": selection.reason,
        "candidates_passing_recall_gate": selection.passed_gate,
        "candidates_failing_recall_gate": selection.failed_gate,
        "recall_uplift_pct_by_candidate": selection.recall_uplift_pct,
        "top_shap_features": global_importance["feature"].tolist(),
        "versions": {
            "python": sys.version.split()[0],
            "sklearn": sklearn.__version__,
            "platform": platform.platform(),
        },
        "dataset": {
            "raw_rows": clean_report["input_rows"],
            "clean_rows": clean_report["output_rows"],
            "fraud_rate": clean_report["output_fraud_rate"],
            "rows_dropped_for_missing": clean_report.get("rows_dropped_for_missing"),
            "full_duplicate_rows_removed": clean_report["duplicates"]["full_duplicate_rows"],
        },
    }
    with open(config.METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    all_metrics = {
        "candidate_models": results,
        "rules_baseline": baseline_metrics,
        "selected_model": best_name,
        "global_shap_importance": global_importance.to_dict(orient="records"),
        "data_cleaning_report": clean_report,
    }
    with open(config.METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    # Reference dataset for drift monitoring: scored validation-window sample
    ref = split.X_valid.copy()
    ref["fraud_probability"] = modeling.predict_proba_positive(best_model, Xt_valid)
    ref["actual"] = split.y_valid.values
    ref.to_parquet(config.REFERENCE_DATA_PATH, index=False)

    schema = {
        "categorical_features": config.CATEGORICAL_FEATURES,
        "boolean_features": config.BOOLEAN_FEATURES,
        "raw_numeric_features": config.RAW_NUMERIC,
        "timestamp_field": config.TIMESTAMP_COL,
        "engineered_at_score_time": True,
        "note": (
            "The API accepts RAW transaction fields (raw_numeric_features + "
            "categorical_features + boolean_features + timestamp_field); time "
            "features and risk flags are engineered server-side, mirroring "
            "novapay_fraud.features.engineer_features."
        ),
    }
    with open(config.FEATURE_SCHEMA_PATH, "w") as f:
        json.dump(schema, f, indent=2)

    logger.info("Step 7/7: done. Artifacts written to %s", config.ARTIFACTS_DIR)
    return metadata


if __name__ == "__main__":
    run()
