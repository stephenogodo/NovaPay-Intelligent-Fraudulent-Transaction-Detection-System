# NovaPay Fraud Model — Retraining Playbook

This is an operational runbook, not aspirational documentation: every step
below maps to a real command in this repo.

## 1. Triggers for retraining

Retrain when **any** of the following occurs — don't wait for all of them:

| Trigger | How it's detected | Typical cadence |
|---|---|---|
| Scheduled cadence | Calendar | Quarterly, minimum |
| Feature drift alert | `novapay_fraud.monitoring.drift_check` — `FEATURE_DRIFT` | As triggered |
| Prediction drift alert | Same module — `PREDICTION_DRIFT` | As triggered |
| Review-queue volume shift | Same module — `FLAG_RATE_SHIFT` | As triggered |
| Confirmed-fraud label backlog resolves | Ops/investigations team signal | Monthly review |
| A new corridor/currency/channel launches | Product roadmap | Ad hoc, before launch if possible |

Quarterly retraining is the floor, not the target — a drift alert firing
mid-quarter should not wait for the calendar.

## 2. Pre-retraining checklist

1. Confirm the label backlog is current: fraud labels come from completed
   investigations and confirmed chargebacks, which lag the transaction by
   days-to-weeks. Retraining on a window with a large fraction of
   *not-yet-labeled* recent fraud silently teaches the model that recent
   fraud didn't happen. Check `data.analyze_missingness_vs_target`-style
   label-lag reporting before pulling the training window.
2. Pull the latest data into `data/nova_pay_combined.csv` (or point
   `config.RAW_DATA_PATH` at the new extract).
3. Diff the new data's schema against `artifacts/feature_schema.json` --
   a silently added/renamed/retyped column is the single most common cause
   of a retraining pipeline producing a broken model.

## 3. Run retraining

```bash
python -m novapay_fraud.train
# or, in Docker:
docker compose run --rm retrain
```

This regenerates, in one deterministic pass:
- `artifacts/fraud_model.joblib` (full sklearn Pipeline: preprocessor + model)
- `artifacts/shap_explainer.joblib`
- `artifacts/model_metadata.json` (chosen model, threshold, recall-uplift
  check against the rules baseline, feature list, library versions)
- `artifacts/metrics.json` (full candidate-model comparison table)
- `artifacts/reference_data.parquet` (new drift-monitoring baseline)

## 4. Promotion gate — do not skip

Before replacing the production model, `model_metadata.json` must satisfy
**all** of:

- [ ] `meets_min_recall_uplift_requirement == true` (≥15% recall uplift vs
      the rules baseline on held-out, time-ordered test data)
- [ ] Test-set precision has not regressed by more than 5 percentage points
      vs the currently-deployed model's `metrics.json` (a recall-chasing
      retrain that floods the review queue is not a safe promotion)
- [ ] PR-AUC on the new test window is within a reasonable band of the
      previous model's — a large *drop* suggests something upstream broke,
      not that the new model is "adapting to new fraud patterns"
- [ ] The new `feature_names_transformed` list is backward compatible with
      the API's `TransactionRequest` schema, or the API schema is updated
      in the same change

If any box is unchecked, do not promote — investigate first. A model that
narrowly misses the bar is a signal to look at the data, not to lower the
bar.

## 5. Shadow / canary before full cutover

1. Deploy the new model artifacts to a shadow endpoint that scores
   production traffic but whose output is logged, not acted on.
2. Compare shadow vs. production flag-rate and score distribution for at
   least 3–7 days of representative traffic (`drift_check.py` works
   equally well for this "new model vs old model" comparison as it does
   for "new data vs old data").
3. Promote to production only after shadow-period metrics match the
   promotion-gate numbers from step 4.

## 6. Rollback plan

Keep the previous `artifacts/` directory versioned (e.g.
`artifacts/2026-Q3/`, `artifacts/2026-Q4/`) rather than overwritten in
place. Rollback is: point `config.ARTIFACTS_DIR` (or the container's mounted
volume) back at the previous quarter's directory and redeploy — no
retraining required to revert.

## 7. Post-deployment

- Re-baseline `artifacts/reference_data.parquet` is done automatically by
  `train.py` — confirm the monitoring dashboard picks up the new reference
  window (a stale reference makes every subsequent drift check meaningless).
- Log the promotion decision (date, model version, metrics, approver) — see
  `docs/decision_log_template.md`.
- Re-run `novapay_fraud.monitoring.drift_check` daily for the first week
  after cutover to catch integration issues early, then fall back to the
  normal monitoring cadence.
