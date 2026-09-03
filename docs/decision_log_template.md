# Model Promotion Decision Log

One entry per retraining/promotion decision. Append-only.

## Entry template

```
Date:               YYYY-MM-DD
Triggered by:       [scheduled | feature_drift | prediction_drift |
                      flag_rate_shift | new_corridor | manual]
Previous model:     <name> v<version>, threshold=<x>
Candidate model:    <name> v<version>, threshold=<x>

Promotion gate:
  [ ] Recall uplift vs rules baseline >= 15%      actual: __%
  [ ] Precision regression <= 5pp vs previous      actual: __pp
  [ ] PR-AUC within band of previous               previous: __  new: __
  [ ] API schema compatible                        [yes/no]

Shadow period:      <start> to <end> (__ days)
Shadow findings:    <flag-rate delta, score-distribution delta, any incidents>

Decision:           [PROMOTED | REJECTED | PROMOTED WITH CAVEATS]
Approved by:        <name/role>
Notes:               <anything that doesn't fit above>
```
