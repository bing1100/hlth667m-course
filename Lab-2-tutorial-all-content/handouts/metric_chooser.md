# Handout 3 — The Metric Chooser

*HLTH 667M Lab 2. One page.*

## Step 1 — What kind of target?

```
  Is the target a CATEGORY or a NUMBER?
        |                        |
     CATEGORY                 NUMBER
        |                        |
   classification            regression
        |                        |
   go to section A          go to section B
```

## A. Classification metrics

| Question you are actually asking | Metric | Direction |
|---|---|---|
| Overall, how often is it right? | Accuracy | higher |
| Of the cases it flagged, how many were real? | **Precision** | higher |
| Of the real cases, how many did it find? | **Recall** (sensitivity) | higher |
| Balance precision and recall in one number | F1 | higher |
| Treat every class as equally important | **Macro F1** | higher |
| How well does it rank cases, at any threshold? | ROC-AUC | higher |
| Same, when the positive class is rare | PR-AUC | higher |

### Choosing between precision and recall

Ask: **which mistake costs more?**

- **Missing a case is worse** (a malignancy, a deteriorating patient) → prioritise **recall**.
- **A false alarm is worse** (an invasive follow-up, an alert that causes fatigue) → prioritise **precision**.
- You cannot maximise both. Moving the threshold trades one for the other.

### When accuracy will mislead you

Whenever the classes are unbalanced. In Part 1A the dummy baseline scores **0.63 accuracy** and **0.39 macro F1** on identical predictions — and it never identifies a single malignant case. Accuracy hid that completely.

> Rule of thumb: if one class is under about 30% of the rows, do not lead with accuracy.

## B. Regression metrics

| Question | Metric | Direction | Units |
|---|---|---|---|
| Typical size of an error | **MAE** | lower | same as target |
| Same, but punish large errors more | **RMSE** | lower | same as target |
| How much better than predicting the mean? | **R²** | higher | unitless |

### MAE or RMSE?

They can rank two models differently.

- **MAE** treats a 10-unit error as exactly twice a 5-unit error.
- **RMSE** treats it as four times worse.

Choose RMSE when a few large misses are genuinely much worse than many small ones. Choose MAE when they are not, or when you need a number you can explain in one sentence.

### Reading R² correctly

| Value | Meaning |
|---|---|
| 1.0 | perfect prediction |
| 0.45 | explains 45% of the variation — **most is still unexplained** |
| 0.0 | no better than predicting the mean |
| **negative** | **worse** than predicting the mean on that evaluation set |

**A negative R² is not a negative correlation.** Never take its absolute value.

## The scikit-learn negation trap

scikit-learn treats higher as better for every scorer, so error metrics are supplied negated:

```python
scoring = {"mae": "neg_mean_absolute_error"}   # returns NEGATIVE values
mae = -scores["test_mae"].mean()               # multiply by -1 to display
```

A negative MAE in your output is a convention, not a bug.

## Step 2 — Before you report any number

- [ ] Did I compare against a **dummy baseline**?
- [ ] Did I choose the metric **before** seeing results?
- [ ] Did I report errors **by class or by subgroup**, not just overall?
- [ ] Did I state which **evaluation set** the number came from?
- [ ] Did I avoid implying the model "knows", "understands", or "decides"?

## What no metric on this page tells you

Calibration, fairness across subgroups, causation, performance at another site or a later date, or whether the model should be used at all. Those need separate evidence.
