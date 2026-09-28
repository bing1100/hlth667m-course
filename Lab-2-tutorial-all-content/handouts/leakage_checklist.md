# Handout 4 — The Leakage Checklist

*HLTH 667M Lab 2. One page. Use it before reporting any result.*

## What leakage is

**Leakage** is when a workflow uses information it would not have at the moment the prediction is made — or lets evaluation data influence training.

It does not cause an error message. It makes results **look better**, which is why it survives review.

## The two forms

### 1. Future / outcome leakage

Using a value that does not exist yet at the prediction time.

| Example from this lab | Why it leaks |
|---|---|
| `Discharge Date` to predict something at admission | the patient has not been discharged |
| `Length of Stay` at admission | derived from discharge date |
| `Billing Amount` at admission | the bill is settled afterwards |
| `Medication` at admission | may be chosen *because of* the test result |

**The test:** write down the exact moment the prediction would be made. For each column ask — *would this value exist, in this system, at that moment?*

### 2. Preprocessing / evaluation leakage

Letting a step that learns from data see rows it should not.

| Mistake | Fix |
|---|---|
| `StandardScaler().fit(X)` before splitting | put the scaler inside the `Pipeline` |
| `SimpleImputer` fitted on all rows | same |
| `OneHotEncoder` fitted on all rows | same |
| `SelectKBest` before cross-validation | same — this one is the worst |
| Re-using the test set to try another model | you cannot undo this; you need new data |

> **Part 1B section 7 demonstrates this.** Selecting 20 features from 2,000 columns of **pure random noise**, before cross-validating, reports **R² = +0.32** on data containing no signal at all. Done correctly inside the pipeline, the same data gives **R² = −0.28**.

## The checklist

Run through this before any result leaves your notebook.

**Prediction time**
- [ ] I have written down the exact moment of prediction.
- [ ] Every feature exists at that moment.
- [ ] No feature is derived from the outcome.
- [ ] No feature is a consequence of the outcome being known.

**Identifiers and grouping**
- [ ] No near-unique identifier (`Name`, record number) is a feature.
- [ ] If one person can appear in several rows, rows are grouped so they cannot straddle the split.
- [ ] High-cardinality site or clinician fields are excluded, or their effect on transferability is stated.

**Preprocessing**
- [ ] The split happened **before** any fitting.
- [ ] Every imputer, scaler, encoder, and selector is inside the `Pipeline`.
- [ ] Cross-validation is run on the pipeline, not on pre-transformed data.
- [ ] Duplicate rows were removed **before** splitting.

**Evaluation**
- [ ] The test set was used exactly once.
- [ ] No model, threshold, or hyperparameter was chosen using test data.
- [ ] The candidate was compared against a dummy baseline.
- [ ] The metric was chosen before the results were seen.

**Reporting**
- [ ] I stated which dataset, which split, and which time period.
- [ ] I did not claim external validity, calibration, fairness, or clinical usefulness.
- [ ] If the model did not beat baseline, I reported that plainly.

## The warning sign

> **If a result is much better than you expected, look for leakage before you celebrate.**

Near-perfect performance on a hard clinical problem is almost always a leak, not a breakthrough. The cost of checking is an hour. The cost of not checking is a published claim you have to withdraw.

## Where this appears

- `part-1-ml/Part-0` section 6 — classifying columns by when they become known
- `part-1-ml/Part-1B` section 7 — the worked noise demonstration
- Documentation: <https://scikit-learn.org/stable/common_pitfalls.html>
