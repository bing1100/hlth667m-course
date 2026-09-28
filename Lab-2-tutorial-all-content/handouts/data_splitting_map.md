# Handout 2 — The Data-Splitting Map

*HLTH 667M Lab 2. One page.*

## The three roles data can play

| Role | How much | Used for | How many times |
|---|---|---|---|
| Training fold | 4/5 of the training portion | fitting the model | every fold |
| Validation fold | 1/5 of the training portion | comparing candidates | every fold |
| Held-out test set | 20% of all rows | one final estimate | **once, ever** |

## The picture

```
  ALL ROWS  (after removing exact duplicates)
  |
  +-- 80% TRAINING PORTION ------------------+   +-- 20% HELD-OUT TEST ----+
  |                                          |   |                        |
  |  5-fold cross-validation:                |   |   sealed until model    |
  |                                          |   |   selection is finished |
  |  fold 1  [VAL][train][train][train][train]|  |                        |
  |  fold 2  [train][VAL][train][train][train]|  |   evaluate ONCE         |
  |  fold 3  [train][train][VAL][train][train]|  |   report the number     |
  |  fold 4  [train][train][train][VAL][train]|  |   do not re-run         |
  |  fold 5  [train][train][train][train][VAL]|  |                        |
  |                                          |   |                        |
  |  average the 5 validation scores         |   |                        |
  +------------------------------------------+   +------------------------+
```

## Which splitter

| Situation | Use | Why |
|---|---|---|
| Classification | `StratifiedKFold` | keeps class proportions in every fold |
| Regression | `KFold` | no classes to balance |
| Repeated visits by the same patient | `GroupKFold` | otherwise one person appears in train *and* validation |
| Predicting future admissions | chronological split | random folds let the future predict the past |

> **In this lab's CSV, `Name` is not a usable patient ID** — the capitalisation is inconsistent and the data are synthetic. It is there to show you *why* a stable patient identifier would be required, not to be used as one.

## The rule that matters most

**Cross-validation does not replace the test set.**

- Cross-validation answers: *which of these candidates should I pick?*
- The test set answers: *how did the chosen one do on data that never influenced it?*

If you look at the test set, change something, and look again, it has become a validation set — and your final number is now optimistic. There is no way to undo this except collecting new data.

## Reading a cross-validation table

```
                     f1_macro mean   f1_macro SD
Dummy prior                  0.385         0.000
Logistic regression          0.976         0.011
```

Ask three questions in this order:

1. **Does the candidate beat the baseline?** If not, stop. Report that.
2. **Is the gap larger than the fold-to-fold variation?** Compare the difference in means against the SDs.
3. **Is the gap large enough to matter for the decision at hand?** This is a judgement, not a calculation.

## Where this appears

- `part-1-ml/Part-1A` sections 2 and 4 · `part-1-ml/Part-1B` sections 2 and 4
- Documentation: <https://scikit-learn.org/stable/modules/cross_validation.html>
