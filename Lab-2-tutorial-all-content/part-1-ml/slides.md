# Part 1 — Presenter-Ready Slide Helper: ML Pipelines

> **Presenting from the Beamer deck?** Use `../../latex-slides/speaker-script.md`. It has one entry per page of `lab2-slides.pdf`, in the deck's current order, and `make script` in that folder checks that it still matches. This file is the longer planning outline for the same material, and its slide order can lag behind the deck.

## Slide 0 — Welcome, roadmap, and goals (3 min)
**Visible text:** “First: describe the course CSV. Then: category or number?” “Today: audit → define → split → pipeline → compare → evaluate → limit.”
**Visual:** Supplied healthcare CSV → descriptive analytics; tumor features → diagnosis; diabetes variables → quantitative target.
**Alt text:** One non-modeling healthcare workflow and two bundled scikit-learn modeling benchmarks.
**Say:** “We will build two complete workflows. A model result is not a clinical or administrative deployment decision.”
**Notebook transition:** Open both notebook titles.

## Slide 1 — Classification versus regression (3 min)
**Visible text:** “Categories need classification. Continuous values need regression.”
**Visual:** Label cards versus number line.
**Alt text:** The breast-cancer diagnosis is discrete; the diabetes target is continuous.
**Ask:** “Which target type determines the model family?”
**Expected:** “The target type.”

## Slide 2 — Prediction time determines eligible features (4 min)
**Visible text:** “Availability depends on prediction time.” “Do not use future information.”
**Visual:** Admission → testing/care → discharge/billing timeline, with later fields crossed out.
**Alt text:** Discharge date and final billing occur after admission and are excluded.
**Say:** “A column can be present in a CSV and still be invalid for a stated prediction time.”

## Slide 3 — Leakage can look like success (3 min)
**Visible text:** “Future information and pre-split preprocessing can inflate performance.”
**Visual:** Two warning paths: discharge date into model; scaler fit on all rows.
**Alt text:** Both future variables and validation-aware preprocessing create invalid estimates.
**Notebook transition:** Run Part 0’s feature-availability section, then each benchmark notebook’s split section.

## Slide 4 — Data audit and duplicates (3 min)
**Visible text:** “Inspect before analysis.” “Remove exact duplicates before summaries.”
**Visual:** Duplicate rows feeding both train and test partitions.
**Alt text:** Duplicate records distort descriptive counts and could inflate future model evaluation.

## Slide 5 — Holdout test set and cross-validation (5 min)
**Visible text:** “Training portion: compare workflows.” “Held-out test set: evaluate once.”
**Visual:** 80% development block split into five folds; separate 20% test block.
**Alt text:** Five validation folds occur only inside the training portion; the final test portion remains untouched.
**Ask:** “Which data should not choose the model?”
**Expected:** “The held-out test set.”

## Slide 6 — Why a pipeline? (4 min)
**Visible text:** “Pipeline = preprocessing + estimator.” “Everything that learns from data goes inside.”
**Visual:** numeric branch: impute → scale; categorical branch: impute → one-hot; both → ColumnTransformer → model.
**Alt text:** A ColumnTransformer combines numeric and categorical preprocessing before the estimator, all inside one pipeline object.
**Say:** “This is not a convenience. During cross-validation each fold refits the preprocessing on its own training rows. Do it yourself beforehand and every fold is scored on information it should not have had.”
**Ask:** “Which steps must go inside?”
**Expected:** “Every step that learns something: imputer, scaler, encoder, feature selector.”
**Notebook transition:** Run Part 1A section 3 for the numeric pipeline, then section 7 for the ColumnTransformer on the real categorical health columns.

## Slide 7 — Categories are not numbers (3 min)
**Visible text:** “`Elective=1, Emergency=2, Urgent=3` asserts an order that does not exist.” “One-hot: one 0/1 column per category.”
**Visual:** One `Admission Type` column expanding into three 0/1 columns, beside a crossed-out 1/2/3 encoding.
**Alt text:** One-hot encoding turns a single categorical column into one indicator column per category, unlike an integer code that implies ranking.
**Say:** “Numbering the categories tells the model that Urgent is greater than Elective, and that the gap between them equals the gap to Emergency. Both claims are false.”
**Ask:** “`Doctor` has 40,000 distinct values. What happens if you one-hot encode it?”
**Expected:** “40,000 columns. That is one concrete reason Part 0 excluded it.”
**Notebook transition:** Run Part 1A section 7 and read the generated feature names.

## Slide 8 — Baselines (3 min)
**Visible text:** “First question: did the fitted model beat a trivial predictor?”
**Visual:** Dummy classifier predicts prior class; dummy regressor predicts mean.
**Alt text:** Baselines establish a minimum comparison point.

## Slide 9 — Classification metrics (5 min)
**Visible text:** “Accuracy: overall correct.” “Precision: predicted class correct?” “Recall: true class found?” “Macro F1: balance classes equally.”
**Visual:** Small binary confusion matrix with TP/FP/FN/TN labels.
**Alt text:** A confusion matrix supports precision, recall, and F1 calculations.

## Slide 10 — Classification graphics (4 min)
**Visible text:** “Read errors by class.” “ROC-AUC ranks scores across thresholds.”
**Visual:** Confusion matrix and ROC sketch.
**Alt text:** Confusion-matrix off-diagonal cells are errors; ROC curves compare true-positive and false-positive rates.
**Say:** “AUC is not accuracy, calibration, utility, or a deployment threshold.”

## Slide 11 — Regression metrics (5 min)
**Visible text:** “MAE: typical absolute error.” “RMSE: larger errors count more.” “R²: comparison with mean predictor.”
**Visual:** Four residual bars and MAE/RMSE formulas.
**Alt text:** Regression errors are vertical distances between actual and predicted amounts.

## Slide 12 — Regression graphics (4 min)
**Visible text:** “Perfect prediction lies on the 45-degree line.” “Residual = actual − predicted.”
**Visual:** actual-vs-predicted and residual-vs-predicted sketches.
**Alt text:** Points far from the identity line have larger errors; residuals should be inspected around zero.

## Slide 13 — Interpret against baseline (3 min)
**Visible text:** “Near baseline can be an honest result.”
**Visual:** Comparison table with baseline and candidate columns.
**Alt text:** A benchmark may beat a simple reference, while the supplied CSV remains a valid example of weak signal.

## Slide 14 — Leakage can manufacture a result (5 min)
**Visible text:** “Select 20 features from 2,000 columns of **pure noise**, before cross-validating.” “Reported R² = +0.32.” “Correct answer: 0.”
**Visual:** Two bars: a leaky pipeline reporting positive R² beside an honest pipeline reporting a slightly negative one, over a caption stating that both used identical random data.
**Alt text:** Feature selection performed before cross-validation reports positive explained variance on data with no signal, while the same data inside a pipeline reports approximately zero.
**Say:** “With 2,000 candidates, some correlate with the target by chance. Selecting on all the data bakes those chance correlations in before the folds exist, so every validation fold is scored on features already tuned to it.”
**Ask:** “The code looks completely reasonable. What makes this dangerous?”
**Expected:** “It produces a better-looking number, so nobody investigates — and it survives a held-out test set if the split came after the selection.”
**Notebook transition:** Run Part 1B section 7. Ask students to predict the two results before running.

## Slide 15 — Result limits (4 min)
**Visible text:** “Internal random-split performance is not deployment evidence.”
**Visual:** Missing-evidence checklist: future time, other sites, subgroups, calibration, workflow, governance.
**Alt text:** A completed notebook still lacks several forms of deployment validation.

## Slide 16 — Exit reflection (2 min)
**Visible text:** “Name one leakage safeguard and justify one metric.”
**Visual:** Two response boxes.
**Alt text:** Students state a safeguard and a metric choice with a reason.

# Optional slide modules

## Module A — Better validation designs
Use four slides to compare random folds, `GroupKFold` for repeated entities, temporal validation for future deployment, and external validation. The supplied CSV has no reliable stable patient identifier for this demonstration.

## Module B — Classification beyond one score
Use four slides on thresholds, precision–recall curves, calibration versus discrimination, and descriptive subgroup checks. State that subgroup differences require uncertainty and governance analysis.

## Module C — Regression beyond summary errors
Use three slides on MAE versus RMSE, residual patterns, and learning curves.

## Module D — From notebook to workflow
Use four slides on parameters versus hyperparameters, tuning without test-set reuse, saving a complete pipeline, and deployment evidence still missing.
