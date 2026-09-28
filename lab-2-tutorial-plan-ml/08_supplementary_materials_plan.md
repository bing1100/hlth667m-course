# 8. Supplementary Materials Plan

## Why separate materials are needed

The core notebooks should remain executable and readable within approximately one hour each. Important concepts that require deeper explanation, manual examples, comparisons, or discussion belong in separate student/instructor materials and the slide deck. The notebooks link to these materials at the point of need.

## Planned `glossary_and_quick_reference.md`

Use one- or two-sentence definitions and a “where used” column.

### Required vocabulary

- observation / row
- feature / predictor / `X`
- target / outcome / `y`
- estimator and model
- fit, transform, and predict
- parameter versus hyperparameter
- classification, class, label, and predicted probability
- regression and residual
- training set, validation fold, and held-out test set
- cross-validation and stratification
- preprocessing, imputation, standardization, and one-hot encoding
- pipeline and `ColumnTransformer`
- baseline
- overfitting and underfitting
- leakage
- accuracy, precision, recall/sensitivity, specificity, F1, macro average
- confusion matrix, threshold, ROC, AUC, and precision–recall
- MAE, RMSE, R², and residual
- calibration and discrimination
- generalization and distribution shift

### Quick-reference tables

1. Classification versus regression.
2. `fit`, `transform`, `fit_transform`, `predict`, and `predict_proba`.
3. Train/test split versus cross-validation.
4. Metric direction: higher-is-better versus lower-is-better.
5. Common scikit-learn objects and their purpose.
6. Common errors and first checks.

## Planned `exercises_and_answer_key.md`

Separate the student prompts from a clearly marked instructor answer key.

### Core exercises

1. **Identify the target:** classify five example targets as classification or regression.
2. **Feature availability:** place dataset fields into “available at admission,” “uncertain timing,” and “future information.”
3. **Manual confusion matrix:** calculate accuracy, precision, recall, and F1 from a small binary example before interpreting the multiclass display.
4. **Metric choice:** choose a classification metric under different false-negative/false-positive consequences.
5. **Manual regression errors:** calculate residuals, MAE, and RMSE for four predictions.
6. **Read cross-validation results:** compare mean performance and fold variability without using the test set.
7. **Interpret weak performance:** write a conclusion when the fitted model does not beat the dummy baseline.
8. **Audit a claim:** revise “the model is 80% accurate, so it is ready for use” into an evidence-bounded statement.

### Optional coding extensions

- Treat `Room Number` as categorical, then compare CV—not test—performance.
- Replace Ridge/logistic regression with a tree-based model and discuss preprocessing differences.
- Add a `GridSearchCV` example using nested or carefully separated evaluation, clearly labelled beyond the core tutorial.
- Save and reload a fitted pipeline with `joblib`, with a warning about version compatibility and untrusted pickle files.

## Planned `troubleshooting.md`

Include direct fixes for:

| Problem | Guidance to provide |
|---|---|
| CSV not found | Confirm exact filename, working directory, upload-to-Colab step, and candidate paths. |
| `ModuleNotFoundError` | Run the documented package-install cell, restart if requested, and rerun from the top. |
| Runtime disconnected/restarted | Reconnect and run all cells from the beginning because variables were cleared. |
| Cells run out of order | Restart kernel and run all; explain hidden state. |
| Unexpected category error | Confirm `OneHotEncoder(handle_unknown="ignore")` is inside the fitted pipeline. |
| Convergence warning | Confirm scaling and `max_iter`; do not silently suppress warnings. |
| Metric warning / undefined precision | Inspect whether a class was never predicted and retain `zero_division=0` with explanation. |
| Negative CV MAE/RMSE | Explain scikit-learn scorer direction and multiply by -1 for display. |
| Negative R² | Explain comparison with the mean baseline; do not take an absolute value. |
| Plot does not display | Ensure the plotting cell completed and `plt.show()` is called. |
| Execution takes too long | Stop the optional extension; core linear pipelines should remain the priority. |

## Deeper concepts for slides or separate handouts

These topics are important but should not expand the core notebooks beyond their one-hour purpose.

### 1. Random, grouped, and temporal validation

- Random folds assume rows are sufficiently independent and similarly distributed.
- Repeated encounters for one person require patient-level grouping to prevent the same person appearing in training and validation.
- Deployment to future admissions may require a chronological split or `TimeSeriesSplit`-style reasoning rather than shuffled folds.
- The available `Name` field is not a reliable patient identifier because of formatting and synthetic-data limitations. Use it only to explain why a proper stable patient ID would be required.
- Link: [visualizing cross-validation behaviour](https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html).

### 2. Probability calibration

- Discrimination asks whether higher scores tend to rank positives above negatives.
- Calibration asks whether predicted probabilities correspond to observed frequencies.
- Cover calibration curves and `CalibratedClassifierCV` conceptually; do not add them to the core multiclass workflow unless time is expanded.
- Links: [probability calibration guide](https://scikit-learn.org/stable/modules/calibration.html) and [`CalibrationDisplay`](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibrationDisplay.html).

### 3. Precision–recall curves and thresholds

- Precision–recall is especially useful when the positive class is uncommon or false positives and false negatives have specific costs.
- Threshold selection requires a stated decision context and separate validation; it is not chosen from the final test set.
- Link: [precision–recall example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html).

### 4. Subgroup performance and fairness

- Overall performance can hide different errors by gender, age group, or another relevant subgroup.
- Small subgroup sample sizes create uncertainty.
- A metric difference does not by itself identify its cause or prove discrimination/fairness.
- The slides can demonstrate a descriptive subgroup table while reserving inferential and governance treatment for a later lesson.

### 5. Learning curves and data sufficiency

- A learning curve compares training and validation performance as training size changes.
- It can help distinguish high variance from limited signal, but it does not establish that collecting more of the same data will solve a validity problem.
- Link: [learning curves](https://scikit-learn.org/stable/modules/learning_curve.html).

### 6. Model assumptions and regularization

- Logistic regression models class log-odds as a linear combination of transformed features.
- Ridge models the outcome as a linear combination and penalizes large coefficients with an L2 penalty.
- Scaling matters for regularization because coefficient penalties depend on feature scale.
- Linear assumptions, correlated features, omitted variables, and coding decisions constrain interpretation.

### 7. Hyperparameters and tuning

- Distinguish learned parameters from settings such as Ridge `alpha` or logistic-regression `C`.
- Introduce `GridSearchCV` as an extension and explain why tuning and final evaluation must remain separated.
- Link: [tuning hyperparameters](https://scikit-learn.org/stable/modules/grid_search.html).

### 8. Uncertainty and repeated evaluation

- CV standard deviation describes variation across the selected folds, not a universal confidence interval.
- A single train/test split is one realization.
- External validation and prospective evaluation answer different questions from internal CV.

### 9. Persistence and deployment boundary

- A fitted pipeline can be saved as one object, preserving preprocessing with the estimator.
- Serialized scikit-learn objects can be version-sensitive and unsafe to load from untrusted sources.
- Persistence does not create monitoring, governance, security, calibration, or clinical validation.
- Link: [model persistence](https://scikit-learn.org/stable/model_persistence.html).

## Optional one-page visual handouts

1. **Pipeline map:** raw columns → split → fold-specific preprocessing → estimator → predictions → metrics.
2. **Data-splitting map:** training portion with five CV folds plus untouched test set.
3. **Metric chooser:** classification and regression questions mapped to candidate metrics.
4. **Leakage checklist:** prediction time, repeated entities, post-outcome variables, preprocessing, and test-set reuse.
