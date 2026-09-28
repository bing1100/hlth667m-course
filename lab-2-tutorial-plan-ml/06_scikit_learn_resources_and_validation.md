# 6. Scikit-learn Resources, Dependencies, and Validation Plan

## Official scikit-learn resources

All notebook instructional links should point to official scikit-learn documentation unless the purpose is explicitly to provide a separate conceptual reading.

### Core workflow

- [Getting started](https://scikit-learn.org/stable/getting_started.html)
- [Choosing the right estimator](https://scikit-learn.org/stable/machine_learning_map.html)
- [Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html)
- [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [`ColumnTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
- [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html)
- [Computing cross-validated metrics](https://scikit-learn.org/stable/modules/cross_validation.html)

### Data preparation

- [`SimpleImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)
- [`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [`OneHotEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
- [Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html)

### Splits and validation

- [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- [`KFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)
- [`StratifiedKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)
- [`cross_validate`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html)
- [Cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html)
- [`GroupKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
- [`TimeSeriesSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [Visualizing cross-validation behaviour](https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html)

### Models and baselines

- [`DummyClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)
- [`DummyRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html)
- [`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [`Ridge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- [Linear models guide](https://scikit-learn.org/stable/modules/linear_model.html)

### Metrics, scoring, and display tools

- [Model evaluation guide](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Scoring parameter](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)
- [`make_scorer`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html)
- [`accuracy_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
- [`balanced_accuracy_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html)
- [`precision_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)
- [`recall_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)
- [`f1_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
- [`roc_auc_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
- [`ConfusionMatrixDisplay`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html)
- [`RocCurveDisplay`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.RocCurveDisplay.html)
- [`mean_absolute_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)
- [`root_mean_squared_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_error.html)
- [`r2_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)

### Supplementary topics

- [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [`CalibrationDisplay`](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibrationDisplay.html)
- [Precision–recall example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)
- [Learning curves](https://scikit-learn.org/stable/modules/learning_curve.html)
- [Hyperparameter tuning](https://scikit-learn.org/stable/modules/grid_search.html)
- [Model persistence](https://scikit-learn.org/stable/model_persistence.html)
- [Inspection and permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html)

## Planned dependency file

Use a deliberately small dependency set. The final `requirements.txt` should contain compatible minimum versions, not unnecessarily rigid full-environment pins:

```text
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.4
jupyter>=1.0
```

The current planning environment contains pandas 2.2.3, numpy 2.2.6, matplotlib 3.10.0, seaborn 0.13.2, scikit-learn 1.6.1, Jupyter 1.1.1, nbformat 5.10.4, and nbclient 0.10.2. The final materials must remain compatible with the stated minimums; `root_mean_squared_error` requires scikit-learn 1.4 or later.

## Notebook execution validation

Before delivery, perform all checks from `/home/bhux/research/proposals/hlth667m-course/` or an equivalent controlled environment.

1. **Structural validation:** parse both `.ipynb` files with `nbformat.read`; verify notebook metadata and that every code cell is preceded by Markdown.
2. **Instructional validation:** inspect each Markdown cell against `07_notebook_markdown_style_guide.md`. Verify numbered sections, plain-language definitions, What/Why/Look-for guidance, post-output “What do we see?” cells, checkpoints, “Try It Yourself,” and “What’s Next?” sections.
3. **Fresh-kernel execution:** execute each notebook top-to-bottom with `nbclient` and the documented data path. No hidden execution state may be required.
4. **Output validation:** verify that tables and all required graphics are produced: classification distribution/CV/confusion matrix/ROC and regression distribution/CV/actual-vs-predicted/residual plots.
5. **Metric validation:** confirm classification uses stratified holdout and `StratifiedKFold`; regression CV displays positive MAE/RMSE after conversion from negative scorers.
6. **Leakage validation:** inspect final feature lists to ensure `Discharge Date`, `Name`, `Doctor`, `Hospital`, and the active target never enter `X`.
7. **Baseline validation:** confirm dummy pipelines use exactly the same preprocessing object structure as the candidate model, so the model comparison changes the estimator rather than the data treatment.
8. **Reproducibility validation:** execute notebooks twice from clean kernels and confirm split sizes and reported values are stable under `RANDOM_STATE = 42`.
9. **Runtime validation:** record elapsed execution time. The core run should be appropriate for a classroom/Colab CPU and should avoid expensive searches.
10. **Accessibility validation:** inspect titles, axis labels, legends, color contrast, explanatory captions, and slide alt text.
11. **Setup validation:** run once with the source absolute path and once using the documented Colab/upload fallback path logic.
12. **Recovery validation:** restart the kernel and confirm that “Run All” succeeds without manually created hidden variables.
13. **Expected-output validation:** confirm the notebooks report the known source shape, duplicate count, target distributions, and concise conditional interpretations.
14. **Companion-material validation:** check that every notebook reference to the glossary, exercise sheet, troubleshooting, slides, and instructor guide points to a planned final file.

## Acceptance criteria

The tutorial implementation is complete only when:

- Both notebooks run independently from top to bottom.
- Both use the supplied dataset and explicitly state its course/educational status.
- Every code cell has an explanatory preceding Markdown cell.
- Important displayed outputs have a following “What do we see?” interpretation cell.
- Numbered sections, checkpoints, a learner exercise, completion checklist, setup guidance, and “What’s Next?” are present in each notebook.
- Each notebook provides a complete pipeline rather than manual, pre-split preprocessing.
- Each notebook compares a baseline and candidate model using CV before a final test evaluation.
- Classification includes metrics and graphics suited to a multiclass target.
- Regression includes metrics and diagnostics suited to a continuous target.
- No result is overstated as clinical validation, fairness evidence, causation, or deployment readiness.
- The slide helper maps concepts and live notebook cells in a presenter-ready form.
- Glossary, exercises/answer key, troubleshooting, and instructor guide are complete and consistent with the executed notebook results.