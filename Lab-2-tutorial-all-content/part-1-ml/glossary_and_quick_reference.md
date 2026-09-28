# Glossary and Quick Reference — Parts 0 to 1B

## Core terms

| Term | Plain-language meaning | Where used |
|---|---|---|
| Observation / row | One recorded admission in Part 0, or one dataset record in Parts 1A/1B. | Data preview |
| Feature / predictor / `X` | Information supplied to a model to make a prediction. | All |
| Target / outcome / `y` | The recorded value a model is asked to predict. | All |
| Estimator | Any scikit-learn object with a `fit` method. | Pipelines |
| Parameter | A value learned from the data, such as a coefficient. | Models |
| Hyperparameter | A setting you choose, such as Ridge `alpha`. | Models |
| Classification | Predicting a category. | Part 1A |
| Regression | Predicting a continuous number. | Part 1B |
| Fit | Estimate settings from training data. | Pipelines |
| Transform | Turn input columns into model-ready columns. | Imputation, scaling, encoding |
| Predict | Produce a label or numeric estimate. | Final evaluation |
| Baseline | A trivial reference a fitted model should beat. | Dummy models |
| Imputation | Filling missing values with a learned substitute. | Preprocessing |
| Standardisation | Rescaling a column to mean 0, standard deviation 1. | Preprocessing |
| **One-hot encoding** | One 0/1 column per category, so no false ordering is implied. | Part 1A §7 |
| **`ColumnTransformer`** | Applies different preprocessing to different columns in one object. | Part 1A §7 |
| Cardinality | How many distinct values a column has. | Part 0 audit |
| Prediction time | The moment the prediction would actually be made. | Part 0 §6 |
| **Leakage** | Using information unavailable at prediction time, or letting evaluation data influence training. | Part 0 §6, Part 1B §7 |
| Held-out test set | Data reserved for one final internal evaluation. | Final evaluation |
| Cross-validation | Repeated train/validation splits inside the training portion. | Model comparison |
| Stratification | Keeping class proportions similar across splits. | Part 1A |
| Overfitting | Fitting noise in the training data, so performance does not transfer. | Interpretation |
| Residual | Actual value minus predicted value. | Part 1B |
| Regularisation | A penalty on coefficient size, as in Ridge. | Part 1B |
| Discrimination | Whether higher scores rank positives above negatives. | ROC-AUC |
| Calibration | Whether predicted probabilities match observed frequencies. | Part 1A §6 caution |
| Distribution shift | The deployment population differs from the training data. | Limitations |

## Classification and regression at a glance

| Question | Classification | Regression |
|---|---|---|
| Target type | Category | Continuous number |
| Tutorial dataset | `load_breast_cancer()` | `load_diabetes()` |
| Baseline | Most-common / prior class | Training mean |
| Main model | Logistic regression | Ridge regression |
| Splitter | `StratifiedKFold` | `KFold` |
| Primary CV metric | Macro F1 (higher is better) | MAE (lower is better) |
| Main diagnostic | Confusion matrix | Residual plots |

## scikit-learn methods

| Method | Meaning |
|---|---|
| `fit(X, y)` | Learn transformation or model settings from training data. |
| `transform(X)` | Apply a fitted transformation. |
| `fit_transform(X)` | Fit, then transform training data. |
| `predict(X)` | Produce labels or numeric predictions. |
| `predict_proba(X)` | Produce class probabilities where supported. |
| `get_feature_names_out()` | Show the column names a transformer produced. |

## Metric direction

| Metric | Better | Meaning |
|---|---|---|
| Accuracy, precision, recall, F1, balanced accuracy, ROC-AUC, R² | Higher | More correct labels, better ranking, or better fit than the mean |
| MAE, RMSE | Lower | Smaller error, in the outcome's units |
| `neg_mean_absolute_error`, `neg_root_mean_squared_error` | Higher internally | scikit-learn negates losses; the notebooks multiply by −1 for display |

## Three numbers students misread

1. **Accuracy on unbalanced classes.** In Part 1A the dummy scores 0.63 accuracy and 0.39 macro F1 on identical predictions, while never identifying a malignant case.
2. **Negative R².** Worse than predicting the mean of the evaluation set. Not a negative correlation.
3. **Negative MAE in CV output.** A scikit-learn convention, not a bug.

## Official documentation

- [Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html)
- [Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)

## Printable handouts

`../handouts/pipeline_map.md` · `../handouts/data_splitting_map.md` · `../handouts/metric_chooser.md` · `../handouts/leakage_checklist.md`
