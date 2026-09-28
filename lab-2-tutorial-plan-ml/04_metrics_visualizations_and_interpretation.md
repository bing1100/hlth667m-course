# 4. Metrics, Visualizations, and Interpretation Guide

## Classification metrics

For a one-versus-rest class, define true positives (TP), false positives (FP), false negatives (FN), and true negatives (TN).

| Metric | Formula | Student-facing interpretation | Important caution |
|---|---|---|---|
| Accuracy | `(TP + TN) / all predictions` | Fraction of all predictions that are correct. | Can conceal poor performance for a minority or high-consequence class. |
| Precision | `TP / (TP + FP)` | Of records predicted as this class, how many truly belong to it? | Does not show how many true cases were missed. |
| Recall / sensitivity | `TP / (TP + FN)` | Of records truly in this class, how many did the model identify? | Does not show how many incorrect alerts/predictions were made. |
| F1 | `2 × precision × recall / (precision + recall)` | A single score that balances precision and recall. | Does not encode all real-world error costs. |
| Balanced accuracy | Mean recall across classes | Gives each class equal weight through recall. | Does not include precision. |
| Macro average | Unweighted mean of a per-class metric | Each class contributes equally, regardless of count. | An appropriate average still requires per-class inspection. |
| ROC-AUC (OVR) | Area under TPR vs FPR curve | How well scores rank one class above all other classes across thresholds. | Does not prove calibration, usefulness, fairness, or a correct threshold. |

### Required classification computations

```python
accuracy_score(y_test, y_pred)
balanced_accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred, average="macro", zero_division=0)
recall_score(y_test, y_pred, average="macro", zero_division=0)
f1_score(y_test, y_pred, average="macro", zero_division=0)
classification_report(y_test, y_pred, zero_division=0)
roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
```

Use `zero_division=0` so a metric remains defined if a model makes no predictions for a class; explain that the resulting zero is a warning about the model, not a technical success.

## Regression metrics

Let `y_i` be the actual amount, `ŷ_i` the prediction, `n` the number of rows, and `ȳ` a reference mean.

| Metric | Formula | Student-facing interpretation | Important caution |
|---|---|---|---|
| MAE | `Σ |y_i − ŷ_i| / n` | Typical absolute prediction error, in billing-amount units. | Treats a $100 overprediction and $100 underprediction equally. |
| RMSE | `sqrt(Σ(y_i − ŷ_i)² / n)` | Error in original units that penalizes larger misses more strongly. | A few large errors can dominate it. |
| R² | `1 − Σ(y_i − ŷ_i)² / Σ(y_i − ȳ)²` | Relative fit compared with a mean-based reference. | Can be negative on held-out data; is not percentage accuracy. |

### Required regression computations

```python
mean_absolute_error(y_test, y_pred)
root_mean_squared_error(y_test, y_pred)
r2_score(y_test, y_pred)
```

For cross-validation, scikit-learn represents losses as negative scorers because its model-selection interface follows “larger score is better.” Therefore:

```python
scoring = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}
# Display errors after multiplying the negative CV values by -1.
```

## Graphic interpretation prompts

| Graphic | Required question | Misinterpretation to prevent |
|---|---|---|
| Class-count bar chart | Are classes equally represented? | “Balanced classes mean accuracy is enough.” |
| CV error-bar comparison | Does the candidate beat baseline consistently across folds? | “The highest mean alone proves superiority.” |
| Confusion matrix | Which true/predicted class pair is the most common error? | “A diagonal total says nothing about consequences.” |
| ROC curve | Does the curve rise above the no-discrimination diagonal? | “AUC selects the deployment threshold.” |
| Billing histogram/boxplot | What is the center, spread, and presence of unusual values? | “Unusual values must be deleted.” |
| Actual vs predicted | How close are points to the perfect-prediction line? | “A cloud around the line proves the model is useful.” |
| Residual vs predicted | Is residual spread centered near zero and roughly stable? | “A random-looking plot validates all assumptions.” |
| Residual histogram | Are errors symmetric/centered, and are tails notable? | “A normal-looking residual histogram makes the result causal.” |

## Documentation links to include in notebook Markdown

- [scikit-learn model evaluation guide](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Classification metrics](https://scikit-learn.org/stable/api/sklearn.metrics.html#classification-metrics)
- [Regression metrics](https://scikit-learn.org/stable/api/sklearn.metrics.html#regression-metrics)
- [`classification_report`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
- [`ConfusionMatrixDisplay`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html)
- [`RocCurveDisplay`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.RocCurveDisplay.html)
- [`precision_recall_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html)
- [`mean_absolute_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)
- [`root_mean_squared_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_error.html)
- [`r2_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)