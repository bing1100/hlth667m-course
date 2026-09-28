# Handout 1 — The Pipeline Map

*HLTH 667M Lab 2. One page. Print it and keep it beside the notebook.*

## What a pipeline is

A **pipeline** is one object that holds every step between raw columns and a prediction. You fit the pipeline, not the individual steps.

```
  RAW COLUMNS
       |
       v
  +----------------------------- SPLIT FIRST -----------------------------+
  |                                                                       |
  |   TRAINING PORTION (80%)                    HELD-OUT TEST SET (20%)   |
  |         |                                            |                |
  |         v                                            |                |
  |   +-----------------------------+                    |                |
  |   |        THE PIPELINE         |                    |  DO NOT TOUCH  |
  |   |                             |                    |  until the     |
  |   |  numeric columns            |                    |  very end      |
  |   |    impute -> scale          |                    |                |
  |   |                             |                    |                |
  |   |  categorical columns        |                    |                |
  |   |    impute -> one-hot        |                    |                |
  |   |          \                  |                    |                |
  |   |           -> ESTIMATOR      |                    |                |
  |   +-----------------------------+                    |                |
  |         |                                            |                |
  |         v                                            v                |
  |   cross-validate, compare with baseline      evaluate ONCE            |
  +-----------------------------------------------------------------------+
```

## Why the steps go inside

Every step that **learns something from the data** must live inside the pipeline:

| Step | What it learns | Leaks if fitted outside |
|---|---|---|
| `SimpleImputer` | the median or most frequent value | yes |
| `StandardScaler` | each column's mean and standard deviation | yes |
| `OneHotEncoder` | the set of categories that exist | yes |
| `SelectKBest` | which features correlate with the target | yes, badly |

When you cross-validate a pipeline, scikit-learn refits **all** of these inside each training fold. Do it yourself beforehand and every fold is scored on information it should not have had.

## The code shape

```python
numeric_branch = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

categorical_branch = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric",     numeric_branch,     numeric_features),
    ("categorical", categorical_branch, categorical_features),
])

full_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model",         LogisticRegression(max_iter=2000)),
])
```

## Three things students get wrong

1. **Scaling everything first "to save time."** This is the most common leak in student work.
2. **Dropping the imputer because there are no missing values.** Keep it. It costs nothing and the next dataset will need it.
3. **Forgetting `handle_unknown="ignore"`.** Without it, one unseen category crashes the pipeline at test time.

## Where this appears

- `part-1-ml/Part-1A_Classification_ML_Pipeline.ipynb` sections 3 and 7
- `part-1-ml/Part-1B_Regression_ML_Pipeline.ipynb` sections 3 and 7
- Documentation: <https://scikit-learn.org/stable/modules/compose.html>
