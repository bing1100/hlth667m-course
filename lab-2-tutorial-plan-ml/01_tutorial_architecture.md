# 1. Tutorial Architecture and Shared Design Decisions

## Learning outcomes

After completing either notebook, students should be able to:

1. Distinguish a feature, target, prediction, classification task, and regression task.
2. Define a prediction-time boundary and identify data leakage.
3. Split data into development and final held-out test sets, and use cross-validation within development data.
4. Build a scikit-learn `Pipeline` containing preprocessing and an estimator.
5. Compare a fitted model with a meaningful dummy baseline.
6. Select, compute, graph, and interpret metrics appropriate to the target type.
7. State limitations of a model result and why predictive performance does not establish clinical usefulness, causation, fairness, or readiness for deployment.

## Independent-notebook design

The notebooks must not depend on one another. Both load the CSV, define a documented data-cleaning policy, create their own features, construct their own preprocessing workflow, and run to completion from a fresh kernel. Repeating some concepts is intentional: it makes each approximately one-hour tutorial useful as a standalone class activity.

| Notebook | Target | Task type | Main model | Baseline | Cross-validation |
|---|---|---|---|---|---|
| `Part-1A_Classification_ML_Pipeline.ipynb` | `Test Results` | Three-class classification | `LogisticRegression` | `DummyClassifier(strategy="prior")` | `StratifiedKFold`, 5 folds |
| `Part-1B_Regression_ML_Pipeline.ipynb` | `Billing Amount` | Continuous regression | `Ridge` | `DummyRegressor(strategy="mean")` | `KFold`, 5 folds |

An optional second estimator may be shown only as a clearly marked extension after the core pipeline works. Do not turn this first part into a hyperparameter-search exercise.

## Suggested timing per notebook (about 60 minutes)

| Segment | Minutes | Student outcome |
|---|---:|---|
| Framing, target, and prediction-time boundary | 5 | Can articulate what is predicted and when. |
| Data audit, duplicates, and target/outcome plot | 8 | Can recognize data structure and basic quality concerns. |
| Split and cross-validation concepts | 8 | Can distinguish a held-out test set from CV folds. |
| Feature engineering and preprocessing pipeline | 12 | Can explain why transformations are inside the pipeline. |
| Baseline, model training, and CV comparison | 12 | Can interpret a comparison table/plot. |
| Test-set metrics and diagnostic graphics | 10 | Can read the principal performance displays. |
| Limitations and exit prompt | 5 | Can make an evidence-bounded interpretation. |

## Prediction-time and feature policy

Frame both exercises as **admission-time predictions**. Only information available by admission should be supplied to the models.

### Task-specific starting features

Do not use one undifferentiated feature list for both targets. A variable can be valid for one prediction question and invalid for another.

| Feature | Classification: predict `Test Results` at admission | Regression: predict final `Billing Amount` at admission | Rationale / question to state |
|---|---|---|---|
| `Age` | Include | Include | Available at admission. |
| `Gender` | Include | Include | Available, but subgroup performance and social meaning require caution. |
| `Blood Type` | Include | Include | Assume it is recorded by admission for this exercise. |
| `Medical Condition` | Include | Include | Assume it is the admission diagnosis/category. |
| `Insurance Provider` | Include | Include | Available administratively; may encode access and structural inequity. |
| `Room Number` | Include as a teaching choice | Include as a teaching choice | Numeric code may not be a meaningful quantity. Prompt students to critique coding/exclusion. |
| `Admission Type` | Include | Include | Available at admission. |
| `Date of Admission` | Derive calendar fields | Derive calendar fields | Available at admission; calendar patterns may not generalize over time. |
| `Medication` | Exclude from core | Exclude from core | Timing is unclear and may be downstream of test results or care decisions. |
| `Billing Amount` | Exclude | Target | Final billing is unavailable when classifying an admission-time test result. |
| `Test Results` | Target | Exclude | Test status may be unavailable at admission and could affect later costs. |

The notebook must state these are assumptions for a teaching example. In a real project, a data dictionary and workflow owner would be required to establish when each field becomes available.

### Excluded fields and rationale

| Field | Decision | Reason |
|---|---|---|
| `Name` | Exclude | Near-identifier; high cardinality; likely memorization, privacy concerns, and poor generalization. |
| `Doctor` | Exclude | High cardinality; likely memorization and contextual/site effects. |
| `Hospital` | Exclude | High cardinality; may encode site/process rather than transferable patient signal. |
| `Discharge Date` | Exclude | Not available at admission; future-information leakage. |
| Derived length of stay | Exclude | Depends on discharge date; leakage under admission-time framing. |
| `Medication` | Exclude from both core pipelines | Timing is ambiguous and may be downstream. |
| `Test Results` | Target in classification; exclude from regression | Must not predict itself; may be unavailable at the regression prediction time. |
| `Billing Amount` | Exclude from classification; target in regression | Final bill is unavailable at the prediction point. |

### Required leakage demonstration

Include a visible Markdown callout before feature selection:

> Leakage occurs when the training workflow uses information unavailable at the time a prediction would be made, or when information from a validation/test set influences preprocessing or model selection. It can create impressive-looking but invalid performance estimates.

Explain both forms:

1. **Temporal/outcome leakage:** using `Discharge Date` for an admission-time prediction.
2. **Preprocessing leakage:** fitting encoders, scalers, imputers, or feature selectors on all rows before cross-validation.

## Shared technical workflow

1. Read the CSV and make a `.copy()`.
2. Report shape, types, missingness, and exact duplicates.
3. Remove exact duplicates using `drop_duplicates()` before any split; report rows removed.
4. Parse `Date of Admission` with `pd.to_datetime(..., errors="coerce")` and report parsing failures.
5. Derive `admission_year`, `admission_month`, and `admission_dayofweek`; then drop the original date from model input.
6. Define `X` and `y` using only the approved feature list.
7. Reserve 20% of rows for a final test set using `random_state=42`; classification uses `stratify=y`.
8. Create a `ColumnTransformer` with numeric and categorical sub-pipelines.
9. Join preprocessing and estimator with `Pipeline`.
10. Compare baseline and candidate using CV on the training portion only.
11. Select the tutorial’s candidate model based on the pre-specified primary metric, not test-set peeking.
12. Fit the selected workflow on all development/training data once and evaluate once on the held-out test set.

## Preprocessing specification

```python
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])
```

The imputer remains despite the current absence of missing values. Teach it as a robustness pattern, not a claim that the CSV has missing values. `handle_unknown="ignore"` prevents a new category in the test set from crashing the encoder; it does not make an unseen category informative.

## Reproducibility and runtime rules

- Define `RANDOM_STATE = 42` in each notebook and use it for all applicable splitters and estimators.
- Print Python and core package versions in an early cell.
- Use fixed column lists, not accidental positional selection.
- Do not mutate the original source CSV.
- Avoid external data, API keys, downloads, or patient data.
- Use `n_jobs=-1` only where the selected function supports it and explain that it uses available CPU cores; do not assume a particular hosted runtime.

## Learner-facing notebook structure

Both notebooks follow the progressive tutorial structure used in the referenced CTGAN notebook:

1. **Title and purpose** — what will be predicted and why this is a teaching example.
2. **0. Before you begin** — prerequisites, runtime, file location, packages, and how to restart.
3. **1. Load and inspect the data** — one operation at a time, followed by short output interpretation.
4. **2. Define the prediction task** — target, prediction time, eligible features, and leakage.
5. **3. Create training and test data** — holdout and CV diagrams/explanations.
6. **4. Build the preprocessing pipeline** — numeric branch, categorical branch, and full pipeline.
7. **5. Train and compare models** — dummy baseline, candidate, CV, and comparison graphic.
8. **6. Evaluate once on test data** — metrics and task-specific diagnostic displays.
9. **7. Interpret and limit the result** — what the output supports and does not support.
10. **8. Try it yourself** — one short modification that does not use the test set for selection.
11. **9. What’s next and where to get help** — references, supplementary materials, and troubleshooting.

Important outputs should be followed by a short Markdown cell beginning **“What do we see?”** This mirrors the referenced notebook’s pattern of explaining results after they appear. It must describe the expected pattern without hard-coding values that could change across library versions.
