# 2. Classification Notebook Blueprint

## Notebook identity

- **Filename:** `Part-1A_Classification_ML_Pipeline.ipynb`
- **Duration:** approximately 60 minutes
- **Question:** Using information available at admission, can a model classify the recorded `Test Results` category?
- **Target:** `Test Results`
- **Primary selection metric:** macro F1 from 5-fold cross-validation on training data
- **Final metrics:** accuracy, balanced accuracy, macro precision, macro recall, macro F1, multiclass ROC-AUC (one-vs-rest), confusion matrix

## Required notebook sequence

The following is a cell-level specification. Each numbered code cell must be preceded by a Markdown explanation that meets the standard in `README.md`.

Use visible numbered headings: `# 0. Before You Begin` through `# 9. What's Next?`. Do not present the notebook as a long uninterrupted sequence of code. After important outputs, add a short **“What do we see?”** Markdown cell before moving to the next operation.

| # | Markdown focus | Code/output requirement |
|---:|---|---|
| 0 | Title, objectives, timing, prerequisites, data-use boundary, definition of multiclass classification, and table of contents. | No code. Include a short roadmap, estimated runtime, and “run cells from top to bottom” instruction. |
| 1 | `# 0. Before You Begin`: explain imports, plotting defaults, version reporting, reproducibility, and that no installation is needed if the environment already has the packages. | Import pandas, numpy, matplotlib, seaborn, scikit-learn classes/metrics; define `RANDOM_STATE = 42`; print versions. Include an optional commented `%pip install` recovery cell only if needed. |
| 2 | Explain local versus Colab source paths, how to upload the CSV to Colab, and why data are copied after reading. | Search a short ordered list of documented candidate paths; raise a helpful `FileNotFoundError`; read CSV; display shape and first rows. |
| 2a | **What do we see?** Define row, column, observation, feature, and target using the displayed table. | No code. State the expected 55,500 × 15 source shape. |
| 3 | Explain an initial audit and that no-missingness today does not remove the need for robust preprocessing. | Display dtypes, missing-value counts, and exact duplicate count. |
| 3a | **What do we see?** State that the current file has no missing values and contains exact duplicates. Clarify that this is a data audit, not yet preprocessing. | No code. |
| 4 | Explain class frequency and why it influences metric interpretation and split design. Define class balance. | Compute counts/proportions; create labeled class-count bar chart. |
| 4a | **What do we see?** Explain that the three labels are close in size and why this does not make all metrics interchangeable. | No code. |
| 5 | Explain duplicate removal and why it occurs before splitting. | `drop_duplicates()`, report removed and remaining rows. |
| 6 | `# 2. Define the Prediction Task`: explain the admission-time question, safe features, excluded identifiers, final billing, ambiguous medication timing, and future-information leakage. | Define task-specific numeric/categorical feature lists and exclusions in code; print lists. Exclude `Medication`, `Billing Amount`, and `Discharge Date`. |
| 7 | Explain date parsing and minimal feature engineering. | Parse admission date; report null parse count; derive year/month/day-of-week; remove raw date. |
| 7a | **Checkpoint 1:** Ask students to name one excluded field and the exact reason it is unavailable or inappropriate. | No code. Give a collapsible/clearly separated suggested answer after the prompt. |
| 8 | `# 3. Create Training and Test Data`: explain `X`, `y`, stratification, and the protected final test set. Link `train_test_split`. | Create 80/20 stratified split; display train/test shapes and class proportions. |
| 8a | **What do we see?** Compare train/test class proportions and explain that the test set will not be used in CV or model selection. | No code. |
| 9 | Explain the numeric/categorical transformation branches and why fitting is deferred to the pipeline. Link `ColumnTransformer`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder`. | Construct numeric and categorical pipelines and a `ColumnTransformer`; print representation. |
| 9a | **What do we see?** Trace one numeric and one categorical feature through the transformer in plain language. | No code. |
| 10 | `# 4. Build Complete Pipelines`: explain an end-to-end `Pipeline`, preventing preprocessing leakage, and estimator replacement. Link `Pipeline`. | Build a dummy pipeline and logistic-regression pipeline. |
| 11 | Explain a prior-probability baseline and why accuracy alone can mislead. Link `DummyClassifier`. | Define `DummyClassifier(strategy="prior")` pipeline. |
| 12 | Explain multinomial logistic regression, regularization, and `max_iter`. Link `LogisticRegression`. | Define pipeline with `LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)`. |
| 13 | Explain stratified 5-fold CV and pre-specified multi-metric scoring. Link `StratifiedKFold` and `cross_validate`. | Define CV splitter and scoring dictionary. |
| 14 | Explain that comparison is conducted on training data only. Explain negative-score conventions only if used. | Run `cross_validate` for baseline and model; make mean/SD results table. |
| 14a | **What do we see?** State whether logistic regression materially exceeds the dummy baseline and note fold variation. Do not promise an impressive result. | No code. Interpret generated values using cautious conditional wording. |
| 15 | Explain primary metric selection and fold-to-fold variability. | Plot model/baseline macro-F1 means with SD error bars; optionally show all fold values. |
| 15a | **Checkpoint 2:** Ask why the final test set does not appear in this comparison. | No code; include a concise suggested answer. |
| 16 | Explain refitting only the selected pre-specified model on all training rows. | Fit logistic pipeline; predict labels and probabilities on test rows. |
| 17 | Explain each final summary metric and classification report support. Link metric APIs. | Calculate/display accuracy, balanced accuracy, macro precision, macro recall, macro F1, classification report. |
| 18 | Explain confusion-matrix axes, diagonal cells, and error patterns; warn it does not establish impact/cost. | Create a clearly labelled `ConfusionMatrixDisplay` with class labels. |
| 19 | Explain probabilities, one-vs-rest ROC curves, AUC, and the limitation of threshold-free discrimination summaries. | Binarize labels; plot one ROC curve per class; compute macro OVR ROC-AUC. |
| 19a | **What do we see?** Guide students to compare curves with the diagonal and to avoid treating AUC as clinical usefulness. | No code. |
| 20 | `# 7. Interpret and Limit the Result`: explain performance against baseline, likely weak signal, assumptions, dataset limitations, subgroup uncertainty, and non-deployment status. | No code or a concise summary table generated from already-computed values. |
| 21 | `# 8. Try It Yourself`: change the stated cost of one error and select a metric; optionally compare normalized and count confusion matrices without retraining. | Short learner task with expected output/answer guidance. |
| 22 | `# 9. What's Next?`: point to the glossary, exercises, troubleshooting, slides, official docs, and optional topics. | No code. End with a completion checklist. |

## Classification graphics: required details

1. **Class-count bar chart:** display counts and percentages; use a colorblind-friendly palette; title must state this is the target distribution after duplicates are removed.
2. **CV macro-F1 comparison:** show dummy and logistic-regression mean scores plus standard-deviation error bars. Clearly label that this is 5-fold CV on training data, not test performance.
3. **Confusion matrix:** use a stable class order; label axes “True label” and “Predicted label.” Prefer a normalized second display only if time permits; never replace counts without explaining normalization.
4. **One-vs-rest ROC:** label each curve with its AUC; include a diagonal no-discrimination reference line; explain that three classes require one-versus-rest treatment here.

## Classification pedagogical cautions

- The observed class balance means raw accuracy is less deceptive here than in an imbalanced dataset, but it remains insufficient because it hides per-class errors and error consequences.
- Do not tune thresholds, tune hyperparameters, or select a model after repeated inspection of the held-out test set.
- Do not claim ROC-AUC is a probability-calibration assessment, a utility measure, or a clinical validation result.
- Do not claim coefficient magnitude is feature importance without careful treatment of scaling, category reference structure, and correlations; defer model interpretation to a later tutorial.

## Expected-output notes

- The source-data preview should report 55,500 rows and 15 columns before duplicate removal.
- The audit should report 534 exact duplicate rows for the supplied file.
- The class proportions should be approximately one third each.
- Model performance may be close to chance/baseline because the safe admission-time predictors contain little obvious target signal. The notebook must treat this as a valid finding.
- Exact metric values may differ slightly across supported scikit-learn versions. Explanatory Markdown should describe patterns and comparisons rather than hard-code all values.