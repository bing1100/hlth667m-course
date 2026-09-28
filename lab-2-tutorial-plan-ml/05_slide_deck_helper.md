# 5. Slide Deck Helper — Pipeline Walkthrough

## Purpose and format

The final `slides.md` is a presenter-ready helper for use alongside the two notebooks. It follows the course convention: each slide includes visible text, a suggested visual and alt text, instructor wording, a transition, and an interaction/check for understanding. It is not a `.pptx`; the content can be copied into presentation software.

The deck should use the same terms, target definitions, and variable names as the notebooks. Insert visible labels such as **“Now run Classification Cell 8”** so the presenter can alternate between concepts and live code.

## Proposed core 20-slide sequence

| Slide | Visible focus | Visual / notebook connection | Speaker prompt or learner check |
|---:|---|---|---|
| 0 | Title, roadmap, and outcomes | Two paths: categorical label vs continuous amount | “Today we build and critique two complete ML pipelines.” |
| 1 | What is a prediction problem? | Features → model → prediction diagram | Ask students to identify target versus feature. |
| 2 | Classification versus regression | `Test Results` versus `Billing Amount` cards | “The model type follows the target, not the dataset name.” |
| 3 | The prediction-time boundary | Admission-time timeline | Ask whether discharge date is available at admission. |
| 4 | Leakage can look like success | Leaky/future data arrow into a model | Name temporal leakage and preprocessing leakage. |
| 5 | First look at the data | Target-count graphic from classification notebook | “What does this plot tell us, and what does it not tell us?” |
| 6 | Duplicates and data quality | Duplicate rows feeding split illustration | “Why remove exact duplicates before splitting?” |
| 7 | Train, cross-validate, test | 80/20 holdout plus five folds inside training set | Students point to the one set that stays untouched during selection. |
| 8 | Why a pipeline? | Raw columns → transformer branches → estimator | “Where should the scaler and encoder learn their parameters?” |
| 9 | Numeric and categorical transformations | Median/scaler and imputer/one-hot branches | Explain `handle_unknown="ignore"` in one sentence. |
| 10 | Classification baseline and candidate | Prior-only predictor versus logistic regression | “What does the dummy model establish?” |
| 11 | Classification metrics | Accuracy, precision, recall, macro F1 cards | Give a false-negative scenario; ask which metric matters. |
| 12 | Read the confusion matrix | Annotated matrix from notebook | Have students find the largest off-diagonal cell. |
| 13 | Scores, probabilities, and ROC | One-vs-rest ROC sketch | “AUC is not a threshold, calibration, or clinical benefit.” |
| 14 | Regression baseline and candidate | Mean-only predictor versus Ridge | “Why is mean prediction a useful benchmark?” |
| 15 | Regression metrics | MAE, RMSE, R² cards with units | Ask why RMSE reacts more strongly to a large error. |
| 16 | Actual vs predicted | Scatter around 45-degree line from notebook | Ask what points far from the line mean. |
| 17 | Residual diagnostics | Residual plot with zero line | Ask whether errors appear centered and evenly spread. |
| 18 | Results are not deployment evidence | Four limits: data, fairness, workflow, validation | “What evidence is still missing before a real use?” |
| 19 | Exit reflection | Prompt card | Students write one safeguard and one metric justification. |

## Supplementary slide modules

These modules can be inserted when the instructor has more time or used in a later session. They add conceptual depth without placing more code into the core notebooks.

### Module A — Better validation designs (4 slides)

1. **Rows are not always independent:** repeated patients, clinicians, or hospitals.
2. **Grouped validation:** keep all records for one entity in one fold; introduce `GroupKFold`.
3. **Temporal validation:** train on earlier admissions and evaluate on later admissions.
4. **Choose the split that matches intended use:** random, grouped, temporal, and external validation comparison.

### Module B — Classification beyond one score (4 slides)

1. **Thresholds change errors:** probability scores become labels through a decision rule.
2. **Precision–recall:** focus on positive predictions and identified cases.
3. **Discrimination versus calibration:** ranking is different from probability accuracy.
4. **Subgroup checks:** overall averages can hide different errors; uncertainty and governance remain necessary.

### Module C — Regression beyond summary errors (3 slides)

1. **Why compare MAE and RMSE:** different sensitivity to large errors.
2. **Residual patterns:** nonlinearity, changing spread, and extreme errors as questions for follow-up.
3. **Learning curves:** whether validation performance changes with more training rows.

### Module D — From notebook to responsible workflow (4 slides)

1. **Parameters versus hyperparameters:** what fitting learns and what analysts choose.
2. **Tuning without test-set reuse:** CV search followed by one final evaluation.
3. **Saving the full pipeline:** persistence benefits and security/version limitations.
4. **Deployment evidence still missing:** external validation, monitoring, calibration, subgroup performance, workflow fit, and governance.

## Required deck content for each slide

Use this exact outline in the final helper:

```markdown
## Slide N — Plain-language title (X min)
**Visible text:** ...
**Visual:** ...
**Alt text:** ...
**Say:** “...”
**Ask:** “...”
**Expected:** “...”
**Notebook transition:** Run `Part-1A...`, Cell N / `Part-1B...`, Cell N.
```

## Slide-deck safeguards

- Do not place code blocks or full metric formulas as dense paragraphs on the slides; reserve detail for the notebook and speaker notes.
- Every visual needs usable alt text.
- State that the dataset is course material and not patient data; do not imply real clinical validation.
- Use baseline comparisons prominently. A sophisticated-looking chart must not obscure the question, “Did this beat a trivial predictor?”
- If classification performance is near baseline, present that honestly as a lesson about features, targets, and evaluation rather than trying to find a more impressive result through test-set iteration.
- Use the supplied worked confusion-matrix and regression-error examples before asking students to interpret the larger notebook outputs.
- Provide a text statement below any slide that depends on color, line style, or plot position.

## Revision of 21 September 2026

The rendered deck is `../latex-slides/lab2-slides.pdf`. Each key notebook section now has an explanation slide followed by a "From the notebook" slide. For Tutorial 1 the embedded figures are the Part 0 distributions and billing-by-admission-type plots, the Part 1A confusion matrix and ROC curve, and the Part 1B actual-versus-predicted and residual plots. Slides that redraw a notebook table natively carry a tag naming the part and section. `latex-slides/export_notebook_figures.py` (`make figures`) re-exports the figures after a notebook is re-executed.
