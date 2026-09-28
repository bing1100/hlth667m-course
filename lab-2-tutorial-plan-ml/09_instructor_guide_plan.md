# 9. Instructor Guide Plan

## Purpose

The final `instructor_guide.md` should let an instructor teach the notebooks while moving between slides, code, discussion, and checkpoints. It should not require reading notebook code aloud.

## Before class checklist

- Confirm the CSV is present at the documented local path and prepare a Colab upload copy.
- Run both notebooks from clean kernels with the teaching environment.
- Record expected approximate CV and test metrics without promising exact values.
- Confirm all figures render at projector-readable size.
- Open the slide helper and notebooks in the order used by the run-of-show.
- Decide whether students run cells individually, in pairs, or watch a demonstration.
- Prepare the supplementary glossary and exercise sheet.
- Have a completed executed notebook available if connectivity/runtime fails.

## Classification run-of-show (60 minutes)

| Time | Slides / notebook | Instructor action | Check for understanding |
|---|---|---|---|
| 0–5 | Slides 0–4; notebook title | Define the task and prediction time. | Students identify target and one leaky field. |
| 5–13 | Data loading/audit | Run one cell at a time; pause at each “What do we see?” cell. | Students state class balance and duplicate count. |
| 13–22 | Feature policy and split | Contrast test set with CV folds. | Students explain stratification. |
| 22–34 | Preprocessing/pipeline | Trace one numeric and one categorical field. | Students locate where fitting occurs. |
| 34–46 | Baseline/model/CV | Compare macro F1 and fold variation. | Students state whether the fitted model beats baseline. |
| 46–56 | Test metrics/graphics | Read confusion matrix and ROC without overstating. | Students identify an error pattern and limitation. |
| 56–60 | Exit | Complete metric-choice prompt. | Collect one sentence per student/pair. |

## Regression run-of-show (60 minutes)

| Time | Slides / notebook | Instructor action | Check for understanding |
|---|---|---|---|
| 0–5 | Task framing | Define continuous outcome and residual. | Students distinguish label from amount. |
| 5–13 | Data/outcome distribution | Discuss negative billing values without inventing causes. | Students name one data-owner question. |
| 13–22 | Feature policy and split | Revisit admission-time boundary. | Students explain why test results are excluded. |
| 22–34 | Pipeline and Ridge | Connect scaling to regularization. | Students trace one categorical field. |
| 34–46 | Baseline/model/CV | Interpret errors in dollars. | Students explain scorer negation. |
| 46–56 | Test metrics/residuals | Read all three diagnostic plots. | Students state residual sign and one visible pattern. |
| 56–60 | Exit | Complete error-interpretation prompt. | Collect one bounded conclusion. |

## Likely misconceptions and corrections

| Misconception | Correction |
|---|---|
| “Cross-validation replaces a final test set.” | CV supports development/model comparison; the held-out test set supports one final internal evaluation. |
| “The pipeline transforms all data before splitting.” | The split happens first. During CV, each cloned pipeline fits preprocessing only on its training fold. |
| “No missing values means we do not need an imputer.” | It is retained as a robust workflow component and is fitted only on training data; report that the current CSV has none. |
| “Accuracy is the percentage probability the model is correct.” | Accuracy is an observed fraction correct on a specified evaluation set. It is not a confidence score. |
| “AUC is accuracy.” | AUC summarizes ranking across thresholds; it is not the fraction of correct class labels. |
| “Negative R² means negative correlation.” | It means predictions are worse than the mean-reference model under squared error on that set. |
| “A model near baseline failed to run.” | The code can run correctly while the features provide little useful signal. That is an important result. |
| “The largest coefficient is the most important cause.” | Coefficients depend on coding, scaling, correlation, and model assumptions; predictive association is not causation. |

## Contingency plan

- **No internet:** use local environment and already downloaded materials; official links remain references for later.
- **Colab disconnects:** use the executed backup for discussion, reconnect, and rerun from the top if time allows.
- **Model cell runs slowly:** show the pre-executed result and discuss the workflow; skip optional extensions.
- **Metrics differ slightly:** focus on baseline comparison and interpretation; verify versions and random state after class.
- **Near-baseline results:** do not search for a more impressive model using the test set. Use the result to discuss target signal and data suitability.

## Instructor-only answer key content

- Expected source dimensions, duplicate count, class balance, and billing range.
- Correct feature-eligibility decisions under the stated admission-time assumptions.
- Answers to each notebook checkpoint and supplementary exercise.
- A result-interpretation template that inserts executed metric values.
- Clear flags for statements that overreach the evidence.

## Result statement template

> On five-fold cross-validation of the training portion, the candidate pipeline achieved
> [metric and variation] compared with [baseline result]. On the one-time held-out test set,
> it achieved [test metrics]. These results estimate predictive performance for this supplied
> course dataset under a random split. They do not establish performance for another hospital,
> future time period, patient subgroup, or clinical/administrative deployment.