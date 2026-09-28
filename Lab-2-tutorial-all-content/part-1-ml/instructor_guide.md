# Instructor Guide — Parts 0 to 1B

## Before class

- Run all three notebooks from fresh kernels and keep the executed copies as a no-network contingency.
- Run `python -m pytest -q` from this folder (expect 27 passing).
- Confirm the CSV is found by Part 0 and by Part 1A section 7, and prepare the Colab upload step if students will use Colab.
- Print `../handouts/` — the leakage checklist and metric chooser are the two students actually keep.
- Record approximate results, but do not promise exact values across environments.
- Decide whether students run cells themselves, work in pairs, or watch a demonstration.

## Part 0 — healthcare processing and analytics (30–45 minutes)

The point of this notebook is **not** the pandas. It is that most of the work happens before any model exists, and that every step ends in a decision or a question for the data owner.

| Time | Activity | Check |
|---:|---|---|
| 0–8 | Load; audit types, missingness, uniqueness, duplicates. | Students distinguish an audit finding from a clinical conclusion. |
| 8–16 | Remove duplicates; process dates; derive length of stay. | Students explain why length of stay does not exist at admission. |
| 16–28 | Describe billing and groups. Negative billing values. | Students give four possible explanations for a negative bill and say they cannot choose between them from the file. |
| 28–38 | Feature availability and the leakage callout. | Students classify each column by when it becomes known. |
| 38–45 | Section 7: what would you tell a colleague? | Students state that the file supports practice but not benchmarking, and say why. |

**Do not rush section 4.** The instinct to delete the negative billing values is exactly the instinct this notebook exists to interrupt.

## Part 1A — classification (60 minutes)

| Time | Activity | Check |
|---:|---|---|
| 0–5 | Frame the benchmark and the boundary. | Students name the feature set and the target. |
| 5–13 | Load and inspect class balance. | Students state that 63% is the accuracy bar a trivial model already clears. |
| 13–22 | Split, stratification, and the CV diagram. | Students explain why the test set cannot choose the model. |
| 22–34 | Build the pipeline. | Students say why the imputer stays even with no missing values. |
| 34–46 | Baseline versus model, cross-validated. | Students explain why the dummy scores 0.63 accuracy but 0.39 macro F1. |
| 46–54 | Test metrics, confusion matrix, ROC. | Students identify which cell holds false negatives and why it matters more here. |
| 54–60 | Section 7: categorical preprocessing. | Students explain why numbering categories 1, 2, 3 would be wrong. |

**The moment that lands:** the dummy classifier's accuracy of 0.63 against its macro F1 of 0.39, on identical predictions, while never identifying a single malignant case. Let students sit with that before moving on.

## Part 1B — regression (60 minutes)

| Time | Activity | Check |
|---:|---|---|
| 0–5 | Frame the continuous target and the residual. | Students distinguish a label from a number. |
| 5–13 | Inspect the target distribution. | Students identify centre and spread, and note that errors will be in these units. |
| 13–22 | Split and pipeline policy. | Students explain why the test data are protected. |
| 22–32 | Build the pipeline; connect scaling to the Ridge penalty. | Students explain why scaling matters *because of* regularisation. |
| 32–42 | Baseline versus model; the negated-scorer convention. | Students convert a negative MAE to a positive error and interpret it in target units. |
| 42–50 | Test metrics and residual plots. | Students read prediction compression from the left plot, not from any metric. |
| 50–60 | Section 7: the leakage demonstration. | Students explain how selecting features before cross-validation produced R² = +0.32 from noise. |

**The moment that lands:** section 7. Ask students to predict the result before running it. Most expect roughly zero from both approaches.

## Common misconceptions

| Misconception | Correction |
|---|---|
| "Cross-validation replaces a test set." | CV supports development and model comparison. The test set supports one final internal estimate. |
| "The pipeline sees all rows first." | The split happens first. During CV each cloned pipeline fits preprocessing only on its training fold. |
| "No missing values means no imputer." | It is a robustness pattern that costs nothing and prevents a break on the next dataset. |
| "Accuracy is the probability the model is right." | It is an observed fraction correct on a specified evaluation set. It is not a confidence score. |
| "AUC is accuracy." | AUC summarises ranking across thresholds. It says nothing about any particular threshold. |
| "High AUC means the probabilities are trustworthy." | A model can rank perfectly and be badly calibrated. Those are different properties. |
| "Negative R² is a negative correlation." | It means worse than the mean-reference predictor under squared error on that set. |
| "A benchmark result proves clinical usefulness." | The bundled datasets demonstrate workflow mechanics, not deployment evidence. |
| "Near-baseline performance means broken code." | A workflow can be correct while the available features carry little signal. Part 0 is a worked example. |
| "The largest coefficient is the most important cause." | Coefficients depend on coding, scaling, correlation, and model assumptions. Association is not causation. |
| "Encoding categories as 1, 2, 3 is fine." | It asserts an order and equal spacing that do not exist. Use one-hot encoding. |

## Contingencies

| Problem | Response |
|---|---|
| Runtime fails | Use an executed notebook and continue the interpretation discussion. |
| Metrics differ slightly | Focus on the workflow and the baseline comparison, not the third decimal place. |
| Students want a better score | Do **not** reuse the held-out test set to search for one. Use the request to discuss why that turns the final number into an optimistic estimate of itself. |
| Running long | Drop Part 1A section 6 (ROC) before dropping section 7. Never drop Part 1B section 7. |
| A student asks why the CSV is not modelled | Part 0 section 7 answers this directly. Have them read it aloud. |

## Result statement template

> On five-fold cross-validation of the training portion, the candidate pipeline achieved
> [metric and variation] compared with [baseline result]. On the one-time held-out test set,
> it achieved [test metrics]. These results estimate predictive performance for this dataset
> under a random split. They do not establish performance for another hospital, a future time
> period, a patient subgroup, or a clinical or administrative deployment.

## Assessment note

These three notebooks are foundation material for Technical Lab 2. The graded rubric in `assessment/03_technical_lab_2/` covers tokenization, attention/decoding, RAG, and structured knowledge — all in `../part-2-rag/`. Parts 0 to 1B build the evaluation literacy that the rubric's "states limitations and communicates clearly" criterion depends on, but they do not themselves produce rubric evidence.
