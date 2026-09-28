# Exercises and Answer Key

## Student exercises

### 1. Identify the task
Classify each target as classification or regression: discharge destination, length of stay in days, test-result category, total cost, readmission yes/no.

### 2. Feature availability
For the supplied healthcare CSV, place these fields into **available at admission**, **uncertain timing**, or **future information**: `Age`, `Medication`, `Discharge Date`, `Billing Amount`, `Admission Type`.

### 3. Manual classification metrics
For one class, suppose TP = 18, FP = 6, FN = 12, and TN = 64. Calculate accuracy, precision, recall, and F1. Which metric answers “of records predicted positive, how many were actually positive?”

### 4. Metric choice
A workflow must identify a harmful result category, and missing an affected record is more costly than an unnecessary follow-up. Which metric deserves attention, and why?

### 5. Manual regression errors
Actual continuous targets are `[100, 200, 300, 400]`; predictions are `[120, 170, 330, 350]`. Calculate residuals using actual minus predicted, MAE, and RMSE.

### 6. Read CV results
A model has mean CV macro F1 of 0.95 (SD 0.01); the dummy baseline has 0.39 (SD 0.00). What should you conclude before looking at the held-out test set?

### 7. Interpret a weak result
Write two sentences that correctly report a candidate model that does not materially beat its dummy baseline.

### 8. Audit a claim
Revise: “The model is 80% accurate, so it is ready for clinical use.”


### 9. Spot the leak
A colleague writes: "I standardised all the features, then split into train and test, then cross-validated the model on the training set." Name the leak and state the one-line fix.

### 10. Categorical encoding
You encode `Admission Type` as `Elective=1, Emergency=2, Urgent=3` and feed it to a linear model. State two specific false claims this makes, and name the correct encoding.

### 11. Read a suspicious result
A student reports R² = 0.94 predicting length of stay at admission, using every column in the CSV. Give the two most likely explanations and say which you would check first.

### 12. Choose the metric
A workflow flags patients for a follow-up telephone call. A missed patient is far worse than an unnecessary call, and staff capacity is not the binding constraint. Which metric should lead, and what else must be stated before choosing a threshold?

## Instructor answer key

1. Classification: discharge destination, test-result category, readmission yes/no. Regression: length of stay, total cost.
2. Available: `Age`, `Admission Type`. Uncertain: `Medication`. Future: `Discharge Date`, final `Billing Amount` under admission-time framing.
3. Accuracy = `(18 + 64) / 100 = 0.82`; precision = `18 / 24 = 0.75`; recall = `18 / 30 = 0.60`; F1 = `2 × .75 × .60 / (.75 + .60) ≈ 0.67`. Precision answers the stated question.
4. Recall/sensitivity deserves attention because it measures the share of true affected records identified. The final choice also requires a stated workflow and false-positive cost.
5. Residuals are `[-20, 30, -30, 50]`; MAE = 32.5; RMSE ≈ 35.36.
6. The candidate appears to improve on the baseline consistently across folds. Select it using training CV, then evaluate it once on the untouched test set.
7. Example: “On the training CV folds, the candidate did not materially improve on the dummy baseline. With these approved features, the recorded target may contain little predictable signal; this does not show the code failed.” The supplied healthcare CSV in Part 0 illustrates this possibility, which is why it is not used for the core modeling benchmarks.
8. Example: “The model achieved 80% accuracy on a specified evaluation set. We would still need error-by-class analysis, validation in the intended population/time, calibration or threshold evidence where relevant, workflow evaluation, and governance evidence before considering use.”

9. **Preprocessing leakage.** The scaler learned each column's mean and standard deviation from the test rows as well as the training rows. Fix: put `StandardScaler` inside the `Pipeline` and cross-validate the pipeline, so it is refitted within each training fold.
10. It claims (a) that the categories have an order, with Urgent greater than Emergency greater than Elective, and (b) that the gaps are equal, so the distance from Elective to Emergency equals the distance from Emergency to Urgent. Neither is true. Use `OneHotEncoder`, which creates one 0/1 column per category.
11. Most likely: (a) **future-information leakage** — `Discharge Date` is in the feature set, and length of stay is computed from it, so the target is derivable from an input; (b) preprocessing or selection performed before the split. Check (a) first: it is both more likely here and immediately verifiable from the feature list. Part 0 section 6 classifies `Discharge Date` as future information for exactly this reason.
12. **Recall (sensitivity)** should lead, because it measures the share of true cases found and a false negative is the costly error. Before choosing a threshold you must also state the false-positive cost, the capacity to act on flagged patients, the population the threshold was tuned on, and the validation set used — and that set must not be the held-out test set.
