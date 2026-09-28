# Part 1 — Machine-Learning Pipelines with scikit-learn

This half of Lab 2 covers the workflow that any supervised model sits inside: audit the data, decide what each column is, split, build a pipeline, compare against a baseline, evaluate once, and state the limits.

> **Teaching boundary:** these notebooks are not clinical decision-support systems, validated billing tools, causal analyses, or deployment instructions.

## Notebooks, in order

| File | Teaches | Data | Runtime |
|---|---|---|---:|
| `Part-0_Healthcare_Data_Processing_and_Analytics.ipynb` | Audit, clean, and describe the supplied healthcare CSV; decide what each column is and when it becomes known. **No model is fitted.** | Course CSV | 30 min |
| `Part-1A_Classification_ML_Pipeline.ipynb` | Predicting a category; class imbalance and metric choice; `ColumnTransformer` and one-hot encoding for categorical health columns. | `load_breast_cancer()` + course CSV | 60 min |
| `Part-1B_Regression_ML_Pipeline.ipynb` | Predicting a number; MAE/RMSE/R²; residual diagnostics; a worked demonstration of preprocessing leakage. | `load_diabetes()` | 60 min |

This folder is Tutorial 1 of Lab 2. Tutorial 2 in `../part-2-rag/` restarts its numbering at Part 0 (Parts 0 to 3), so the lab writes "Tutorial 1 Part 0" or "Tutorial 2 Part 0" wherever a reference could be read either way.

## Why two different datasets

Part 0 examines the supplied healthcare CSV and reaches a conclusion: the columns available at admission carry very little information about the recorded outcomes, and several distributions do not behave like the real processes they represent. Balanced thirds in `Test Results` and near-identical billing across admission types are the clearest signs.

That is a finding worth teaching, not a failure. But it makes the CSV a poor benchmark, so Parts 1A and 1B teach the mechanics on scikit-learn's bundled datasets, where a signal genuinely exists.

The CSV returns in Part 1A section 7, where its categorical columns demonstrate `ColumnTransformer` and one-hot encoding — preprocessing only, with no performance claim.

## Supporting material

| File | Purpose |
|---|---|
| `slides.md` | Presenter-ready companion deck. |
| `instructor_guide.md` | Run-of-show, checkpoints, misconceptions, contingencies. |
| `glossary_and_quick_reference.md` | Terms, metric direction, and scikit-learn object reference. |
| `exercises_and_answer_key.md` | Student practice plus instructor answers. |
| `troubleshooting.md` | Direct fixes for common problems. |
| `VALIDATION.md` | What was verified, when, and in which environment. |
| `../handouts/` | Four printable one-pagers: pipeline map, splitting map, metric chooser, leakage checklist. |
| `../student/part-1-ml/` | Student versions with `# TODO` cells. |

## Data

Part 0 and Part 1A section 7 read `healthcare_dataset.csv`. The notebooks search the working directory and each folder above it, so they work whether you run from this folder, from the lab root, or from the `student/` copy.

If you use VS Code with a Colab kernel, right-click `healthcare_dataset.csv` and choose **Upload to Colab**, then run from the top.

Parts 1A and 1B otherwise use the bundled loaders `load_breast_cancer()` and `load_diabetes()`, which need neither an upload nor a network download.

## Setup

Python 3.10+ where possible.

```bash
python -m pip install -r requirements.txt
python -m pytest -q          # structural checks on the notebooks
```

Open a notebook, choose a Python or Colab kernel, and run cells from top to bottom. If a runtime restarts, use **Run All** — a restarted kernel forgets every variable.

## Workflow rules these notebooks follow

- Exact duplicates are removed and reported **before** any split or summary.
- Every column is classified by **when its value becomes known**, relative to a stated prediction time.
- Every step that learns from data — imputer, scaler, encoder, selector — lives **inside** a `Pipeline`.
- Cross-validation compares candidates using the training portion only.
- The held-out test set is evaluated **once**, after selection is complete.
- Every candidate is compared against a dummy baseline before anything else is claimed.

## Expected runtime

Each notebook takes a few minutes of computation on a typical local or Colab CPU; the class discussion is designed to fill about an hour each. Cross-validation is the slowest step because it fits each pipeline five times.

## Support

Read `troubleshooting.md` before changing code. Use the official scikit-learn links embedded in the notebook Markdown for API details. Do not upload real patient data, credentials, or other restricted material to a hosted runtime.
