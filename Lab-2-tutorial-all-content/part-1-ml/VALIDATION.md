# Validation Record — Parts 0 to 1B

## This revision

**Date:** 2026-09-20
**Execution environment:** Linux; Python 3.13.5; CPU. Package versions at validation time:

```
numpy==2.2.6   pandas==2.2.3   matplotlib==3.10.0   seaborn==0.13.2   scikit-learn==1.6.1
```

> **The course target remains Python 3.10+.** Run the documented clean environment from `requirements.txt` before delivery and record the resulting `pip freeze` here.

## What was verified

| Check | Result |
|---|---|
| Structural tests | Passed: `27 passed` via `python -m pytest -q` |
| Clean-kernel execution | Passed: all three notebooks ran top to bottom with no cell errors |
| Markdown-before-code contract | Passed: every code cell is preceded by a Markdown cell |
| Output-explanation contract | Passed: each notebook has at least three "What do we see?" cells |
| One statement per line | Passed: no semicolon-chained statements remain in any code cell |
| Portable data lookup | Passed: the CSV is located by searching upward, so no absolute home path is required |
| Source data shape | Passed: 55,500 rows × 15 columns before cleaning |
| Duplicate removal | Passed: 534 exact duplicates removed, 54,966 rows retained |
| Non-positive billing | Passed: 106 records with `Billing Amount` ≤ 0 retained and reported, not deleted |
| Class balance (Part 1A) | Passed: 62.7% benign / 37.3% malignant |
| Baseline contrast (Part 1A) | Passed: dummy reaches 0.626 accuracy but 0.385 macro F1 on identical predictions |
| Candidate performance (Part 1A) | Passed: logistic regression 0.976 CV macro F1, 0.981 on the held-out test set |
| Categorical preprocessing (Part 1A §7) | Passed: 6 input columns expand to 25 encoded columns with traceable feature names |
| Target spread (Part 1B) | Passed: mean 152.1, SD 77.1, range 25–346 |
| Baseline contrast (Part 1B) | Passed: dummy MAE 66.87 versus Ridge MAE 45.00 in CV |
| Negative R² is exercised | Passed: the dummy's test R² is −0.01, so the concept appears in real output |
| Prediction compression (Part 1B §6) | Passed: prediction SD 53.7 against actual SD 73.2, with residual/prediction correlation −0.080 |
| Leakage demonstration (Part 1B §7) | Passed: leaky mean R² **+0.318**, correct mean R² **−0.280**, on data with no signal |
| Handouts referenced by notebooks exist | Passed: four files in `../handouts/` |
| Student notebook generation | Passed: `python ../make_student_notebooks.py --check` reports up to date |

## Executed notebook outputs

| Notebook | Code cells | Figures | "What do we see?" cells |
|---|---:|---:|---:|
| `Part-0_Healthcare_Data_Processing_and_Analytics.ipynb` | 10 | 2 | 6 |
| `Part-1A_Classification_ML_Pipeline.ipynb` | 10 | 2 | 7 |
| `Part-1B_Regression_ML_Pipeline.ipynb` | 9 | 2 | 6 |

## Notes on interpretation claims

Every interpretive claim in the notebook Markdown was checked against the executed output rather than written from expectation. Two were corrected during validation:

1. The Part 1B residual discussion originally described a "downward tilt" in the residual-versus-predicted plot. The measured correlation is **−0.080** — essentially flat. The text now reports the flat residuals as a passed check and moves the finding to prediction compression, which is real and is backed by printed standard deviations.
2. The Tutorial 2 Part 0 tokenization discussion originally implied that presence in the training corpus determines token cost. The executed table shows `metformin` appearing twice and still costing 8 tokens. The text now attributes compression to **frequency**, which the sorted table demonstrates monotonically.

## Known limitations of these materials

- The supplied healthcare CSV is synthetic. Part 0 documents the specific signatures: no missing values, `Test Results` split into near-exact thirds, billing nearly identical across admission types, and a near-uniform length-of-stay distribution.
- Parts 1A and 1B use bundled benchmark datasets. Results demonstrate workflow mechanics and establish nothing about clinical usefulness.
- All results come from a single random split with `RANDOM_STATE = 42`. Fold-to-fold variation is reported but no confidence intervals are computed.
- No calibration, subgroup, fairness, or temporal-validation analysis is performed. These are named as gaps rather than covered.

## Before-class checklist

1. Create a clean environment from `requirements.txt`; record `pip freeze` above.
2. Run `python -m pytest -q` from this folder.
3. Execute all three notebooks from clean kernels and keep the executed copies.
4. Confirm the CSV is found by Part 0 and Part 1A section 7.
5. Print `../handouts/`.
6. Regenerate student notebooks if anything changed: `python ../make_student_notebooks.py`.
