# HLTH 667M Lab 2 Tutorial Plan — Part 1: Scikit-learn ML Pipelines

> **Status: delivered.** These materials were built and now live at
> `/home/bhux/research/proposals/hlth-667-668/Lab-2-tutorial/`, not at the path
> planned below. The delivery also departs from this plan in several deliberate
> ways; see **Delivered structure** at the end of this file.

## Purpose and scope

This planning package specifies the first two approximately one-hour tutorial parts for Lab 2. The materials teach reproducible supervised machine-learning workflows with **scikit-learn**, using the supplied synthetic/de-identified healthcare dataset. The tutorials are intentionally separated into two independently runnable notebooks:

1. **Classification pipeline:** predict `Test Results` (`Abnormal`, `Inconclusive`, or `Normal`).
2. **Regression pipeline:** estimate `Billing Amount`.

Each notebook teaches the complete workflow: data audit, feature eligibility, held-out test split, cross-validation, preprocessing in a `Pipeline`, baseline comparison, model fitting, metric computation, graphics, interpretation, and limitations. The companion slide helper supports a live walkthrough rather than duplicating notebook code.

> **Teaching boundary:** These are educational demonstrations using a course dataset. They are not clinical decision-support systems, validated billing tools, causal analyses, or deployment-ready models.

## Source and destination locations

| Purpose | Absolute path |
|---|---|
| Source dataset | `/home/bhux/research/proposals/hlth-667-668/Lab-2-tutorial/healthcare_dataset.csv` |
| **Delivered** tutorial materials | `/home/bhux/research/proposals/hlth-667-668/Lab-2-tutorial/part-1-ml/` |
| This planning package | `/home/bhux/research/proposals/hlth667m-course/lab-2-tutorial-plan/` |

The final notebooks should not modify the source CSV. They should include a documented path variable and, if Colab use is supported, instructions for selecting/uploading the supplied file rather than silently downloading unrelated data.

## Planned final deliverables

```text
/home/bhux/research/proposals/hlth-667-668/Lab-2-tutorial/part-1-ml/
├── README.md
├── requirements.txt
├── Part-0_Healthcare_Data_Processing_and_Analytics.ipynb
├── Part-1A_Classification_ML_Pipeline.ipynb
├── Part-1B_Regression_ML_Pipeline.ipynb
├── slides.md
├── glossary_and_quick_reference.md
├── exercises_and_answer_key.md
├── troubleshooting.md
├── instructor_guide.md
├── VALIDATION.md
└── tests/
```

## Planning documents

| Document | Use |
|---|---|
| `01_tutorial_architecture.md` | Shared instructional decisions, data rules, timing, file structure, and implementation standards. |
| `02_classification_notebook_blueprint.md` | Cell-by-cell specification for the classification notebook. |
| `03_regression_notebook_blueprint.md` | Cell-by-cell specification for the regression notebook. |
| `04_metrics_visualizations_and_interpretation.md` | Metric definitions, formulas, graphics, cautions, and documentation links. |
| `05_slide_deck_helper.md` | Presenter-ready slide sequence aligned to notebook checkpoints. |
| `06_scikit_learn_resources_and_validation.md` | Authoritative resources, dependency policy, execution checks, and acceptance criteria. |
| `07_notebook_markdown_style_guide.md` | Exact learner-facing Markdown pattern, section structure, transitions, output explanations, and examples. |
| `08_supplementary_materials_plan.md` | Glossary, exercises, worked metric examples, troubleshooting, extensions, and slide-only concepts. |
| `09_instructor_guide_plan.md` | Instructor run-of-show, checkpoints, likely outputs, misconceptions, contingencies, and answer-key plan. |

## Dataset facts established during planning

The source CSV contains 55,500 rows and 15 columns. It has no missing values in its current form, but the tutorials retain imputation inside the pipeline because real health data commonly have missingness and preprocessing must be learned within training folds. There are 534 exact duplicate rows; remove exact duplicates before the train/test split to avoid duplicated records inflating apparent generalization.

- `Test Results` is a nearly balanced three-class target: 18,627 `Abnormal`, 18,517 `Normal`, and 18,356 `Inconclusive` records before duplicate removal.
- `Billing Amount` is continuous (mean approximately 25,539; observed range approximately -2,008 to 52,764). Negative values should be inspected and discussed, not automatically discarded.
- `Name`, `Doctor`, and `Hospital` have very high cardinality and are excluded from this introductory pipeline.
- `Discharge Date` is excluded from admission-time prediction because it is future information. A derived length of stay is likewise not admissible for this prediction-time framing.

## Required instructional standard

Every executable notebook cell must have a Markdown cell immediately before it. That explanation must say:

1. **What this cell does.**
2. **Why this step belongs in a valid predictive workflow.**
3. **How to read the output, table, or graphic.**
4. **Relevant assumptions, limitations, or common pitfalls.**
5. **At least one official scikit-learn documentation link** when a scikit-learn object, model-selection method, or metric is introduced.

Use plain language first, then provide correct technical vocabulary. Students should be able to explain why a step was taken, not only run it.

## Gaps identified and corrected in this revision

The first plan established the technical workflow, but it did not fully specify how the notebooks would teach that workflow. Comparison with the course CTGAN notebook identified the need for:

- numbered sections and a visible beginning-to-end narrative;
- short definitions immediately before a new term or API is used;
- brief Markdown after important outputs to state what happened and what to notice;
- expected-output and runtime notes;
- setup checks, file-path guidance, troubleshooting, and recovery from a restarted runtime;
- progress checkpoints, short learner tasks, and a “what’s next?” section;
- separate reference, exercise, and instructor materials so the notebooks remain readable;
- coverage of important topics that do not fit the core hour, including grouped and temporal validation, probability calibration, precision–recall curves, subgroup checks, learning curves, uncertainty, and model persistence.

These requirements are now specified in Documents 07–09 and incorporated into the revised notebook blueprints.

## Delivered structure

The delivery departs from this plan in four deliberate ways.

1. **A third notebook, Part 0, was added.** It audits the supplied CSV and reaches a conclusion this plan did not anticipate: under an admission-time framing the CSV has very little signal for its recorded outcomes, and several distributions are artifacts of synthetic generation (`Test Results` split into near-exact thirds; billing nearly identical across admission types; a near-uniform length-of-stay distribution).

2. **Parts 1A and 1B therefore use bundled scikit-learn datasets**, not the CSV. This plan specified `Test Results` and `Billing Amount` as targets. Teaching a modelling workflow on data with no signal would have meant teaching students to read near-chance results as success. The CSV's *weakness* is now the lesson, in Part 0 section 7.

3. **The CSV returns for categorical preprocessing.** Part 1A section 7 builds the `ColumnTransformer` and `OneHotEncoder` specified here, using the real `Admission Type`, `Insurance Provider`, `Blood Type`, `Medical Condition`, and `Gender` columns — preprocessing only, with no performance claim.

4. **The leakage demonstration is worked, not described.** Part 1B section 7 selects 20 features from 2,000 columns of pure noise before cross-validating and reports R² = +0.32 on data with no signal, against −0.28 when the same selection sits inside the pipeline.

The supplementary materials specified in document 08 were delivered as four printable one-pagers in `../../hlth-667-668/Lab-2-tutorial/handouts/`: the pipeline map, data-splitting map, metric chooser, and leakage checklist.

Student versions with `# TODO` cells, required by the syllabus ("tutorials will provide partially completed code"), are generated into `../../hlth-667-668/Lab-2-tutorial/student/` by `make_student_notebooks.py`.
