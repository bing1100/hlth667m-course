# 7. Notebook Markdown Style Guide

## Purpose

The notebooks must be understandable without a lecturer translating each code cell. Follow the progressive structure of the referenced CTGAN tutorial: introduce one operation, run it, explain the output, and then move to the next operation. The new notebooks should provide more explanation than the source notebook because model evaluation introduces unfamiliar terms and several easy-to-miss validity rules.

Apply the course plain-language standard in `/home/bhux/research/proposals/hlth667m-course/current_course_materials/plain_language_standard.md`: use short sentences, state the point first, define technical terms on first use, and tell students exactly what to inspect.

## Required Markdown pattern around code

Not every explanation needs five visible subheadings. Keep simple cells brief. Across the Markdown immediately before and after each code cell, cover the following:

### Before a code cell

1. **Purpose:** one sentence stating the next action.
2. **New term:** define any new technical term in plain language.
3. **Why:** state why this step is needed in this workflow.
4. **Look for:** tell students what output to inspect.
5. **Documentation:** link the official API or guide when first introduced.

### After an important output

Add a Markdown cell titled **“What do we see?”** that:

- reports the expected pattern or approximate value;
- explains what it means for the next step;
- says what the output cannot establish;
- avoids claiming that the model “knows,” “understands,” or “decides.”

Small import or variable-definition cells do not need post-output commentary if they display nothing meaningful.

## Reusable cell templates

### Template A — Introduce a simple operation

```markdown
## 1.1 Preview the data

The next cell displays the first five rows. A **row** is one recorded admission, and a
**column** is one recorded variable.

Look for the mix of numeric, categorical, and date columns. This preview checks that the
file loaded correctly; it does not evaluate data quality.
```

### Template B — Introduce a scikit-learn object

```markdown
## 4.2 Combine the transformations

`ColumnTransformer` applies different preprocessing steps to different columns. We will
scale numeric columns and one-hot encode categorical columns.

Keeping these operations in the model pipeline prevents the transformations from learning
from validation or test rows. This helps prevent **preprocessing leakage**.

In the printed pipeline, look for a numeric branch and a categorical branch.

Documentation: [ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
```

### Template C — Explain an output

```markdown
### What do we see?

The three target classes contain similar numbers of rows. A classifier that always predicts
the most common class will therefore be correct only about one third of the time.

This balance makes class comparisons easier, but it does not tell us whether the available
features can predict the target. We will test that with cross-validation.
```

### Template D — Runtime or caution note

```markdown
> **Runtime note:** Cross-validation fits each pipeline five times. This cell may take longer
> than the earlier inspection cells. Wait for the table to appear before continuing.
```

```markdown
> **Do not use the test set here.** The test set is reserved for one final evaluation after
> model comparison is complete.
```

### Template E — Learner checkpoint

```markdown
### Checkpoint

Which field did we exclude because it occurs after admission? Write the field name and one
sentence explaining why including it could produce an invalid performance estimate.

<details>
<summary>Suggested answer</summary>

`Discharge Date` occurs after admission. It would give the admission-time model information
from the future.
</details>
```

### Template F — Try-it-yourself task

```markdown
## 8. Try It Yourself

Find the largest off-diagonal cell in the confusion matrix. State the true class, predicted
class, and number of records. Then name one question you would ask before deciding whether
this error is important.

Do not retrain the model or inspect the test set repeatedly for model selection.
```

## Required notebook navigation

Each notebook starts with:

- title and one-sentence purpose;
- last-updated date;
- estimated class and execution time;
- learning objectives;
- prerequisite vocabulary;
- table of contents with numbered sections;
- dataset and safety notice;
- instruction to run cells in order;
- link to `troubleshooting.md`.

Each notebook ends with:

- concise result summary;
- “what this result supports / does not support” table;
- completion checklist;
- links to glossary, exercise sheet, slides, troubleshooting, and official documentation;
- optional “continue learning” list clearly separated from required work.

## Code-cell size and explanation rules

- Prefer one conceptual operation per cell.
- Keep most code cells below roughly 20 lines; plotting cells can be longer when necessary.
- Do not hide core logic in large helper functions merely to shorten the notebook.
- Use descriptive variable names such as `classification_pipeline`, not `model1`.
- Put comments beside non-obvious code, but teach concepts in Markdown rather than long code comments.
- Display compact outputs. Avoid printing all rows, all one-hot feature names, or large arrays.
- Use a consistent class order and color palette across tables and plots.
- Never use a code cell without a preceding explanatory Markdown cell.

## Language and formatting rules

- Use “fit” for estimating preprocessing/model parameters and “predict” for producing outputs.
- Use “training portion” or “development data” consistently; define either term once.
- Use “held-out test set,” not merely “test data,” when emphasizing its final role.
- Say “recorded target” rather than implying ground truth is necessarily error-free.
- Use callout blockquotes for leakage, runtime, and safety warnings.
- Use equations only after a plain-language explanation.
- Do not copy large passages from scikit-learn. Paraphrase, link the source, and reproduce only short definitions or formulas needed for instruction.

## Accessibility rules

- Every plot must have a descriptive title and axis labels with units where applicable.
- Do not communicate model identity or quality through color alone; use labels, markers, or line styles.
- Use colorblind-friendly palettes.
- Include a one- or two-sentence text interpretation after each plot.
- Keep table column labels explicit, including “CV mean,” “CV standard deviation,” and metric direction (“higher is better” or “lower is better”).
