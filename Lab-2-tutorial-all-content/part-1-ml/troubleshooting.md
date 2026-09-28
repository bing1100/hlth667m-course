# Troubleshooting — Parts 0 to 1B

| Problem | What to do |
|---|---|
| `FileNotFoundError` for the CSV | The notebooks search the working directory and up to four folders above it. Put `healthcare_dataset.csv` next to the notebook or in the lab root. In VS Code + Colab, upload it to the Colab runtime and rerun from the top. Parts 1A section 1–6 and all of 1B do not need it. |
| `ModuleNotFoundError` | Run `python -m pip install -r requirements.txt`, restart the kernel if prompted, then **Run All**. |
| Runtime restarted or disconnected | Reconnect, then run every cell from the beginning. Variables and fitted models are not retained. |
| Cells were run out of order | Restart the kernel and use **Run All**. Do not create missing variables by hand. |
| Logistic regression convergence warning | Confirm the pipeline includes `StandardScaler` and that `max_iter=2000` is set. Do not suppress the warning without checking it. |
| Undefined precision warning | Check whether a class was never predicted. The notebooks use `zero_division=0` so this is an interpretable warning rather than a crash. |
| Negative CV MAE or RMSE | This is scikit-learn's scoring convention: higher is better for every scorer, so losses are supplied negated. The notebooks multiply by `-1` before display. |
| Negative R² | The model is worse than predicting the mean of that evaluation set under squared error. It is **not** a negative correlation. Do not take the absolute value. |
| `ValueError: Found unknown categories` | Confirm `OneHotEncoder(handle_unknown="ignore")` is inside the fitted pipeline. |
| One-hot encoding produced a huge number of columns | Check the cardinality of your categorical columns. A column like `Doctor` with 40,000 distinct values produces 40,000 columns. Part 0 section 6 explains why such columns are excluded. |
| Plot is missing | Rerun the plot cell and confirm it ends with `plt.show()`. |
| Training is slow | The five-fold CV cells fit each pipeline five times. Wait for completion; skip optional extensions before reducing core work. |
| Results differ slightly from the guide | Confirm the package versions and `RANDOM_STATE`. Focus the discussion on the baseline comparison rather than the third decimal place. |
| Your model beats the baseline by a suspiciously large margin | Work through `../handouts/leakage_checklist.md` before reporting it. Part 1B section 7 shows leakage producing R² = +0.32 from pure noise. |
| An AI or code suggestion looks plausible but unclear | Pause, inspect the code and output, check the official documentation, and ask the instructor. Do not submit work you cannot explain. |

## First checks, in order

1. Is the correct kernel selected?
2. Did you run from the first code cell?
3. Does the displayed dataset shape match the expected source shape before duplicate removal?
4. Was the held-out test set used only at final evaluation?
5. Is every step that learns from data inside the `Pipeline`?
