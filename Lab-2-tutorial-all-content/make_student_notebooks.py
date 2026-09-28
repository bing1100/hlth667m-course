#!/usr/bin/env python3
"""Generate student versions of the Lab 2 notebooks.

The syllabus promises "partially completed code so that students can focus on
changing model parameters, observing behaviour, interpreting results and
explaining technical choices". This script produces that version from the
complete notebooks, so the two can never drift apart.

For each notebook it replaces a small number of chosen code cells with a TODO
stub, clears all stored outputs, and writes the result into ``student/``.
The complete notebooks remain the instructor answer key.

Usage:
    python make_student_notebooks.py           # write student/
    python make_student_notebooks.py --check   # verify student/ is up to date
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "student"

BANNER = """> ### Your turn
>
> The next code cell is incomplete. Replace every `# TODO` with working code.
> The surrounding explanation tells you what the cell must do and what to look for.
> If you get stuck, the reasoning you need is in the Markdown directly above."""

# notebook -> list of (unique substring identifying the cell, replacement source)
SPECS: dict[str, list[tuple[str, str]]] = {
    "part-1-ml/Part-0_Healthcare_Data_Processing_and_Analytics.ipynb": [
        (
            'feature_policy = pd.DataFrame(',
            '''# TODO: classify each group of columns by when its value becomes known.
# Use exactly one of these labels in the second position:
#   "Potentially available" | "Identifier" | "Operational"
#   "Timing ambiguous"      | "Future information" | "Outcome" | "Recorded outcome"
# Prediction time for this exercise is ADMISSION.

feature_policy = pd.DataFrame(
    [
        ("Age, Gender, Blood Type, Medical Condition, Insurance Provider, Admission Type",
         "TODO", "TODO: why?"),
        ("Name", "TODO", "TODO: why?"),
        ("Doctor, Hospital, Room Number", "TODO", "TODO: why?"),
        ("Medication", "TODO", "TODO: why?"),
        ("Discharge Date, Length of Stay", "TODO", "TODO: why?"),
        ("Billing Amount", "TODO", "TODO: why?"),
        ("Test Results", "TODO", "TODO: why?"),
    ],
    columns=["Fields", "Status at admission", "Reason"],
)
display(feature_policy)''',
        ),
    ],
    "part-1-ml/Part-1A_Classification_ML_Pipeline.ipynb": [
        (
            "X_train, X_test, y_train, y_test = train_test_split(",
            '''# TODO: split the data.
#   - hold out 20% for the test set
#   - keep the class proportions the same in both parts
#   - use RANDOM_STATE so the split is reproducible
X_train, X_test, y_train, y_test = train_test_split(
    X, y,  # TODO: add the three arguments described above
)

split_summary = pd.DataFrame({
    "training %": (y_train.value_counts(normalize=True).sort_index() * 100).round(1),
    "test %": (y_test.value_counts(normalize=True).sort_index() * 100).round(1),
})
split_summary.index = [target_names[i] for i in split_summary.index]

print(f"Training rows: {len(X_train)} | test rows: {len(X_test)}")
display(split_summary)''',
        ),
        (
            'preprocessing = Pipeline([\n    ("imputer", SimpleImputer(strategy="median")),',
            '''# TODO: build the preprocessing pipeline.
#   step 1: impute missing values using the median
#   step 2: standardise the columns
preprocessing = Pipeline([
    # TODO
])

# TODO: wrap the preprocessing and DummyClassifier(strategy="prior") into one pipeline
dummy_pipeline = Pipeline([
    # TODO
])

# TODO: wrap the preprocessing and LogisticRegression into one pipeline.
# Use max_iter=2000 and random_state=RANDOM_STATE.
classification_pipeline = Pipeline([
    # TODO
])

print(f"Missing values in this dataset: {int(X.isna().sum().sum())}")
print(classification_pipeline)''',
        ),
        (
            'numeric_branch = Pipeline([',
            '''if CSV_PATH is not None:
    # TODO: build the numeric branch - impute with the median, then standardise.
    numeric_branch = Pipeline([
        # TODO
    ])

    # TODO: build the categorical branch - impute with the most frequent value,
    # then one-hot encode. Set handle_unknown="ignore" so an unseen category
    # does not crash the pipeline.
    categorical_branch = Pipeline([
        # TODO
    ])

    # TODO: combine the two branches with ColumnTransformer, applying each one
    # to the right list of columns.
    column_preprocessor = ColumnTransformer([
        # TODO
    ])

    encoded = column_preprocessor.fit_transform(health_X)
    feature_names = column_preprocessor.get_feature_names_out()

    print(f"Columns supplied to the transformer : {health_X.shape[1]}")
    print(f"Columns produced for the model      : {encoded.shape[1]}\\n")

    expansion = pd.DataFrame([
        {"source column": column,
         "categories": health_X[column].nunique(),
         "columns produced": sum(name.startswith(f"categorical__{column}_") for name in feature_names)}
        for column in categorical_features
    ])
    display(expansion)
    print("First 12 generated feature names:")
    for name in feature_names[:12]:
        print(" ", name)''',
        ),
    ],
    "part-1-ml/Part-1B_Regression_ML_Pipeline.ipynb": [
        (
            'preprocessing = Pipeline([\n    ("imputer", SimpleImputer(strategy="median")),',
            '''# TODO: build the preprocessing pipeline (impute with median, then standardise).
preprocessing = Pipeline([
    # TODO
])

# TODO: baseline pipeline using DummyRegressor(strategy="mean")
dummy_pipeline = Pipeline([
    # TODO
])

# TODO: candidate pipeline using Ridge(alpha=1.0)
regression_pipeline = Pipeline([
    # TODO
])

print(regression_pipeline)''',
        ),
        (
            "honest_pipeline = Pipeline([",
            '''# TODO: build the CORRECT version. Put SelectKBest INSIDE the pipeline so it is
# refitted within each training fold, then cross-validate the pipeline itself.
honest_pipeline = Pipeline([
    # TODO
])
honest_scores = cross_val_score(honest_pipeline, X_noise, y_noise, cv=noise_cv, scoring="r2")

display(pd.DataFrame({
    "approach": ["WRONG: selection before cross-validation", "CORRECT: selection inside the pipeline"],
    "mean R2": [leaky_scores.mean().round(3), honest_scores.mean().round(3)],
    "fold R2 values": [np.round(leaky_scores, 3), np.round(honest_scores, 3)],
}))
print("\\nThere is no signal in this data. The true R2 is 0.")''',
        ),
    ],
    "part-2-rag/Part-0_Tokenization_for_Health_Text.ipynb": [
        (
            "merges, final_vocabulary = learn_bpe_merges(TRAINING_CORPUS, num_merges=60)",
            '''# TODO: add three sentences of your own fictional clinic text to TRAINING_CORPUS
# below, then choose a number of merges. Try 60 first, then try 20 and 200 and
# compare which words become single tokens.
NUM_MERGES = 60  # TODO: experiment with this

merges, final_vocabulary = learn_bpe_merges(TRAINING_CORPUS, num_merges=NUM_MERGES)

merge_table = pd.DataFrame([
    {"step": m["step"], "merged pair": f'{m["pair"][0]!r} + {m["pair"][1]!r}',
     "new symbol": m["merged"], "times seen": m["count"]}
    for m in merges
])
print(f"learned {len(merges)} merges")
display(merge_table.head(18))''',
        ),
        (
            'probe_words = ["the", "portal", "result"',
            '''corpus_words = [w for s in TRAINING_CORPUS for w in re.findall(r"[a-z0-9]+", s.lower())]
corpus_counts = Counter(corpus_words)

# TODO: add at least three clinical terms of your own to this list.
# Predict how many tokens each will need BEFORE you run the cell, then check.
probe_words = ["the", "portal", "result", "appointment", "laboratory",
               "metformin", "hba1c", "myocardial", "tachycardia"]

rows = []
for word in probe_words:
    pieces = encode_with_bpe(word, merges)
    rows.append({"word": word,
                 "times in training corpus": corpus_counts.get(word.lower(), 0),
                 "tokens": len(pieces),
                 "pieces": " | ".join(pieces)})

probe_table = pd.DataFrame(rows).sort_values("times in training corpus", ascending=False)
display(probe_table.reset_index(drop=True))''',
        ),
    ],
    "part-2-rag/Part-1_Word2Vec_and_tSNE.ipynb": [
        (
            "WORD2VEC_CONFIG = dict(",
            '''# TODO: complete the training settings.
#   vector_size : coordinates per word vector (try 50)
#   window      : how many nearby tokens count as context (try 3)
#   min_count   : drop tokens seen fewer than this many times (use 1 here)
#   sg          : 1 for skip-gram, 0 for CBOW
#   workers, seed : keep at 1 and RANDOM_STATE so results are reproducible
WORD2VEC_CONFIG = dict(vector_size=50, window=3, min_count=1, workers=1, sg=1,
                       seed=RANDOM_STATE, epochs=100, negative=10, sample=1e-3)

baseline_model = Word2Vec(sentences=baseline_sentences, **WORD2VEC_CONFIG)
model = Word2Vec(sentences=tokenized_sentences, **WORD2VEC_CONFIG)
assert baseline_model.wv.vectors.shape == (44,50) and model.wv.vectors.shape == (83,50)
print('baseline vector matrix:', baseline_model.wv.vectors.shape)
print('varied vector matrix:  ', model.wv.vectors.shape)
print('clinic vector, first 8 coordinates:', np.round(model.wv['clinic'][:8], 3))''',
        ),
    ],
    "part-2-rag/Part-2_Attention_and_Self_Attention.ipynb": [
        (
            "def scaled_dot_product_attention_from_scratch(query, key, value, allowed_mask=None):",
            '''def scaled_dot_product_attention_from_scratch(query, key, value, allowed_mask=None):
    if query.shape[-1] != key.shape[-1]: raise ValueError('Query/key dimensions must match.')
    if key.shape[-2] != value.shape[-2]: raise ValueError('Key/value sequence lengths must match.')

    # TODO steps 1-2 (match, scale): compare every query with every key, then divide by sqrt(d_k).
    scores = None  # TODO

    if allowed_mask is not None:
        if torch.any(~allowed_mask.any(dim=-1)): raise ValueError('Every query row needs an allowed key.')
        # TODO step 3 (mask): set disallowed positions to -inf BEFORE the softmax.
        scores = None  # TODO

    # TODO step 4 (normalize): row-wise softmax so each row sums to one.
    weights = None  # TODO

    # TODO step 5 (combine): weighted sum of the value rows.
    return None, weights, scores  # TODO: replace the first None


query = torch.tensor([[1., 0.], [0., 1.]], dtype=DTYPE)
key   = torch.tensor([[1., 0.], [0., 1.]], dtype=DTYPE)
value = torch.tensor([[10., 0.], [0., 20.]], dtype=DTYPE)
labels = ['clinic', 'portal']
output, weights, scores = scaled_dot_product_attention_from_scratch(query, key, value)

official = torch.nn.functional.scaled_dot_product_attention(query[None, None], key[None, None], value[None, None])[0, 0]
torch.testing.assert_close(output, official, rtol=1e-6, atol=1e-7)
print("Assertion passed - your implementation matches PyTorch.")
print('Weight row sums:', weights.sum(-1).tolist())''',
        ),
        (
            "sliding = causal & (idx[:, None] - idx[None, :] < window)",
            '''sentence = ['the', 'patient', 'reports', 'cough', 'today', '[PAD]']
n = len(sentence)
full = torch.ones(n, n, dtype=torch.bool)
# TODO: a token may read itself and earlier tokens. Build this mask from `full`.
causal = None  # TODO
not_pad = torch.tensor([t != '[PAD]' for t in sentence])
padding = full & not_pad[None, :]
prefix_len = 3                                         # "the patient reports" is the prompt
prefix = causal.clone(); prefix[:prefix_len, :prefix_len] = True
window = 3
idx = torch.arange(n)
# TODO: causal AND at most `window` tokens back. idx[:, None] - idx[None, :] is the distance from query to key.
sliding = None  # TODO

patterns = [('Bidirectional (full)', full), ('Causal', causal), ('Padding', padding),
            ('Causal + padding', causal & padding), (f'Prefix (first {prefix_len} tokens)', prefix & padding),
            (f'Sliding window (w = {window})', sliding & padding)]
fig, axes = plt.subplots(2, 3, figsize=(12, 7.2))
for ax, (title, mask) in zip(axes.ravel(), patterns):
    sns.heatmap(mask.numpy().astype(int), cmap=MASK_CMAP, vmin=0, vmax=1, cbar=False, linewidths=1.2, linecolor='white',
                xticklabels=sentence, yticklabels=sentence, ax=ax, square=True)
    ax.set_title(f'{title}\n{int(mask.sum())} of {n*n} pairs allowed', fontsize=10)
    ax.tick_params(axis='x', rotation=45, labelsize=8); ax.tick_params(axis='y', rotation=0, labelsize=8)
fig.suptitle('Green = the row token may read the column token', y=1.0)
plt.tight_layout(); plt.show()

display(pd.DataFrame({title: [', '.join(t for t, ok in zip(sentence, mask[3]) if ok)] for title, mask in patterns},
                     index=['"cough" may read']).T)''',
        ),
    ],
    "part-2-rag/Part-3_RAG_With_and_Without_Retrieval.ipynb": [
        (
            "window_chunks = []",
            '''# TODO: try at least two different window sizes and record what changes.
# Start with 40 words and 10 overlap, then try 25/5 and 80/20.
WORDS_PER_CHUNK = 40   # TODO: experiment
OVERLAP_WORDS = 10     # TODO: experiment

window_chunks = []
for record in manifest:
    window_chunks.extend(chunk_markdown_fixed_window(CORPUS_DIR/record['filename'], record,
                                                     words_per_chunk=WORDS_PER_CHUNK,
                                                     overlap_words=OVERLAP_WORDS))

comparison = pd.DataFrame([
    {'strategy': 'heading (h2-v1)', 'chunks': len(chunks),
     'min words': min(len(c['text'].split()) for c in chunks),
     'max words': max(len(c['text'].split()) for c in chunks),
     'crosses a topic boundary': 'no'},
    {'strategy': f'fixed window ({WORDS_PER_CHUNK}w/{OVERLAP_WORDS} overlap)', 'chunks': len(window_chunks),
     'min words': min(len(c['text'].split()) for c in window_chunks),
     'max words': max(len(c['text'].split()) for c in window_chunks),
     'crosses a topic boundary': 'yes'},
])
display(comparison)

print('A fixed window that straddles two sections:\\n')
straddling = next(c for c in window_chunks if c['text'].count('##') >= 1 and not c['text'].startswith('##'))
print(f"  {straddling['chunk_id']}: {straddling['text'][:170]}...")''',
        ),
        (
            "PATIENT_QUESTION = 'Who tells me about my bloodwork if something is wrong?'",
            '''terminology = load_terminology(TERMINOLOGY_DIR/'health_terminology.json')
print(f"code system: {terminology['code_system']} ({terminology['code_system_version']}) | concepts: {len(terminology['concepts'])}")
display(pd.DataFrame([{'code': c['code'], 'preferred term': c['preferred_term'],
                       'synonyms': ', '.join(c['synonyms'][:4]) + ('...' if len(c['synonyms'])>4 else '')}
                      for c in terminology['concepts']]))

# TODO: write a question the way a patient would actually ask it, using words that
# do NOT appear in the fictional corpus. Then check which concept it matched.
# If nothing matched, add a synonym to data/terminology/health_terminology.json and rerun.
PATIENT_QUESTION = 'Who tells me about my bloodwork if something is wrong?'  # TODO: change this

expansion = expand_query_with_terminology(PATIENT_QUESTION, terminology)
print(f"\\noriginal : {expansion['original_query']}")
print(f"matched  : {[(c['code'], c['matched_surface_forms']) for c in expansion['matched_concepts']]}")
print(f"added    : {expansion['added_terms']}")
print(f"expanded : {expansion['expanded_query']}")''',
        ),
    ],
}


def make_student_copy(source: Path, replacements: list[tuple[str, str]]) -> dict:
    notebook = json.loads(source.read_text(encoding="utf-8"))
    remaining = list(replacements)
    cells = []

    for cell in notebook["cells"]:
        source_text = "".join(cell["source"])
        matched = next((r for r in remaining if r[0] in source_text), None)

        if matched is not None and cell["cell_type"] == "code":
            remaining.remove(matched)
            cells.append({"cell_type": "markdown", "metadata": {},
                          "source": BANNER.splitlines(keepends=True)})
            cell = dict(cell)
            cell["source"] = matched[1].splitlines(keepends=True)

        if cell["cell_type"] == "code":
            cell = dict(cell, outputs=[], execution_count=None)
        cells.append(cell)

    if remaining:
        raise SystemExit(f"{source.name}: could not locate {len(remaining)} target cell(s): "
                         + ", ".join(repr(r[0][:50]) for r in remaining))

    notebook["cells"] = cells
    return notebook


SUPPORT_FILES = ["part-2-rag/tutorial_utils.py", "part-2-rag/rag_chat_app.py"]
SUPPORT_LINKS = ["part-2-rag/data"]


def copy_support(check: bool) -> list[str]:
    """Student copies need the helper module and the fictional data beside them."""
    import shutil
    stale = []
    for relative in SUPPORT_FILES:
        source, target = ROOT/relative, OUT_DIR/relative
        if check:
            if not target.exists() or target.read_bytes() != source.read_bytes():
                stale.append(relative)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            print(f"copied {target.relative_to(ROOT)}")
    for relative in SUPPORT_LINKS:
        source, target = ROOT/relative, OUT_DIR/relative
        if check:
            if not target.exists():
                stale.append(relative)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink() or target.exists():
                if target.is_symlink() or target.is_file():
                    target.unlink()
                else:
                    shutil.rmtree(target)
            try:
                target.symlink_to(Path("../..")/relative, target_is_directory=True)
                print(f"linked {target.relative_to(ROOT)} -> ../../{relative}")
            except OSError:
                shutil.copytree(source, target)
                print(f"copied {target.relative_to(ROOT)} (symlink unsupported)")
    return stale


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="verify student/ matches the current notebooks")
    args = parser.parse_args()

    stale = copy_support(args.check)
    for relative, replacements in SPECS.items():
        source = ROOT / relative
        if not source.exists():
            raise SystemExit(f"missing source notebook: {relative}")
        student = make_student_copy(source, replacements)
        target = OUT_DIR / relative
        rendered = json.dumps(student, indent=1) + "\n"

        if args.check:
            if not target.exists() or target.read_text(encoding="utf-8") != rendered:
                stale.append(relative)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(rendered, encoding="utf-8")
            print(f"wrote {target.relative_to(ROOT)} ({len(replacements)} TODO cell(s))")

    if args.check:
        if stale:
            print("student notebooks are out of date:")
            for name in stale:
                print("  -", name)
            print("run: python make_student_notebooks.py")
            return 1
        print(f"student/ is up to date ({len(SPECS)} notebooks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
