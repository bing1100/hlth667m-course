# 2. Word2Vec and t-SNE Notebook Blueprint

## Notebook identity

- **Filename:** `Part-1_Word2Vec_and_tSNE.ipynb`
- **Duration:** approximately 25 minutes
- **Question:** How can local word context produce useful numerical representations?
- **Data:** an explicit fictional corpus constructed in the notebook
- **Main objects:** `gensim.models.Word2Vec`, `sklearn.manifold.TSNE`
- **Primary outputs:** vocabulary table, nearest-neighbor examples, selected-word vector map

## Required notebook sequence

Every code cell must be preceded by explanatory Markdown. Important outputs need a following “What do we see?” cell.

| Section | Markdown focus | Code/output requirement |
|---|---|---|
| Title | Define embedding, Word2Vec, t-SNE, scope, runtime, and safety boundary. | No code. State that the corpus is fictional and tiny. |
| 0. Setup | Explain fixed seeds, package versions, and CPU runtime. | Import packages; set NumPy/Gensim/sklearn seeds; print versions. |
| 1. Build example corpus | Explain tokens, sentences, context window, and why a synthetic corpus is used. | Generate the exact deterministic template corpus in Document 11 (240 sentences across scheduling, laboratory, and governance contexts); lowercase/tokenize with the specified regex; display examples and token frequencies. |
| 2. Train Word2Vec | Define CBOW and skip-gram. Explain `vector_size`, `window`, `min_count`, `sg`, `workers`, `seed`, and `epochs`. | Fit a small skip-gram model with `workers=1`; print vocabulary size and vector matrix shape. Optionally fit CBOW with the same settings for comparison, clearly labelled as an extension. |
| 3. Inspect vectors | Explain vector lookup and cosine similarity. | Display `model.wv["clinic"]`; call `most_similar` for 3–5 seed words; compare a pair of cosine similarities. |
| 4. Map embeddings with t-SNE | Explain that t-SNE maps high-dimensional vectors to two dimensions and is sensitive to perplexity/random seed. | Select the fixed 18-word vocabulary in Document 11; fit `TSNE(n_components=2, perplexity=5, init="pca", random_state=42, learning_rate="auto", max_iter=1000)`; plot labelled points. |
| What do we see? | Explain local neighborhoods and the limits of the map. | No code. State that 2-D distances are visualization-dependent and not a proof of clinical or linguistic meaning. |
| 5. Try it yourself | Ask students to change `window` or include/exclude a word and rerun only the model/map cells. | Explain what must remain fixed for a fair comparison. |
| Conclusion | State what the notebook supports and does not support; link to attention notebook. | No code. |

## Corpus design requirements

Use the exact sentence templates and deterministic expansion in Document 11, not a downloaded corpus. The repeated contexts intentionally create visible local signal. Do not use real patient narratives or claim that the model learns general language semantics.

Recommended conceptual sentence families:

- `clinic`, `nurse`, `appointment`, `followup`, `portal`, `schedule`
- `screening`, `result`, `laboratory`, `review`, `referral`
- `privacy`, `consent`, `record`, `access`, `audit`

Keep tokenization simple and visible. A helper such as `sentence.lower().split()` is acceptable for this lesson; explain that production tokenization is more complex.

## Expected implementation details

```python
from gensim.models import Word2Vec
model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=50,
    window=3,
    min_count=1,
    workers=1,
    sg=1,
    seed=42,
    epochs=100,
)
```

Use `model.wv.key_to_index` and `model.wv.get_vector(word)`/`model.wv[word]` for vector access. Use `model.wv.most_similar(word)` for a demonstration, not as a validated semantic judgment.

For t-SNE, use perplexity 5 for 18 selected points and state that perplexity must be smaller than the number of samples. Do not run multiple seeds and select the most attractive plot.

## Required theory before code

The notebook and slides must show the two training objectives conceptually:

- **CBOW:** surrounding context words are inputs and the center word is the prediction target.
- **Skip-gram:** the center word is the input and nearby context words are prediction targets.

Introduce the distributional premise in bounded language: words appearing in similar local contexts can receive similar vectors. Define cosine similarity as the normalized dot product and explain why magnitude is removed. Do not derive negative sampling in the core 25 minutes; include it in a “what training omits from this walkthrough” callout.

## Pedagogical cautions

- Word2Vec is not a dictionary and does not assign a fixed human-defined meaning to a word.
- Similarity reflects the training corpus, preprocessing, hyperparameters, and random initialization.
- A small synthetic corpus is useful for mechanism inspection but cannot support claims about broad language or health terminology.
- t-SNE preserves some local relationships for visualization; global axes, distances, cluster sizes, and orientation are not directly interpretable.
- Embeddings can encode unwanted associations when trained on real data; the fictional corpus avoids exposing those issues but does not remove the general risk.
