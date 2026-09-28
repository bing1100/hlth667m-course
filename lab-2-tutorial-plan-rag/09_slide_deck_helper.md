# 9. Presenter-Ready Slide Deck Content — Representations, Attention, and RAG

## Purpose and timing

The final `slides.md` should contain these 24 slides and work alongside the three notebooks. Slides provide theory, formulas, diagrams, safeguards, and questions. Notebooks provide executable calculations and outputs. The 90-minute class interleaves slides and code according to the instructor guide. The per-slide minutes below are pacing maxima that include the linked discussion or short notebook handoff; they are not added to notebook durations. Their sum is therefore not a second class-length calculation.

Use the exact slide fields below. Every visual requires alt text. Keep formulas readable and define every symbol.

---

## Slide 0 — From text to grounded responses (3 min)

**Visible text:**
- Word vectors → attention → retrieval-augmented generation
- Main question: How does text become numerical input, and how can external evidence enter a generated response?

**Visual:** Three connected boxes: local context produces word vectors; Q/K/V produce weighted combinations; a question retrieves chunks that enter a generation prompt.

**Alt text:** A left-to-right sequence connects Word2Vec, attention, and RAG while showing that each stage transforms text or vectors.

**Say:** “We will inspect three different mechanisms. Word2Vec learns word-level vectors from co-occurrence. Attention combines value vectors using query-key similarity. RAG retrieves external text and places it into a generation request.”

**Ask:** “Which of these stages retrieves an external document?”

**Expected:** “RAG retrieval.”

**Notebook transition:** Open all three notebook titles; do not run code yet.

---

## Slide 1 — Boundaries for this tutorial (2 min)

**Visible text:**
- Fictional teaching text only
- No patient data or medical advice
- Computed similarity is not clinical meaning
- Generated wording is not validated evidence

**Visual:** Checklist with data, interpretation, and deployment boundaries.

**Alt text:** Four safeguards restrict the tutorial to fictional text and educational mechanism inspection.

**Say:** “The examples use fictional health-services language. We inspect operations, not clinical validity. Never send protected or confidential text to an external API in this tutorial.”

**Ask:** “Can a local vector database safely contain unrestricted patient text merely because it is local?”

**Expected:** “No. Local persistence is not an access-control or governance system.”

**Notebook transition:** Part 1 title and safety notice.

---

## Slide 2 — Tokens, context, and representations (3 min)

**Visible text:**
- Token: a text unit used by the model
- Context window: nearby tokens used during training
- Embedding: a dense numerical vector
- Distributional premise: similar local contexts can produce similar vectors

**Visual:** Sentence `clinic schedules followup through portal` with a window of three tokens around `followup`.

**Alt text:** A sentence highlights a center token and nearby context tokens used by Word2Vec.

**Say:** “Word2Vec does not begin with dictionary definitions. It creates training examples from neighboring tokens. Its result depends on the supplied corpus and preprocessing.”

**Ask:** “If the corpus changes, must the vectors stay the same?”

**Expected:** “No. Vectors depend on corpus context and settings.”

**Notebook transition:** Run Part 1 corpus-generation and token-frequency cells.

---

## Slide 3 — Two Word2Vec training directions (3 min)

**Visible text:**
- CBOW: context → center word
- Skip-gram: center word → nearby context words
- Both learn vector parameters from prediction tasks

**Visual:** Two diagrams using `clinic schedules followup`: arrows from `clinic` and `followup` to `schedules` for CBOW; arrows from `schedules` to neighboring words for skip-gram.

**Alt text:** CBOW predicts a center token from context, while skip-gram predicts context tokens from the center token.

**Say:** “The notebook uses skip-gram. The prediction task is a training device; after training, we keep the vectors for lookup and similarity.”

**Ask:** “Does skip-gram generate a full natural-language answer?”

**Expected:** “No. It predicts local context during training and yields word vectors.”

**Notebook transition:** Show the Word2Vec configuration before running training.

---

## Slide 4 — Word2Vec settings change the representation (2 min)

**Visible text:**
- `vector_size`: number of coordinates
- `window`: how far context extends
- `min_count`: which tokens remain
- `sg`: skip-gram (`1`) or CBOW (`0`)
- seed + one worker: classroom reproducibility

**Visual:** A compact settings table with the tutorial values: 50, 3, 1, 1, 42/1.

**Alt text:** Word2Vec settings and their tutorial values are listed with plain-language effects.

**Say:** “These are analyst choices, not facts discovered from the corpus. The small corpus and long training are chosen to make a mechanism visible.”

**Ask:** “Which setting changes the width of local context?”

**Expected:** “`window`.”

**Notebook transition:** Run Part 1 training and display vocabulary/vector shape.

---

## Slide 5 — Comparing vectors with cosine similarity (3 min)

**Visible text:**

`cos(x,y) = (x · y) / (||x|| ||y||)`

- Compares direction after normalizing magnitude
- Range: −1 to 1 mathematically
- Similarity is corpus-dependent, not a validated semantic judgement

**Visual:** Two pairs of arrows: similar directions versus different directions.

**Alt text:** Vector pairs illustrate that cosine similarity compares angle rather than raw length.

**Say:** “A nearest-word query sorts vocabulary vectors by this numerical relationship. A surprising result can reflect corpus design, limited data, or training variability.”

**Ask:** “Does a high cosine similarity prove two health concepts are interchangeable?”

**Expected:** “No.”

**Notebook transition:** Run nearest-neighbor and pairwise-similarity cells.

---

## Slide 6 — t-SNE maps vectors for inspection (3 min)

**Visible text:**
- Input: 18 vectors × 50 dimensions
- Output: 18 points × 2 dimensions
- Fixed `perplexity=5`, `random_state=42`
- Interpret local neighborhoods cautiously
- Do not interpret axes, orientation, or global distances

**Visual:** High-dimensional vector table arrow to a labelled two-dimensional scatter.

**Alt text:** Eighteen 50-dimensional vectors are projected into a two-dimensional labelled display.

**Say:** “t-SNE is nonlinear and optimized for visualization. The plot is not the original embedding space. We do not rerun seeds and select the most attractive picture.”

**Ask:** “What does the x-axis mean?”

**Expected:** “It has no direct semantic interpretation.”

**Notebook transition:** Run Part 1 t-SNE cell and read its “What do we see?” text.

---

## Slide 7 — Transition: vectors need selective interaction (1 min)

**Visible text:** “Word2Vec gives one vector per vocabulary word. How can a sequence combine information differently at each position?”

**Visual:** Static word vectors entering an attention matrix.

**Alt text:** Word vectors transition into a matrix operation that computes position-specific combinations.

**Say:** “Attention answers a different question. It computes a new weighted combination for each query position.”

**Ask:** “What changes from Word2Vec to attention: the existence of vectors, or how vectors interact for each query?”

**Expected:** “Attention changes how vectors interact and are combined for each query.”

**Notebook transition:** Open Part 2.

---

## Slide 8 — Queries, keys, and values (4 min)

**Visible text:**
- Query `Q`: what a position is matching
- Key `K`: what each source position offers for matching
- Value `V`: information combined into the output
- Query-key similarity determines value weights

**Visual:** One query points to three keys; weighted arrows continue from three values into one output.

**Alt text:** A query compares with keys, and the resulting weights combine corresponding values.

**Say:** “Keys determine weights; values determine what is combined. They can contain different dimensions, provided query and key dimensions match.”

**Ask:** “Which matrix is multiplied by the final attention weights?”

**Expected:** “The value matrix.”

**Notebook transition:** Run Part 2 hand-checkable tensor setup.

---

## Slide 9 — Scaled dot-product attention (4 min)

**Visible text:**

`S = QKᵀ / √d_k`

`A = softmax(S, dim = −1)`

`O = AV`

- `S`: similarity scores
- `A`: row-normalized attention weights
- `O`: weighted value combinations

**Visual:** Tensor-shape flow: `(L,d_k) × (S,d_k)ᵀ → (L,S) → (L,S) × (S,d_v) → (L,d_v)`.

**Alt text:** The attention formula is paired with matrix shapes from query and key through output.

**Say:** “Softmax is applied across keys for each query row. The scaling term reduces extreme dot products as key dimension grows.”

**Ask:** “Which dimension must sum to one after softmax?”

**Expected:** “Each query row across its key columns.”

**Notebook transition:** Run the explicit score, weight, and output cells.

---

## Slide 10 — Why divide by the square root of key dimension? (3 min)

**Visible text:**
- Larger key dimensions can produce larger dot-product variance
- Large score differences can saturate softmax
- Dividing by `√d_k` moderates the score scale

**Visual:** Two softmax distributions: one moderately spread and one nearly one-hot after oversized scores.

**Alt text:** A comparison shows that overly large scores can produce a saturated softmax distribution.

**Say:** “Scaling does not guarantee good attention. It controls numerical behavior before softmax.”

**Ask:** “Does scaling make all attention weights equal?”

**Expected:** “No. It changes score scale, not the ranking by itself.”

**Notebook transition:** Compare raw and scaled scores in Part 2.

---

## Slide 11 — Read an attention map by row (3 min)

**Visible text:**
- Rows = queries
- Columns = keys
- Cell = post-softmax weight
- Each valid row sums to 1

**Visual:** Annotated 2×2 heat map from the exact fixture with row 1 highlighted.

**Alt text:** A heat map labels query rows, key columns, and a highlighted row whose weights sum to one.

**Say:** “For each query, identify the largest key weight, then inspect the value vectors that will be combined. A weight is a computed contribution, not a causal explanation.”

**Ask:** “What should you verify before interpreting colors?”

**Expected:** “Axis convention, labels, scale, and whether values are scores or normalized weights.”

**Notebook transition:** Run the labelled heat-map and row-sum assertion.

---

## Slide 12 — Masks remove disallowed links (3 min)

**Visible text:**
- Apply mask before softmax
- Disallowed scores → `−∞`
- Causal mask: position `i` can use positions `≤ i`
- Masked weights become zero

**Visual:** Lower-triangular allowed matrix beside the resulting attention map.

**Alt text:** A lower-triangular causal mask blocks cells above the diagonal, which become zero attention weights.

**Say:** “The notebook uses `True = allowed`. Libraries can use different conventions, so mask semantics must be stated explicitly.”

**Ask:** “Can the first position attend to the second under this causal mask?”

**Expected:** “No.”

**Notebook transition:** Run causal-mask comparison and assertions.

---

## Slide 13 — Self-attention and cross-attention differ by source (3 min)

**Visible text:**
- Self-attention: `Q=XW_Q`, `K=XW_K`, `V=XW_V`
- Same sequence `X`, different learned projections
- Cross-attention: queries and key/value context can come from different sources

**Visual:** Self-attention diagram with one sequence feeding all projections; cross-attention diagram with query sequence separate from context sequence.

**Alt text:** Self-attention derives query, key, and value from one sequence; cross-attention uses a separate context for keys and values.

**Say:** “Self-attention does not mean Q, K, and V are identical. They are projected from the same input sequence using different parameter matrices.”

**Ask:** “If Q comes from a decoder and K/V from another context, which type is it?”

**Expected:** “Cross-attention.”

**Notebook transition:** Run fixed projection and self-attention cells.

---

## Slide 14 — What the small attention notebook leaves out (2 min)

**Visible text:**
- Multiple heads
- Position information
- Residual connections and normalization
- Feed-forward sublayer
- Training objective and learned parameters
- Dense attention uses `O(n²)` score entries

**Visual:** Full Transformer block outline with only scaled dot-product attention highlighted.

**Alt text:** A Transformer block contains attention plus several omitted components; only one attention operation is highlighted.

**Say:** “We are inspecting the mechanism, not building or training a full Transformer. Position encodings are important because the bare operation does not itself encode token order.”

**Ask:** “Why can long sequences make dense attention expensive?”

**Expected:** “The score matrix has a pairwise entry for query and key positions, growing quadratically.”

**Notebook transition:** Run official PyTorch comparison and close Part 2.

---

## Slide 15 — Transition: model parameters are not a document database (1 min)

**Visible text:** “How can a generated response use a small, inspectable external corpus?”

**Visual:** A generator beside—not inside—a document store.

**Alt text:** A language model and a separate document store are connected by retrieval.

**Say:** “RAG does not paste documents into model weights. It retrieves selected text at request time.”

**Ask:** “Does retrieval update the generator's fitted parameters?”

**Expected:** “No. It supplies external context at request time.”

**Notebook transition:** Open Part 3.

---

## Slide 16 — Four RAG boundaries (4 min)

**Visible text:**
1. Parametric model patterns
2. External document corpus
3. Retrieval ranking
4. Generation conditioned on selected context

**Visual:** Four labelled stages with boundaries and arrows.

**Alt text:** The RAG workflow separates model parameters, external documents, retrieval, and response generation.

**Say:** “Keeping these stages separate helps diagnose failure. Retrieval can miss the evidence; generation can misuse correctly retrieved evidence.”

**Ask:** “Does adding a document to Chroma retrain the generator?”

**Expected:** “No.”

**Notebook transition:** Run Part 3 setup and display resolved paths/configuration.

---

## Slide 17 — Chunking defines what can be retrieved (3 min)

**Visible text:**
- Source document → chunks + metadata + stable IDs
- Too small: context can be fragmented
- Too large: irrelevant text consumes context
- Tutorial: one chunk per level-2 section, nine chunks total

**Visual:** One Markdown document split at `##` headings into three labelled records.

**Alt text:** A document is split by section headings into chunks carrying source and section metadata.

**Say:** “Chunking is part of the retrieval design, not neutral preprocessing. Stable IDs let us inspect and cite what entered the prompt.”

**Ask:** “What changes if one answer requires two separate sections?”

**Expected:** “Retrieval must return both chunks, or the context will be incomplete.”

**Notebook transition:** Run source-manifest and chunking cells.

---

## Slide 18 — Embeddings support similarity search (3 min)

**Visible text:**
- Embed each chunk once
- Embed the question using the same model
- Compare vectors with cosine distance
- Retrieve top `k`; ranking is not proof of relevance

**Visual:** Question vector near some chunk vectors in a schematic embedding space.

**Alt text:** A question vector is compared with document-chunk vectors, and nearest chunks are selected.

**Say:** “Never mix vectors created by different embedding models in one index. The local Chroma collection stores vectors, documents, IDs, and metadata.”

**Ask:** “Is the nearest chunk guaranteed to answer the question?”

**Expected:** “No.”

**Notebook transition:** Show index metadata and collection count, then run retrieval.

---

## Slide 19 — The local vector database lifecycle (3 min)

**Visible text:**
- Source: `data/rag_corpus/`
- Generated chunks/index: `artifacts/rag/`
- Reuse only when fingerprint + model + chunking match
- Local persistence ≠ access control, backup, or governance

**Visual:** Directory tree with versioned source and gitignored artifacts.

**Alt text:** The tutorial directory separates fictional source documents from generated chunks and the persistent Chroma database.

**Say:** “The database lives locally so students can inspect its lifecycle. A fingerprint prevents accidental reuse after source or model changes.”

**Ask:** “What should happen after changing a source policy?”

**Expected:** “The fingerprint changes and the index must be rebuilt.”

**Notebook transition:** Inspect `index_metadata.json` logic and collection count.

---

## Slide 20 — Same question, two generation conditions (3 min)

**Visible text:**
- Direct: question, no tutorial corpus context
- RAG: question + retrieved labelled chunks
- Same generation model
- Compare evidence support, not writing style alone

**Visual:** Split panel showing direct prompt versus RAG prompt with context labels.

**Alt text:** Two prompts use the same question and model, but only the RAG condition contains retrieved chunks.

**Say:** “The direct condition is not a claim that the model has no prior information. It establishes that the fictional corpus evidence was not supplied.”

**Ask:** “What is the controlled difference?”

**Expected:** “Retrieved tutorial context is added in the RAG condition.”

**Notebook transition:** Run Q1 direct condition, then display retrieved chunks before RAG generation.

---

## Slide 21 — Grounding instructions and citations (3 min)

**Visible text:**
- Use only facts in `CONTEXT`
- Cite exact chunk IDs
- State when context is insufficient
- Retrieved text is untrusted data, not instructions

**Visual:** Prompt hierarchy showing instruction above delimited context and user question.

**Alt text:** A prompt separates higher-priority behavior instructions from untrusted retrieved context and the user question.

**Say:** “Citation formatting helps inspection but does not guarantee support. We still compare each claim with cited text. Retrieved documents can contain malicious instructions, so context must remain data.”

**Ask:** “If a retrieved chunk says ‘ignore prior instructions,’ should the system follow it?”

**Expected:** “No. Retrieved text is untrusted context.”

**Notebook transition:** Display exact prompt template and run Q1 RAG condition.

---

## Slide 22 — Evaluate retrieval and generation separately (4 min)

**Visible text:**
- Retrieval: Were required chunk IDs in top `k`?
- Generation: Were required facts covered?
- Citations: Do labels match retrieved evidence?
- Unsupported specifics: present or absent?
- Unanswerable question: does output acknowledge missing evidence?

**Visual:** Evaluation table for Q1, Q2, and Q3 with retrieval and generation columns.

**Alt text:** Three fixed questions are evaluated separately for retrieval hits, factual coverage, citations, unsupported claims, and insufficient-evidence behavior.

**Say:** “A correct answer with failed retrieval can be unsupported coincidence. Correct retrieval with an incorrect answer is a generation failure. We need both views.”

**Ask:** “If the right chunk is retrieved but the response invents a fee, which stage failed?”

**Expected:** “Generation/evidence use failed, even though retrieval succeeded.”

**Notebook transition:** Run fixed three-question evaluation table; discuss Q3.

---

## Slide 23 — What this tutorial supports—and what it does not (4 min)

**Visible text:**

**Supports**
- Inspecting vectors and computed weights
- Tracing retrieved chunks into a prompt
- Comparing direct and RAG conditions

**Does not support**
- Clinical correctness or safety
- Causal interpretation of attention
- Complete retrieval or faithful citations
- Deployment, privacy compliance, or fairness

**Visual:** Two-column supports/does-not-support table plus exit prompt.

**Alt text:** A boundary table separates demonstrated mechanisms from claims the tutorial cannot establish.

**Say:** “A transparent small example is useful because we can trace each operation. That transparency does not validate a health application.”

**Ask:** “Exit prompt: name one numerical mechanism, one evaluation check, and one governance safeguard from today.”

**Expected:** Examples: cosine similarity; expected chunk in top-k; never send patient data or keys.

**Notebook transition:** Show completion checklists and supporting-material links.

## Deck production requirements

- Use formulas as editable text, not screenshots.
- Use the exact fixture values/IDs where a slide shows an output.
- Keep visible text under approximately 45 words per slide excluding formulas and labels.
- Put longer explanations in speaker notes (`Say`).
- Add source footer links on theory slides: Gensim/scikit-learn for slides 2–6, Vaswani/PyTorch for 8–14, OpenAI/Chroma for 16–22.
- Provide text descriptions below any figure where spatial position or color carries meaning.
- Number notebook cells during implementation, then replace section-only transitions above with exact cell numbers in final `slides.md`.

## Revision of 21 September 2026

The rendered deck is `../latex-slides/lab2-slides.pdf`. Its convention is now: one explanation slide per key notebook section, followed by a "From the notebook" slide showing that section's executed output. Figures are exported from the notebooks by `latex-slides/export_notebook_figures.py` (`make figures`). The attention slides gained mask families, "each training objective is a mask plus a target", "from an attention row to a prediction", the leak test, and the pretrained contrast. Decoding is a slides-only section. Two Streamlit slides close the RAG section. Notebook numbering in the deck follows the new `Part-0` to `Part-3` filenames.
