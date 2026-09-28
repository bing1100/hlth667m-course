# Slides — Tokenization, Representations, Attention, Decoding, and RAG (Tutorial 2)

> **Presenting from the Beamer deck?** Use `../../latex-slides/speaker-script.md`. It has one entry per page of `lab2-slides.pdf`, in the deck's current order, and `make script` in that folder checks that it still matches. This file is the longer planning outline for the same material, and its slide order can lag behind the deck.

## Purpose and timing

This deck accompanies the four notebooks in `part-2-rag/` (Tutorial 2, Parts 0 to 3). Slides carry the theory, formulas, diagrams, safeguards, and discussion questions. The notebooks carry the executable calculations and outputs.

The deck pairs explanation slides with **From the notebook** example slides. Each example slide shows one executed figure or table, and its **Visual** field names the notebook and section the figure comes from.

| Slides | Section | Notebook |
|---|---|---|
| 0–1 | Opening and boundaries | — |
| 2–4 | Tokenization | Part 0 |
| 5–12 | Word2Vec and t-SNE, and the transition to attention | Part 1 |
| 13–35 | Attention, masks, and training objectives | Part 2 |
| 36–39 | Decoding | **Slides only — no notebook** |
| 40–55 | RAG, structured knowledge, and the Streamlit chat app | Part 3 |

Decoding (temperature, top-k, top-p) is taught from this deck only. The former decoding notebook is archived at `archive/Decoding_Temperature_TopK_TopP.ipynb`.

The per-slide minutes are pacing maxima that include the linked discussion or a short notebook handoff. They are not added to the notebook durations, so their sum is not a second estimate of class length. Follow `instructor_guide.md` as the authoritative schedule.

Every visual has alt text. Every formula defines its symbols.

### Preview and screenshot note

Open this file in a Markdown preview that supports MathJax/KaTeX (for example, VS Code's built-in Markdown preview). Equations are written as standalone double-dollar LaTeX blocks so they render at presentation size and can be captured cleanly. Keep an equation and its symbol key together when taking a screenshot.

---

## Slide 0 — From text to grounded responses (3 min)

**Visible text:**
- Tokens → word vectors → attention → decoding → retrieval-augmented generation
- Main question: How does text become numerical input, and how can external evidence enter a generated response?

**Visual:** Three connected boxes: local context produces word vectors; Q/K/V produce weighted combinations; a question retrieves chunks that enter a generation prompt.

**Alt text:** A left-to-right sequence connects Word2Vec, attention, and RAG while showing that each stage transforms text or vectors.

**Say:** “We will inspect three different mechanisms. Word2Vec learns word-level vectors from co-occurrence. Attention combines value vectors using query-key similarity. RAG retrieves external text and places it into a generation request.”

**Ask:** “Which of these stages retrieves an external document?”

**Expected:** “RAG retrieval.”

**Notebook transition:** Open all four notebook titles (Parts 0 to 3); do not run code yet.

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

**Notebook transition:** Part 3 title and safety notice.

---

## Slide 2 — A model never reads words (3 min)

**Visible text:**
- Text → **tokens** → numbers
- Frequent words become one token; rare clinical terms become several

**Visual:** One sentence split twice: by spaces into words, then by a subword tokenizer into smaller pieces, with the pieces of `myocardial` highlighted.

**Alt text:** The same clinical sentence splits into fewer whitespace words than subword tokens, and the drug and diagnosis terms break into multiple fragments.

**Say:** "Before any model does anything, a tokenizer converts text into units from a fixed vocabulary. Common English words usually cost one token. Clinical vocabulary usually costs several."

**Ask:** "How many tokens do you think `HbA1c` becomes?"

**Expected:** "Five — roughly one per character."

**Notebook transition:** Open Part 0 and run sections 1 and 2.

---

## Slide 3 — Byte pair encoding learns the vocabulary (3 min)

**Visible text:**
- Start from characters
- Repeatedly merge the most frequent adjacent pair
- The ordered list of merges *is* the tokenizer

**Visual:** Three rows showing `l o w e r` merging to `lo w e r`, then `low e r`, then `lower`, with a merge counter beside each step.

**Alt text:** Successive byte pair encoding merges combine adjacent symbols until a frequent word becomes a single token.

**Say:** "Nobody tells the algorithm what a word is. It merges whatever is frequent. That is why the vocabulary reflects the training corpus rather than any linguistic or clinical theory."

**Ask:** "If a term never appeared in training, what happens to it?"

**Expected:** "It still encodes, but as many small fragments. Nothing is ever unknown."

**Notebook transition:** Run Part 0 sections 3 to 5 and read the merge table.

---

## Slide 4 — Tokens decide cost and context (3 min)

**Visible text:**
- Plain English ≈ 1.1 tokens per word
- Laboratory panel ≈ 4 tokens per word
- Budget and context limits are measured in tokens, not words

**Visual:** From the notebook — `Part-0_Tokenization_for_Health_Text.ipynb`, section 7: horizontal bar chart of tokens per word for plain English, clinic admin text, clinical narrative, medication list, and lab panel, with a dashed line at 1.0.

**Alt text:** Tokens per word rises from about 1.1 for plain prose to about 4.0 for a laboratory panel.

**Say:** "A budget estimated from ordinary prose will understate clinical text by a factor of two to four. A context window described in words holds much less clinical text than you expect."

**Ask:** "Two phrases mean the same thing to a clinician. Must they share tokens?"

**Expected:** "No. `bloodwork` and `laboratory result` share none, which is why Part 3 adds a terminology."

**Notebook transition:** Run Part 0 sections 7 and 8, then close Part 0.

---

## Slide 5 — From token identity to token meaning (3 min)

**Visible text:**
- Part 0 gave each token an **identity**. It gave it no **meaning**.
- Context window: the nearby tokens used to build training examples
- Embedding: a dense numerical vector
- Distributional premise: similar local contexts can produce similar vectors

**Visual:** Sentence `clinic schedules followup through portal` with a window of three tokens around `followup`.

**Alt text:** A sentence highlights a center token and nearby context tokens used by Word2Vec.

**Say:** “A token ID is just a row number. Word2Vec does not begin with dictionary definitions either — it creates training examples from neighbouring tokens. What it learns depends entirely on the supplied corpus and preprocessing.”

**Ask:** “If the corpus changes, must the vectors stay the same?”

**Expected:** “No. Vectors depend on corpus context and settings.”

**Notebook transition:** Run Part 1's repetitive baseline and varied-corpus cells. Ask why duplicated rows are not the same as new contexts.

---

## Slide 6 — Two Word2Vec training directions (3 min)

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

## Slide 7 — Word2Vec settings change the representation (2 min)

**Visible text:**
- `vector_size`: number of coordinates
- `window`: how far context extends
- `min_count`: which tokens remain
- `sg`: skip-gram (`1`) or CBOW (`0`)
- seed + one worker: classroom reproducibility

**Visual:** A compact settings table with the tutorial values: 50, 3, 1, 1, 42/1.

**Alt text:** Word2Vec settings and their tutorial values are listed with plain-language effects.

**Say:** “These are analyst choices, not facts discovered from the corpus. We hold the main settings fixed while comparing a repetitive template with more varied contexts. More rows help only when they supply useful evidence.”

**Ask:** “Which setting changes the width of local context?”

**Expected:** “`window`.”

**Notebook transition:** Run Part 1 training and display vocabulary/vector shape.

---

## Slide 8 — Comparing vectors with cosine similarity (3 min)

**Visible text:**

$$
\operatorname{cos\_sim}(\mathbf{x},\mathbf{y})
=
\frac{\mathbf{x}^{\mathsf T}\mathbf{y}}
{\lVert\mathbf{x}\rVert_2\,\lVert\mathbf{y}\rVert_2}
$$

- Compares direction after normalizing magnitude
- Range: −1 to 1 mathematically
- Similarity is corpus-dependent, not a validated semantic judgement

**Visual:** Two pairs of arrows: similar directions versus different directions.

**Alt text:** Vector pairs illustrate that cosine similarity compares angle rather than raw length.

**Say:** “Read the numerator as a dot product: vectors pointing in similar directions have a larger value. The denominator divides by both lengths, so direction matters more than magnitude. Here, $\mathbf{x}$ and $\mathbf{y}$ are vectors, $\mathbf{x}^{\mathsf T}\mathbf{y}$ is their dot product, and $\lVert\mathbf{x}\rVert_2$ is the Euclidean length of $\mathbf{x}$. A nearest-word query sorts vocabulary vectors by this numerical relationship. Similarity still reflects the corpus and training setup—not validated clinical meaning.”

**Ask:** “Does a high cosine similarity prove two health concepts are interchangeable?”

**Expected:** “No.”

**Notebook transition:** Compare the baseline/varied cosine summaries, then inspect the fixed queries and 50-D heat map.

---

## Slide 9 — From the notebook: similarity in the varied corpus (2 min)

**Visible text:**
- Read one row at a time
- Words used in similar sentences score high
- The block structure is what retrieval will rely on
- Similarity is corpus-dependent

**Visual:** From the notebook — `Part-1_Word2Vec_and_tSNE.ipynb`, section 5: annotated heat map of cosine similarities between the selected words in the varied-corpus model.

**Alt text:** A square heat map of pairwise cosine similarities shows blocks of higher values among words that appear in similar fictional clinic sentences.

**Say:** “Judge pair similarity from this heat map, which uses all 50 dimensions. The 2-D map on the next slides is a view of these vectors and carries less information.”

**Ask:** “Which should you trust for a pair of words: this heat map or the distance on a 2-D map?”

**Expected:** “The heat map.”

**Notebook transition:** Run the Part 1 section 5 neighbours and heat-map cells.

---

## Slide 10 — t-SNE maps vectors for inspection (3 min)

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

**Notebook transition:** Run Part 1 section 6 t-SNE cell and read its “What do we see?” text.

---

## Slide 11 — From the notebook: the map from one run (2 min)

**Visible text:**
- 18 labelled points from one fixed seed
- Trust the **local neighbourhoods**
- A different seed moves the groups and changes the distances between them

**Visual:** From the notebook — `Part-1_Word2Vec_and_tSNE.ipynb`, section 6: labelled 2-D t-SNE scatter of the 18 selected word vectors.

**Alt text:** A scatter plot places eighteen labelled words in two dimensions, with related words near one another.

**Say:** “This is one run with `perplexity=5` and `random_state=42`. Read which words sit together. The axes, the orientation, and the gaps between groups carry no meaning.”

**Ask:** “Two groups sit far apart on this map. What can you conclude about them?”

**Expected:** “Nothing about how different they are. Global distance on a t-SNE map is unreliable.”

**Notebook transition:** Read the Part 1 section 6 “What do we see?” text, then close Part 1.

---

## Slide 12 — Transition: vectors need selective interaction (1 min)

**Visible text:** “Word2Vec gives one vector per vocabulary word. How can a sequence combine information differently at each position?”

**Visual:** Static word vectors entering an attention matrix.

**Alt text:** Word vectors transition into a matrix operation that computes position-specific combinations.

**Say:** “Attention answers a different question. It computes a new weighted combination for each query position.”

**Ask:** “What changes from Word2Vec to attention: the existence of vectors, or how vectors interact for each query?”

**Expected:** “Attention changes how vectors interact and are combined for each query.”

**Notebook transition:** Open Part 2.

---

## Slide 13 — Learning language from raw text: predict the next token (3 min)

**Visible text:**
- Labelled health text is scarce; unlabelled text is almost unlimited
- Let the text label itself: hide part of it, train the model to predict what was hidden
- **Autoregressive (AR):** hide the next token, read only what came before

$$
P(x_1,\dots,x_n)=\prod_{t=1}^{n}P\left(x_t\mid x_1,\dots,x_{t-1}\right)
$$

**Visual:** The tokens `An urgent result is communicated by the` followed by a gold `?` box, with a brace under the visible tokens labelled "sees only what came before", and a loop: predict one token → append it → repeat.

**Alt text:** A sentence prefix feeds a prediction of the next token, and a loop shows that the predicted token is appended before the next prediction.

**Say:** “This is self-supervised learning: the text supplies its own labels. The formula says the probability of a whole sentence is a product of next-token probabilities, where $x_t$ is the token at position $t$ and each factor conditions on the tokens before it. GPT-style models generate this way, one token at a time, left to right.”

**Ask:** “Who wrote the labels for this training task?”

**Expected:** “Nobody. The next token in the text is the label.”

**Notebook transition:** Read the Part 2 guiding question. Do not run code yet.

---

## Slide 14 — Masked language modelling: fill in the blank (3 min)

**Visible text:**
- **Masked language modelling (MLM):** hide a random share of tokens, about 15% in BERT
- Predict each hidden token using the words on **both** sides
- These models encode text into representations for classifying and searching

**Visual:** The tokens `The nurse [MASK] the laboratory result` with braces marking left context and right context, and the prediction `reviewed` above the `[MASK]` box.

**Alt text:** A sentence with one masked token shows context on both sides of the blank contributing to the prediction of the hidden word.

**Say:** “The second way to let text label itself is to blank out tokens anywhere in the sentence. The model may read to the left and to the right of each blank. BERT-style encoders and the embedding models used for retrieval are trained this way.”

**Ask:** “Which side of the blank holds the most useful clue in this sentence?”

**Expected:** “The right side: `the laboratory result` says what kind of action fits.”

**Notebook transition:** Stay on the slides.

---

## Slide 15 — Same goal, different strengths, one shared operation (3 min)

**Visible text:**

| | Autoregressive | Masked language model |
|---|---|---|
| Hides | the next token | random tokens anywhere |
| May look at | earlier tokens only | tokens on both sides |
| Natural use | writing: answers, summaries, drafts | reading: classifying notes, coding, search embeddings |
| Families | GPT-style decoders | BERT-style encoders |

**Visual:** Two 4×4 grids side by side. The autoregressive grid is green on and below the diagonal; the masked grid is green everywhere. Rows are the predicting position and green cells are positions it may use.

**Alt text:** A lower-triangular grid represents autoregressive visibility and a fully filled grid represents masked-language-model visibility.

**Say:** “To predict a hidden token, a position gathers information from the positions it is allowed to see. That gathering step is attention. The two training styles use the same attention arithmetic and differ in which positions are visible. Part 2 trains one small model each way.”

**Ask:** “Which family would you choose to turn a paragraph into one search vector?”

**Expected:** “The masked family, because every token is read in the context of the whole passage.”

**Notebook transition:** Run Part 2 section 0 (setup).

---

## Slide 16 — Attention ends in a weighted average (3 min)

**Visible text:**
- Attention computes a weighted combination of **value vectors**
- Weights are non-negative and sum to one
- Example: weights `0.75` and `0.25` on values `[10, 0]` and `[0, 20]` give `[7.5, 5.0]`

$$
\mathbf{o}_i=\sum_{j=1}^{S} a_{ij}\mathbf{v}_j,
\qquad \sum_{j=1}^{S}a_{ij}=1
$$

**Visual:** Two value vectors scaled by 0.75 and 0.25 and added into one output vector. The numbers come from Part 2 section 1 of `Part-2_Attention_and_Self_Attention.ipynb`.

**Alt text:** Two value rows are multiplied by their weights and summed into the output row seven point five, five point zero.

**Say:** “Start with the destination. For query position $i$, the output $\mathbf{o}_i$ is a blend of value vectors $\mathbf{v}_j$, and the weights $a_{ij}$ across the $S$ available keys sum to one. Attention blends several values. A larger weight means a larger numerical contribution in this operation. The rest of the notebook is about where the weights come from and which positions may receive any weight.”

**Ask:** “With weights 0.75 and 0.25, how much more does the first value contribute than the second?”

**Expected:** “Three times as much.”

**Notebook transition:** Run Part 2 section 1 and confirm the output `[7.5, 5.0]`.

---

## Slide 17 — Queries, keys, and values (3 min)

**Visible text:**
- Query `Q`: what a position is matching
- Key `K`: what each source position offers for matching
- Value `V`: information combined into the output
- Query–key similarity determines the value weights

**Visual:** One query points to three keys; weighted arrows continue from three values into one output.

**Alt text:** A query compares with keys, and the resulting weights combine corresponding values.

**Say:** “Queries and keys decide the weights; values provide the information being blended. This bridge from the weighted average is more important than memorizing the letters.”

**Ask:** “Which matrix is multiplied by the final attention weights?”

**Expected:** “The value matrix.”

**Notebook transition:** Read the five-step recipe in Part 2 section 2.

---

## Slide 18 — Scaled dot-product attention, with a Mask step (4 min)

**Visible text:**

$$
\mathbf{S}=\frac{\mathbf{Q}\mathbf{K}^{\mathsf T}}{\sqrt{d_k}}
$$

$$
\widetilde{\mathbf{S}}=\operatorname{mask}(\mathbf{S}),
\qquad
\mathbf{A}=\operatorname{softmax}(\widetilde{\mathbf{S}},\text{ across keys}),
\qquad
\mathbf{O}=\mathbf{A}\mathbf{V}
$$

1. **Match** 2. **Scale** 3. **Mask** 4. **Normalize** 5. **Combine**

**Visual:** Five boxes in a row: `Q × Kᵀ` → `÷ √d_k` → `mask: blocked → −∞` (outlined in bold) → `softmax` → `× V`, with the tensor shapes `(L,d_k) × (S,d_k)ᵀ → (L,S) → (L,S) × (S,d_v) → (L,d_v)` underneath. This is the recipe figure from Part 2 section 2 of `Part-2_Attention_and_Self_Attention.ipynb`.

**Alt text:** A five-step flow shows match, scale, mask, normalize, and combine, with the mask step emphasised and matrix shapes listed from query and key through output.

**Say:** “Read this in five pauses. $\mathbf{Q}\mathbf{K}^{\mathsf T}$ compares every query with every key. Dividing by $\sqrt{d_k}$ keeps scores moderate. The mask sets every disallowed query–key score to negative infinity. Softmax across each row gives weights summing to one, and a negative-infinity score becomes a weight of exactly zero. Multiplying by $\mathbf{V}$ blends the values. Here $L$ is the number of queries, $S$ the number of source positions, $d_k$ the query/key width, and $d_v$ the value width. Step 3 is the one this notebook studies; the other four are identical in every model you will meet.”

**Ask:** “Which dimension must sum to one after softmax?”

**Expected:** “Each query row across its key columns.”

**Notebook transition:** Run the Part 2 section 2 recipe figure.

---

## Slide 19 — Read an attention map by row (3 min)

**Visible text:**
- Rows = queries
- Columns = keys
- Cell = post-softmax weight
- Each valid row sums to 1

$$
\operatorname{softmax}(s_{i,j})
=
\frac{e^{s_{i,j}}}{\sum_{m=1}^{S}e^{s_{i,m}}}
\quad\Rightarrow\quad
\sum_{j=1}^{S}a_{i,j}=1
$$

**Visual:** Annotated 2×2 heat map from the exact fixture with row 1 highlighted.

**Alt text:** A heat map labels query rows, key columns, and a highlighted row whose weights sum to one.

**Say:** “Hold query row $i$ fixed. Exponentiate each score in that row, then divide by the row's total. That creates non-negative weights summing to one. Read one row slowly: identify its largest key weight, then inspect the corresponding value vectors. A weight is a computed contribution in this operation, not automatically a causal explanation.”

**Ask:** “What should you verify before interpreting colors?”

**Expected:** “Axis convention, labels, scale, and whether values are scores or normalized weights.”

**Notebook transition:** Run Part 2 section 3.

---

## Slide 20 — From the notebook: scores, weights, output (3 min)

**Visible text:**
- Query `clinic`: scaled scores `[0.707, 0.000]`
- Softmax weights `[0.670, 0.330]`
- Output `0.670 × [10, 0] + 0.330 × [0, 20] = [6.70, 6.60]`
- Weights and values both matter

**Visual:** From the notebook — `Part-2_Attention_and_Self_Attention.ipynb`, section 3: three annotated heat maps titled "Scaled match scores", "Row-wise softmax weights", and "Weighted output" for the two tokens `clinic` and `portal`.

**Alt text:** Three small heat maps show the scaled scores, the softmax weights whose rows sum to one, and the blended output for a two-token example.

**Say:** “Five lines of code produce all three tables. The second output coordinate is large even though its weight is small, because the second value contains 20. The `assert_close` line confirms that our function matches PyTorch's `scaled_dot_product_attention`. Everything that follows changes what goes into this function.”

**Ask:** “Why is the second output coordinate 6.60 when the second weight is only 0.330?”

**Expected:** “The weight multiplies a value of 20.”

**Notebook transition:** Read "Read one row slowly" under the Part 2 section 3 figure.

---

## Slide 21 — Why divide by the square root of key dimension? (2 min)

**Visible text:**
- Larger key dimensions can produce larger dot-product variance
- Large score differences can saturate softmax
- Dividing by `√d_k` moderates the score scale

$$
\operatorname{Var}(\mathbf{q}\cdot\mathbf{k})\approx d_k
\quad\Longrightarrow\quad
\operatorname{Var}\!\left(\frac{\mathbf{q}\cdot\mathbf{k}}{\sqrt{d_k}}\right)\approx 1
$$

**Visual:** Two softmax distributions: one moderately spread and one nearly one-hot after oversized scores.

**Alt text:** A comparison shows that overly large scores can produce a saturated softmax distribution.

**Say:** “This is an intuition under simplifying assumptions, not a promise about every learned vector. If coordinates contribute roughly independent unit-scale terms, summing $d_k$ products makes variance grow with $d_k$. Dividing by $\sqrt{d_k}$ keeps the scale more stable before softmax. Scaling does not make the weights equal or guarantee useful attention.”

**Ask:** “Does scaling make all attention weights equal?”

**Expected:** “No. It changes score scale, not the ranking by itself.”

**Notebook transition:** Slides only. The scores in Part 2 section 3 are already scaled: `0.707` is `1 / √2`.

---

## Slide 22 — Self-attention and cross-attention differ by source (3 min)

**Visible text:**
- Self-attention: one input sequence supplies Q, K, and V
- Same sequence `X`, different learned projections
- Cross-attention: queries and key/value context can come from different sources

$$
\mathbf{Q}=\mathbf{X}\mathbf{W}_Q,
\qquad
\mathbf{K}=\mathbf{X}\mathbf{W}_K,
\qquad
\mathbf{V}=\mathbf{X}\mathbf{W}_V
$$

**Visual:** Self-attention diagram with one sequence feeding all projections; cross-attention diagram with query sequence separate from context sequence.

**Alt text:** Self-attention derives query, key, and value from one sequence; cross-attention uses a separate context for keys and values.

**Say:** “$\mathbf{X}$ is the sequence representation matrix. Multiplying it by three learned matrices creates three roles for the same positions. Self-attention therefore does not mean Q, K, and V are identical. In cross-attention, the query sequence and the key/value context can instead come from different sources.”

**Ask:** “If Q comes from a decoder and K/V from another context, which type is it?”

**Expected:** “Cross-attention.”

**Notebook transition:** Run the Part 2 section 4 projection and self-attention cell.

---

## Slide 23 — A causal mask hides the future (3 min)

**Visible text:**
- Apply the mask before softmax
- Disallowed scores → `−∞`, so masked weights become exactly zero
- Causal mask: position `i` can use positions `≤ i`
- First row of the worked example: `[0.670, 0.330] → [1, 0]`

$$
\widetilde{s}_{ij}=
\begin{cases}
s_{ij}, & j\le i \\
-\infty, & j>i
\end{cases}
\qquad
a_{ij}=\operatorname{softmax}(\widetilde{s}_{ij})
$$

**Visual:** A 4×4 grid with green cells on and below the diagonal and `−∞` in the cells above it, with rows labelled query position and columns labelled key position.

**Alt text:** A lower-triangular causal mask blocks cells above the diagonal, which become zero attention weights.

**Say:** “This is the autoregressive rule, enforced inside attention. For a causal mask, query position $i$ may use key position $j$ only when $j\le i$. Replacing a blocked score by negative infinity makes its exponential, and therefore its softmax weight, zero. The notebook uses `True = allowed`; libraries can use different conventions, so always state the convention.”

**Ask:** “Can the first position attend to the second under this causal mask?”

**Expected:** “No.”

**Notebook transition:** Look at the right-hand panel of the Part 2 section 4 figure.

---

## Slide 24 — From the notebook: same Q, K and V; only the mask differs (3 min)

**Visible text:**
- Left: no mask, every token reads every token
- Right: causal mask, a token reads itself and earlier tokens
- Blocked cells are exactly 0 and every row still sums to 1
- `clinic` reads only itself: `[1, 0, 0, 0]`

**Visual:** From the notebook — `Part-2_Attention_and_Self_Attention.ipynb`, section 4: two 4×4 annotated heat maps for `clinic schedules followup today`, titled "No mask" and "Causal mask".

**Alt text:** Two attention heat maps share the same tokens; the causal one has zeros above the diagonal and larger weights on the remaining cells.

**Say:** “Both matrices use the same Q, K and V. The weight that would have gone to later tokens is redistributed over the allowed ones. Masking happens before softmax, so it changes what information is available to a position.”

**Ask:** “Where did the weight from the blocked cells go?”

**Expected:** “It was redistributed across the allowed cells in the same row, because each row still sums to one.”

**Notebook transition:** Read the Part 2 section 4 "What do we see?" text.

---

## Slide 25 — A mask is a table of who may read whom (3 min)

**Visible text:**

| Pattern | Rule for each row | Where it is used |
|---|---|---|
| Bidirectional | read every token | BERT-style encoders |
| Causal | read yourself and earlier tokens | GPT-style decoders, chat |
| Padding | nobody reads `[PAD]` filler | batches of unequal-length texts |
| Causal + padding | both rules at once | decoder training in batches |
| Prefix | prompt both ways, then causal | encoder–decoder models (T5) |
| Sliding window | read only the last `w` tokens | long-context models |

**Visual:** The mask-families table above, with one small green-and-blank grid icon beside each row.

**Alt text:** A table lists six attention-mask patterns with the rule each applies to a query row and the model families that use it.

**Say:** “A mask is a Boolean table with one row per query and one column per key. The arithmetic never changes. Model families differ, to a large extent, in which cells are open.”

**Ask:** “Which pattern does a chat model use while it writes a reply?”

**Expected:** “Causal. It cannot read tokens it has not produced yet.”

**Notebook transition:** Run Part 2 section 5.

---

## Slide 26 — From the notebook: six masks on one sentence (3 min)

**Visible text:**
- One sentence: `the patient reports cough today [PAD]`
- Follow the row for `cough` across the six panels
- Full: `cough` reads `today`; causal: it stops at itself
- Sliding window `w = 3`: 14 of 36 pairs allowed

$$
\text{full attention: } n^2 \text{ comparisons}
\qquad
\text{window of width } w:\; \approx n\times w
$$

**Visual:** From the notebook — `Part-2_Attention_and_Self_Attention.ipynb`, section 5: a 2×3 gallery of green (allowed) and pale (blocked) grids titled Bidirectional, Causal, Padding, Causal + padding, Prefix, and Sliding window, each with its count of allowed pairs.

**Alt text:** Six mask grids on the same six-token sentence show progressively different sets of allowed query–key pairs, with the padding column blocked in four of them.

**Say:** “The padding mask removes a column: no token reads `[PAD]`. The `[PAD]` row is still computed, and its output is ignored when the loss is calculated. The sliding window allows the fewest pairs, which is its purpose. Here $n$ is the number of tokens and $w$ the window width.”

**Ask:** “Under the sliding window, which tokens may `cough` read?”

**Expected:** “`patient`, `reports`, and `cough`.”

**Notebook transition:** Read the `"cough" may read` table under the Part 2 section 5 gallery.

---

## Slide 27 — Each training objective is a mask plus a target (4 min)

**Visible text:**

| | Autoregressive | Masked language model |
|---|---|---|
| Input | `the patient reports cough` | `the patient reports [MASK]` |
| Target | `patient reports cough so` (input shifted by one) | `— — — cough` (original token at each `[MASK]`) |
| Scored positions | every position | about 15% |
| Mask | **causal** | **full** |

- “Mask” has two meanings: the **attention mask** is a table of permissions; the `[MASK]` **token** is a change to the input

**Visual:** Two columns of token boxes. The autoregressive column shows input tokens above gold target tokens shifted by one. The masked column shows a `[MASK]` box in the input and a single gold target beneath it.

**Alt text:** The autoregressive objective pairs each input token with the following token as its target, while the masked objective scores only the position holding the mask token.

**Say:** “For AR the causal mask is required: without it, position $i$ could read position $i+1$, which is its answer. For MLM the attention mask is full, because the clues can sit on either side of the blank. Keep the two meanings of the word mask apart.”

**Ask:** “Why does MLM score only the masked positions?”

**Expected:** “An unmasked position can see its own token, so predicting it teaches nothing.”

**Notebook transition:** Run Part 2 section 6.

---

## Slide 28 — From the notebook: both objectives on one sentence (3 min)

**Visible text:**
- AR: causal mask, 11 of 11 input positions scored
- MLM: full mask, 2 of 12 positions scored
- The `[MASK]` row may read `chest xray` to its **right**
- The AR row for the same position stops at `reports`

**Visual:** From the notebook — `Part-2_Attention_and_Self_Attention.ipynb`, section 6: two mask grids for `the patient reports cough so the nurse orders a chest xray today`, the AR grid lower-triangular with `input → target` row labels and the MLM grid full with the two `[MASK]` rows outlined.

**Alt text:** A triangular mask with eleven scored rows sits beside a full mask in which only two outlined rows are scored.

**Say:** “AR extracts more training signal from each sentence, which is one reason the largest generative models are trained this way. MLM gets something in exchange: the blank may read both directions. AR models generate. MLM models represent, which suits classification, entity recognition, and the embedding models used for retrieval in Part 3.”

**Ask:** “How many training signals does each objective get from this one sentence?”

**Expected:** “Eleven for AR and two for MLM.”

**Notebook transition:** Run Part 2 section 7 and watch both losses fall. The AR loss settles near 0.33, which is correct.

---

## Slide 29 — From an attention row to a prediction (3 min)

**Visible text:**
1. **Attend** — one row of weights over the positions the mask allows
2. **Combine** — weights × value vectors give one output vector
3. **Predict** — head → logits → softmax gives one probability per vocabulary word

$$
\mathbf{z}_i=\mathbf{W}_{\text{head}}\,\mathbf{o}_i+\mathbf{b},
\qquad
\mathbf{p}_i=\operatorname{softmax}(\mathbf{z}_i)
$$

- AR: the output at position `i` predicts token `i + 1`
- MLM: the output at a `[MASK]` predicts the hidden token

**Visual:** Three boxes left to right labelled Attend, Combine, Predict, with notes beneath each: "over the positions the mask allows", "weights × value vectors", "one probability per vocabulary word".

**Alt text:** A three-stage flow turns one row of attention weights into an output vector and then into a probability distribution over the vocabulary.

**Say:** “The prediction head is one linear layer from the output vector $\mathbf{o}_i$ to a score for every word in the vocabulary. Those scores $\mathbf{z}_i$ are the logits, and softmax turns them into probabilities $\mathbf{p}_i$. $\mathbf{W}_{\text{head}}$ and $\mathbf{b}$ are the head's learned weights and bias.”

**Ask:** “What is the size of the logit vector in the toy model?”

**Expected:** “Twenty-six, one score per vocabulary word.”

**Notebook transition:** Run Part 2 section 8.

---

## Slide 30 — From the notebook: one architecture, trained two ways (4 min)

**Visible text:**
- AR after `reports`: 0.25 to each of four symptoms
- AR after `a`: confident `chest`, with attention on `cough`
- MLM at the blank: recovers `cough`, with attention on `chest xray` to the right
- Change the test on the right and the MLM prediction follows every time

**Visual:** From the notebook — `Part-2_Attention_and_Self_Attention.ipynb`, section 8: three rows of paired panels titled "Attention row → output vector → prediction head → probabilities". Each left panel is a bar chart of attention weights with hidden positions in grey; each right panel is a horizontal bar chart of the top five predicted words.

**Alt text:** Three attention bar charts with their prediction charts show an even four-way split for the autoregressive model after reports, a confident chest prediction after a, and a confident cough prediction for the masked model.

**Say:** “Row 1 is the honest answer: nothing to the left says which symptom is coming. Row 2 shows attention carrying the symptom forward eight positions to predict the test. Row 3 shows the MLM model reading to the right. The three rows use the same architecture and the same attention function. The mask decided what each position could read, and the objective decided what it was asked to predict.”

**Ask:** “Is the 0.25 split a sign of an under-trained model?”

**Expected:** “No. It is the best possible answer given the left context.”

**Notebook transition:** Run the "Change the right-hand context" cell in Part 2 section 8.

---

## Slide 31 — Testing the mask: can the future leak? (3 min)

**Visible text:**
- Edit two **late** tokens: `chest xray` → `blood test`
- Measure how far the scores at **earlier** positions move
- AR with causal mask: largest earlier change **0.0**
- MLM with full mask: largest earlier change **12.5**

**Visual:** From the notebook — `Part-2_Attention_and_Self_Attention.ipynb`, section 9: the leak-test table with one row per token, a column of exact zeros for the AR model before the change, and non-zero values for the MLM model. Two large key numbers, 0.0 and 12.5, sit beside it.

**Alt text:** A table of per-position score changes shows zeros for the autoregressive model at every position before the edited tokens and values up to twelve point five for the masked model.

**Say:** “A causal mask guarantees that a prediction at position $i$ cannot depend on any later token, and we can test that directly. The exact zero is what makes generation possible. For the MLM model the change is the intended behaviour. An AR model with a non-zero result here has been reading its own answers, which is the sequence version of the data leakage from Tutorial 1.”

**Ask:** “You train an AR model with the full mask by mistake. What happens to the loss?”

**Expected:** “It collapses toward zero, and the low loss is worthless.”

**Notebook transition:** Run Part 2 section 9.

---

## Slide 32 — From the notebook: the same contrast in pretrained models (3 min)

**Visible text:**
- `distilgpt2` (AR) after `The patient went to the`: `hospital` first at 0.24
- `distilbert` (MLM) with `… to pick up the prescription`: `pharmacy` at 0.80
- `distilbert` with `… to have the surgery`: `hospital` at 0.53
- General-purpose models; their completions are not medical statements

**Visual:** From the notebook — `Part-2_Attention_and_Self_Attention.ipynb`, section 10: three horizontal bar charts of top-five predictions, one for `distilgpt2` and two for `distilbert-base-uncased`.

**Alt text:** Three bar charts show a spread-out next-token distribution for the autoregressive model and two confident, different fill-in predictions for the masked model.

**Say:** “The left context is identical in all three panels. DistilBERT's answer changes with the words to the right of the blank. The section downloads about 600 MB once through `transformers`, and it prints a notice and skips itself when the download is unavailable.”

**Ask:** “What caused DistilBERT's answer to change?”

**Expected:** “Only the tokens to the right of the blank.”

**Notebook transition:** Run the first cell of Part 2 section 10. If it prints `PRETRAINED is None`, use this slide.

---

## Slide 33 — From the notebook: the masks are visible in real attention maps (3 min)

**Visible text:**
- `distilgpt2`: a lower triangle; largest weight above the diagonal is exactly 0.0
- `distilbert`: a filled square; about 38% of its weight sits above the diagonal
- Maps average 12 heads in the first layer
- A map shows where weight went

**Visual:** From the notebook — `Part-2_Attention_and_Self_Attention.ipynb`, section 10: two attention heat maps, "distilgpt2: causal mask → lower triangle" and "distilbert: full mask → every cell in use".

**Alt text:** One heat map is lower-triangular with a bright first column, and the other has weight spread across the whole square.

**Say:** “The triangle is the same guarantee the leak test confirmed for the toy model. The bright first column is a known habit of GPT-style models: a head with nothing specific to read parks its weight on the first token. Individual heads look quite different from one another, and the map leaves open why the model produced its answer.”

**Ask:** “Which feature of the left-hand map is the causal mask?”

**Expected:** “The empty upper triangle.”

**Notebook transition:** Run the attention-map cell, then work Part 2 section 11 and the learner check.

---

## Slide 34 — Beyond the notebook: attention is blind to word order (2 min)

**Visible text:**
- Same tokens, opposite meaning: `result normal not abnormal` / `result abnormal not normal`
- Shuffle the input and the outputs are simply shuffled
- Position has to be **added to the input**, before Q, K, and V
- Output change after a shuffle: 0.00 with token content only, 0.89 with positional encoding

**Visual:** Two token rows with `normal` and `abnormal` swapped, beside a diagram in which `tokens X` and `position` are added and then feed the Q, K, V projections.

**Alt text:** Two sentences with the same words in different order illustrate order-blindness, and a diagram adds a position signal to token vectors before the attention projections.

**Say:** “This material is no longer in the notebook. The toy model in Part 2 section 7 adds a position embedding, `self.pos`, for this reason. The worked shuffle test lives in `archive/Attention_full_walkthrough_v1.ipynb`.”

**Ask:** “Where must position information be added?”

**Expected:** “To the input `X`, before the Q/K/V projections.”

**Notebook transition:** Beyond the notebook. Point at `self.pos` in the Part 2 section 7 model.

---

## Slide 35 — What the small attention notebook leaves out (2 min)

**Visible text:**
- Multiple heads and multiple layers
- Residual connections and normalization at scale
- Realistic vocabulary, corpus, and tokenizer
- Dense attention uses `O(n²)` score entries

$$
\mathbf{Q}\mathbf{K}^{\mathsf T}\in\mathbb{R}^{n\times n}
\quad\Rightarrow\quad n^2\text{ pairwise scores}
$$

**Visual:** Full Transformer block outline with only scaled dot-product attention highlighted.

**Alt text:** A Transformer block contains attention plus several omitted components; only one attention operation is highlighted.

**Say:** “For $n$ sequence positions, every query compares with every key, producing an $n$ by $n$ matrix. Doubling $n$ makes four times as many score entries, which is why sliding-window masks exist. The toy model has one head, one layer, 36 template sentences, and a 26-word vocabulary. It demonstrates a mechanism and nothing about medicine.”

**Ask:** “Why can long sequences make dense attention expensive?”

**Expected:** “The score matrix has a pairwise entry for query and key positions, growing quadratically.”

**Notebook transition:** Read the Part 2 summary and limits, then close Part 2.

---

## Slide 36 — Slides only — no notebook: how the next word is chosen (1 min)

**Visible text:**
- “Why does the same model, asked the same question, give a different answer each time?”
- The next three slides have no notebook
- Archived demonstration: `archive/Decoding_Temperature_TopK_TopP.ipynb`

**Visual:** Section divider labelled "Slides only · Decoding".

**Alt text:** A divider slide marks the decoding section as taught from slides with no accompanying notebook.

**Say:** “Part 2 ended with probabilities over the vocabulary. Decoding is the rule that turns those probabilities into chosen words. We teach it from the slides.”

**Ask:** “Where did we last see a probability for every vocabulary word?”

**Expected:** “At the prediction head in Part 2 section 8.”

**Notebook transition:** Slides only — no notebook.

---

## Slide 37 — Slides only: decoding is the step after the model finishes (3 min)

**Visible text:**
- The model outputs **logits**: one score per vocabulary token
- Softmax turns scores into probabilities
- **Decoding** picks one token — and it is your setting, not the model's

**Visual:** A bar chart of eight candidate next tokens, first as raw logits and then as probabilities after softmax.

**Alt text:** Eight unbounded scores become eight probabilities that sum to one, preserving their order.

**Say:** "Everything on this slide happens after the model's computation is complete. The scores never change. Only the rule for reading them changes."

$$p_i=\frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$$

Here $z_i$ is the logit for token $i$, $T$ is the temperature, and $p_i$ is the resulting probability.

**Ask:** "Does raising the temperature change what the model computed?"

**Expected:** "No. It rescales the scores before the softmax."

**Notebook transition:** Slides only — no notebook. The archived notebook's sections 1 and 2 show the same chart.

---

## Slide 38 — Slides only: temperature, top-k, and top-p (4 min)

**Visible text:**
- **Temperature** < 1 sharpens, > 1 flattens
- **Top-k** keeps a fixed number of candidates
- **Top-p** keeps the smallest set reaching probability $p$ — so it adapts

**Visual:** Three panels of the same distribution: reshaped by temperature, truncated by top-k, truncated by top-p, with discarded bars greyed to zero.

**Alt text:** Temperature changes the shape of the distribution while top-k and top-p set some probabilities to exactly zero before renormalising.

**Say:** "Top-k and top-p do not make unlikely tokens less likely. They make them impossible, then renormalise what is left."

**Ask:** "Why does top-p keep a different number of tokens in different situations?"

**Expected:** "It keeps whatever is needed to reach the cumulative threshold, so a confident distribution needs fewer."

**Notebook transition:** Slides only — no notebook. The archived notebook's sections 3 to 6 show the same panels.

---

## Slide 39 — Slides only: the same request gives different answers (3 min)

**Visible text:**
- Any temperature above 0 → repeated identical requests differ
- `temperature=0` gives **repeatability, not accuracy**
- For audit: pin the setting *and* record the model version

**Visual:** A table of five runs of one identical prompt: one row under greedy decoding (all identical) and one under sampling (all different).

**Alt text:** Greedy decoding returns one output five times while sampling returns five different outputs from the same request.

**Say:** "If you summarise the same discharge note twice and get two summaries, that is decoding. It is not the model changing its mind, and it is not evidence that either summary is right."

**Ask:** "Can any decoding setting stop a model stating a fact that is not in its input?"

**Expected:** "No. That needs supplied evidence and human checking — Part 3."

**Notebook transition:** Slides only — no notebook. The five-run table comes from section 8 of `archive/Decoding_Temperature_TopK_TopP.ipynb`.

---

## Slide 40 — Transition: model parameters are not a document database (1 min)

**Visible text:** “How can a generated response use a small, inspectable external corpus?”

**Visual:** A generator beside—not inside—a document store.

**Alt text:** A language model and a separate document store are connected by retrieval.

**Say:** “RAG does not paste documents into model weights. It retrieves selected text at request time.”

**Ask:** “Does retrieval update the generator's fitted parameters?”

**Expected:** “No. It supplies external context at request time.”

**Notebook transition:** Open Part 3.

---

## Slide 41 — Four RAG boundaries (4 min)

**Visible text:**
1. Parametric model patterns
2. External document corpus
3. Retrieval ranking
4. Generation conditioned on selected context

**Visual:** Four labelled stages with boundaries and arrows.

**Alt text:** The RAG workflow separates model parameters, external documents, retrieval, and response generation.

**Say:** “Walk through one request slowly. Documents are chunked and embedded before the question arrives. At request time, the question is embedded, chunks are ranked, selected text enters a prompt, and the generator writes a response. Keeping these stages separate helps diagnose failure: retrieval can miss evidence, while generation can misuse correctly retrieved evidence.”

**Ask:** “Does adding a document to Chroma retrain the generator?”

**Expected:** “No.”

**Notebook transition:** Run Part 3 setup. Confirm the printed `run_mode`. `replay` needs no key and no network; `live` makes paid calls. In both cases the key value and `.env` path stay hidden.

---

## Slide 42 — Chunking defines what can be retrieved (3 min)

**Visible text:**
- Source document → chunks + metadata + stable IDs
- Too small: context can be fragmented
- Too large: irrelevant text consumes context
- Tutorial: one chunk per level-2 section, nine chunks total

$$
D\longrightarrow\{c_1,c_2,\ldots,c_9\},
\qquad c_j=(\text{ID}_j,\text{text}_j,\text{metadata}_j)
$$

**Visual:** One Markdown document split at `##` headings into three labelled records.

**Alt text:** A document is split by section headings into chunks carrying source and section metadata.

**Say:** “Treat the source document $D$ as being transformed into retrieval units $c_j$. Each unit needs text, a stable ID, and metadata. Chunking is a design decision: too small can separate facts that belong together; too large can dilute relevance and consume prompt space. Stable IDs let us trace exactly what entered the request.”

**Ask:** “What changes if one answer requires two separate sections?”

**Expected:** “Retrieval must return both chunks, or the context will be incomplete.”

**Notebook transition:** Run the Part 3 source-manifest and chunking cells (sections 2 and 3).

---

## Slide 43 — From the notebook: nine chunks, one per heading (2 min)

**Visible text:**
- Three fictional sources → nine chunks
- Each chunk has a **stable ID** such as `results_policy::000`
- Every retrieval score, citation, and audit later in the notebook refers to these nine IDs

**Visual:** From the notebook — `Part-3_RAG_With_and_Without_Retrieval.ipynb`, section 3: the chunk table (ID, source, section, word count, preview) and the horizontal bar chart of words per chunk, coloured by source document.

**Alt text:** Nine horizontal bars, one per chunk ID and coloured by the three source documents, show chunk lengths between about 24 and 34 words.

**Say:** “Stable IDs are what make the rest of the notebook checkable. When an answer cites `[access_guide::000]`, a reader can open exactly that text.”

**Ask:** “How many chunks can a top-4 retrieval return from this corpus for any question?”

**Expected:** “Always four of the nine, whether or not they are relevant.”

**Notebook transition:** Run Part 3 section 3, then compare with the fixed-window chunks in section 3b.

---

## Slide 44 — Embeddings support similarity search (3 min)

**Visible text:**
- Embed each chunk once
- Embed the question using the same model
- Compare vectors with cosine distance
- Retrieve top `k`; ranking is not proof of relevance

$$
d_{\cos}(\mathbf{q},\mathbf{c}_j)
=1-\frac{\mathbf{q}^{\mathsf T}\mathbf{c}_j}
{\lVert\mathbf{q}\rVert_2\lVert\mathbf{c}_j\rVert_2}
$$

$$
R_k(q)=\left\{\text{the }k\text{ chunk IDs with the smallest }d_{\cos}(\mathbf{q},\mathbf{c}_j)\right\}
$$

**Visual:** Question vector near some chunk vectors in a schematic embedding space.

**Alt text:** A question vector is compared with document-chunk vectors, and nearest chunks are selected.

**Say:** “The question becomes vector $\mathbf{q}$ and chunk $j$ becomes vector $\mathbf{c}_j$. Smaller cosine distance means more similar direction, so top-$k$ selects the $k$ smallest distances. In the notebook display, similarity is $1-d_{\cos}$ so larger is better. Neither value is a probability. Never mix vectors from different embedding models in one index.”

**Ask:** “Is the nearest chunk guaranteed to answer the question?”

**Expected:** “No.”

**Notebook transition:** Run the retrieval cell. Read rank, chunk ID, cosine similarity, and expected-evidence star from the retrieval table.

---

## Slide 45 — From the notebook: every question against every chunk (3 min)

**Visible text:**
- Four questions × nine chunks, scored by word overlap
- Top-k means **highest ranked**
- Q3, the unanswerable question, scores 0.000 everywhere and top-k still returns four chunks
- Relevance is a separate check

**Visual:** From the notebook — `Part-3_RAG_With_and_Without_Retrieval.ipynb`, section 7: annotated heat map of lexical scores with one row per question (Q1 to Q4) and one column per chunk ID; stars mark the expected evidence.

**Alt text:** A heat map of question-by-chunk overlap scores shows starred peaks for the answerable questions and a row of zeros for the unanswerable one.

**Say:** “A ranking always has a first place. The parking-fee question has no answer in the corpus, every one of its scores is zero, and retrieval still returns four chunks. Q4 is the harder case: its scores are as high as Q1's, and the evidence is relevant and incomplete. Keep this picture in mind for the chat app at the end of Part 3.”

**Ask:** “Does a top-ranked chunk show that an answer exists?”

**Expected:** “No. It shows only which chunk ranked highest.”

**Notebook transition:** Run Part 3 sections 6 and 7.

---

## Slide 46 — A terminology connects patient words to document words (4 min)

**Visible text:**
- Patients say `bloodwork`; policies say `laboratory result`
- A **terminology** states the equivalence explicitly: code, preferred term, synonyms
- **Query expansion** adds the document's vocabulary before retrieving

**Visual:** A patient question flowing into a concept card (`NC-00412`, preferred term, synonym list) and out as an expanded query, with retrieval scores rising for the results-policy chunks.

**Alt text:** A concept record maps a patient phrase to the corpus vocabulary, and the expanded query raises the retrieval score of the relevant chunks.

**Say:** "The fixture here uses an invented code system. A production system would use SNOMED CT, LOINC, or ICD-10 — which adds licensing, versioning, and mapping maintenance that this fixture does not model."

**Ask:** "What did the terminology supply that the embedding model did not?"

**Expected:** "An explicit, inspectable, maintainable statement of equivalence that a human can version and correct."

**Notebook transition:** Run Part 3 section 8 and compare scores before and after expansion.

---

## Slide 47 — A knowledge graph adds relationships and provenance (4 min)

**Visible text:**
- Terminology: which words mean the same thing
- Graph: how concepts **relate** — subject → predicate → object
- Every edge carries the chunk it came from

**Visual:** A small graph with `urgent laboratory result --communicated_by--> designated clinician`, each edge labelled with its evidence chunk ID, and a one-hop boundary drawn around the results-policy nodes.

**Alt text:** Graph edges connect concepts to roles and values, and each edge is annotated with the corpus chunk that justifies it.

**Say:** "The value here is provenance. A reviewer can ask why a passage was shown and get a traceable relationship, not a similarity number. That is not the same as the graph being correct — it is hand-built and inherits its authors' errors."

**Ask:** "Why is a two-hop walk not automatically better than one hop?"

**Expected:** "Each hop reaches material further from the question. Two hops pulled in the registration-code chunk, which is connected but irrelevant."

**Notebook transition:** Run Part 3 section 9 and compare one hop with two.

---

## Slide 48 — The local vector database lifecycle (3 min)

**Visible text:**
- Source: `data/rag_corpus/`
- Generated chunks/index: `artifacts/rag/`
- Reuse only when fingerprint + model + chunking match
- Local persistence ≠ access control, backup, or governance

**Visual:** Directory tree with versioned source and gitignored artifacts.

**Alt text:** The tutorial directory separates fictional source documents from generated chunks and the persistent Chroma database.

**Say:** “The database lives locally so students can inspect its lifecycle, but the embeddings were produced through a paid network request. A fingerprint summarizes the chunk identities, content hashes, chunking method, and embedding model. If any of these change, rebuild rather than mixing incompatible vectors.”

**Ask:** “What should happen after changing a source policy?”

**Expected:** “The fingerprint changes and the index must be rebuilt.”

**Notebook transition:** Inspect `index_metadata.json`, then confirm that the collection contains exactly nine records (live mode only).

---

## Slide 49 — Same question, two generation conditions (3 min)

**Visible text:**
- Direct: question, no tutorial corpus context
- RAG: question + retrieved labelled chunks
- Same generation model
- Compare evidence support, not writing style alone

$$
\begin{aligned}
\text{Direct input} &= I_{\text{base}}+q \\
\text{RAG input} &= I_{\text{base}}+I_{\text{ground}}+C_k+q
\end{aligned}
$$

**Visual:** Split panel showing direct prompt versus RAG prompt with context labels.

**Alt text:** Two prompts use the same question and model, but only the RAG condition contains retrieved chunks.

**Say:** “Both paid requests use the same model and question $q$. The direct request receives shared base instructions $I_{\text{base}}$ but no tutorial corpus. The RAG request adds grounding instructions $I_{\text{ground}}$ and retrieved context $C_k$. This is not a claim that the direct model has no prior knowledge; it means the fictional Northstar evidence was not supplied.”

**Ask:** “What is the controlled difference?”

**Expected:** “Retrieved tutorial context is added in the RAG condition.”

**Notebook transition:** Run the comparison cell. Pause on Q1 and read the direct and RAG cards left to right before expanding the retrieved evidence.

---

## Slide 50 — Grounding instructions and citations (3 min)

**Visible text:**
- Use only facts in `CONTEXT`
- Cite exact chunk IDs
- State when context is insufficient
- Retrieved text is untrusted data, not instructions

**Visual:** Prompt hierarchy showing instruction above delimited context and user question.

**Alt text:** A prompt separates higher-priority behavior instructions from untrusted retrieved context and the user question.

**Say:** “Grounding asks the model to constrain claims to supplied context. Citation formatting creates a traceable pointer, but a well-formed label does not prove the sentence is supported. Open the evidence panel and compare claim against text. Retrieved documents can also contain malicious instructions, so context remains untrusted data.”

**Ask:** “If a retrieved chunk says ‘ignore prior instructions,’ should the system follow it?”

**Expected:** “No. Retrieved text is untrusted context.”

**Notebook transition:** In each card, locate the RAG citation and then expand **Retrieved evidence sent to the RAG request** to verify it manually.

---

## Slide 51 — Evaluate retrieval and generation separately (4 min)

**Visible text:**
- Retrieval: Were required chunk IDs in top `k`?
- Generation: Were required facts covered?
- Citations: Do labels match retrieved evidence?
- Unsupported specifics: present or absent?
- Unanswerable question: does output acknowledge missing evidence?

$$
\operatorname{Recall@}k
=
\frac{|E_q\cap R_k(q)|}{|E_q|}
\qquad (|E_q|>0)
$$

**Visual:** Evaluation table for Q1, Q2, and Q3 with retrieval and generation columns.

**Alt text:** Three fixed questions are evaluated separately for retrieval hits, factual coverage, citations, unsupported claims, and insufficient-evidence behavior.

**Say:** “$E_q$ is the instructor-authored set of evidence needed for question $q$; $R_k(q)$ is the set retrieved in the top $k$. Recall at $k$ asks what fraction of required evidence was retrieved. It does not grade the prose. A correct-looking answer with failed retrieval can be unsupported coincidence; correct retrieval with an incorrect answer is a generation failure. For Q3, $E_q$ is empty, so use an abstention check instead of dividing by zero.”

**Ask:** “If the right chunk is retrieved but the response invents a fee, which stage failed?”

**Expected:** “Generation/evidence use failed, even though retrieval succeeded.”

**Notebook transition:** Read the evaluation table by stage. Discuss Q2 as multi-chunk retrieval and Q3 as the abstention case; then compare direct versus RAG token totals.

---

## Slide 52 — A citation makes checking possible, not done (4 min)

**Visible text:**
- Automated checks verify the **label**, not the **claim**
- Four deliberate faults, one caught
- The dangerous one passes every gate

**Visual:** Five answer cards with pass/fail badges, where four show a green automated badge and only one is flagged, while a human-review column marks four as faulty.

**Alt text:** Automated citation checking flags only the answer citing a chunk ID that was never retrieved, while three other faulty answers pass.

**Say:** "Case B cites a real retrieved chunk, states one true fact from it, and appends an invented fee. Every automated gate passes. Only a person reading the cited chunk catches it."

**Ask:** "A correct chunk was retrieved but the answer invents a fee. Which stage do you investigate?"

**Expected:** "Generation and evidence use. Record retrieval as having succeeded."

**Notebook transition:** Run Part 3 section 13 and work the audit table before revealing the key.

---

## Slide 53 — From notebook to chat interface: how Streamlit works (3 min)

**Visible text:**
- **Streamlit** turns a Python script into a web page
- The script **reruns top to bottom** on every interaction
- `st.session_state` keeps the conversation; `@st.cache_data` keeps the corpus and embeddings
- `st.chat_input` and `st.chat_message` draw the chat box
- `retrieval only` (default, no key, no cost) and `generate` (key, paid)

**Visual:** A vertical flow of five boxes: notebook `%%writefile rag_chat_app.py` → `streamlit run` on `localhost` → `retrieve()`, the same functions as sections 6–8 → `generate()`, only with an API key → reply + evidence panel.

**Alt text:** A five-step flow runs from a notebook cell that writes the app file, through a local Streamlit server and the retrieval and generation functions, to a reply with an evidence panel.

**Say:** “The interface adds no intelligence. It gives someone else a way to use what you built. The server is bound to `localhost`, so only your machine can reach it. Making it reachable by other people is a deployment decision and it is outside this tutorial. Upload only fictional or public text, and never enter PHI.”

**Ask:** “The script reruns on every interaction. Where does the conversation survive?”

**Expected:** “In `st.session_state`.”

**Notebook transition:** Run the Part 3 section 14 `%%writefile` cell and find where `retrieve` and `generate` are called.

---

## Slide 54 — From the notebook: the chat app, asked a patient's question (4 min)

**Visible text:**
- **You:** Who tells me about my bloodwork if something is wrong?
- **App:** No model was called. The best-matching passage is **[results_policy::000]** (Routine results).
- Terminology added: `laboratory result results`

| # | chunk | score |
|---:|---|---:|
| 1 | `results_policy::000` | 0.125 |
| 2 | `results_policy::002` | 0.107 |
| 3 | `results_policy::001` | 0.038 |
| 4 | `access_guide::000` | 0.000 |

**Visual:** From the notebook — `Part-3_RAG_With_and_Without_Retrieval.ipynb`, section 14: the `AppTest` cell output in retrieval-only mode (no key, no cost), shown as a chat exchange beside the evidence-panel ranking. The app runs on `localhost:8501`.

**Alt text:** A chat exchange in retrieval-only mode names results policy chunk zero as the best match, beside a four-row evidence table whose last row scored zero.

**Say:** “The ranking matches section 8, because the app calls the same functions. Rank 4 scored 0.000 and was still shown. A chat box looks finished and authoritative, and nothing on the screen says that the corpus has nine chunks, that top-k always returns k passages, or that a cited answer can still be wrong. Put the limits on the screen.”

**Ask:** “Ask the app Q3, the unanswerable question. What should it say, and what does it say?”

**Expected:** “It should decline. It names a best-matching passage anyway, so `answer()` needs a score threshold.”

**Notebook transition:** Run the Part 3 section 14 test and launch cells, open the printed address, then set `STOP_CHAT_APP = True` and run the stop cell. Finish with the section 15 learner checks.

---

## Slide 55 — What this tutorial supports—and what it does not (4 min)

**Visible text:**

**Supports**
- Inspecting vectors and computed weights
- Tracing retrieved chunks into a prompt
- Comparing direct and RAG conditions
- Putting a retrieval prototype behind a local chat page

**Does not support**
- Clinical correctness or safety
- Causal interpretation of attention
- Complete retrieval or faithful citations
- Deployment beyond `localhost`, privacy compliance, or fairness

**Visual:** Two-column supports/does-not-support table plus exit prompt.

**Alt text:** A boundary table separates demonstrated mechanisms from claims the tutorial cannot establish.

**Say:** “A transparent small example is useful because we can trace each operation. That transparency does not validate a health application.”

**Ask:** “Exit prompt: name one numerical mechanism, one evaluation check, and one governance safeguard from today.”

**Expected:** Examples: cosine similarity or the causal mask; expected chunk in top-k or the leak test; never send patient data or keys, and keep the chat app on `localhost`.

**Notebook transition:** Show completion checklists and supporting-material links.

## Deck production requirements

- Keep formulas in this source as editable LaTeX. Render the double-dollar display blocks in Markdown preview; screenshots of the rendered equations may be placed in presentation software when native equation editing is impractical.
- Use the exact fixture values/IDs where a slide shows an output.
- Keep visible text under approximately 45 words per slide excluding formulas and labels.
- Put longer explanations in speaker notes (`Say`).
- Add source footer links on theory slides: Gensim/scikit-learn for the Part 1 slides, Vaswani/PyTorch and Hugging Face `transformers` for the Part 2 slides, OpenAI/Chroma/Streamlit for the Part 3 slides.
- On every **From the notebook** slide, keep the notebook name and section in the footer, matching the **Visual** field. Export figures from the executed notebooks; do not redraw them.
- Label every decoding slide "Slides only — no notebook".
- Provide text descriptions below any figure where spatial position or color carries meaning.
- Number notebook cells during implementation, then replace section-only transitions above with exact cell numbers in final `slides.md`.
