# Glossary and Quick Reference — Tutorial 2, Parts 0 to 3

## Core terms

| Term | Plain-language definition | First used |
|---|---|---|
| Token | The unit of text a model actually reads. Often part of a word. | Part 0 |
| Vocabulary | The fixed list of tokens a model knows. | Part 0 |
| Tokenizer | The program that converts text into vocabulary entries. | Part 0 |
| Subword | A token smaller than a word, so nothing is ever unknown. | Part 0 |
| Byte pair encoding (BPE) | Repeatedly merge the most frequent adjacent symbol pair to learn a vocabulary. | Part 0 |
| Context window | The maximum number of tokens a model accepts in one request. | Part 0 |
| Embedding | A dense numerical vector representing a token or a piece of text. | Part 1 |
| Word2Vec | Learns word vectors from nearby-token prediction. | Part 1 |
| CBOW / skip-gram | Predict centre from context / predict context from centre. | Part 1 |
| Cosine similarity | Normalised dot product comparing vector direction. | Part 1 |
| t-SNE | Nonlinear 2-D visualisation. Axes carry no meaning. | Part 1 |
| Query, key, value | Match request, matchable representation, and the information combined. | Part 2 §2 |
| Attention weight | A post-softmax number describing how much a value row contributes. | Part 2 §1 |
| Softmax | Converts scores into probabilities that sum to one. | Part 2 §2; slides |
| Self-attention | Q, K, and V all come from projections of one sequence. | Part 2 §4 |
| Cross-attention | Queries come from one sequence, keys and values from another. | Part 2 §4 |
| Attention mask | A Boolean table with one row per query and one column per key, saying which positions each token may read. Here `True = allowed`. | Part 2 §2, §5 |
| Causal mask | Each position reads itself and earlier positions. Blocked scores become `−∞` before the softmax, so their weights are exactly zero. | Part 2 §4 |
| Bidirectional (full) mask | Every token reads every token. Used by BERT-style encoders and embedding models. | Part 2 §5 |
| Padding mask | No token reads `[PAD]` filler. It removes a column; the `[PAD]` row is computed and ignored in the loss. | Part 2 §5 |
| Prefix mask | The prompt is read both ways and the continuation is causal. | Part 2 §5 |
| Sliding-window attention | Each position reads only the last `w` tokens, cutting the cost from about `n²` to about `n × w`. | Part 2 §5 |
| Autoregressive (AR) | Training objective: predict the next token at every position, under a causal mask. Produces models that generate. | Part 2 §6 |
| Masked language modelling (MLM) | Training objective: hide about 15% of tokens and predict the originals, under a full mask. Produces models that represent a whole passage. | Part 2 §6 |
| Attention mask vs `[MASK]` token | The attention mask is the Boolean table of allowed reads. The `[MASK]` token is a change to the input that hides a word. MLM uses a `[MASK]` token together with a full attention mask. | Part 2 §6 |
| Prediction head | A linear layer that turns one position's output vector into one score per vocabulary word. | Part 2 §7 |
| Logits | The raw per-token scores a model produces before softmax and before any choice is made. | Part 2 §8; slides |
| Leak test | Change late tokens and measure how much the scores at earlier positions move. A causal model must show exactly `0.0`. | Part 2 §9 |
| Pretrained model | A model whose weights were already fitted on a large corpus. `distilgpt2` is AR and `distilbert-base-uncased` is MLM. | Part 2 §10 |
| Positional encoding | A per-position signature added to the input, because attention is order-blind. | Slides ("beyond the notebook"); `archive/Attention_full_walkthrough_v1.ipynb` |
| Decoding | The rule that turns scores into one chosen token. | Slides only |
| Temperature | Divides logits before softmax: below 1 sharpens, above 1 flattens. | Slides only |
| Greedy decoding | Always take the highest-probability token. Deterministic. | Slides only |
| Top-k | Keep the k highest-probability tokens, then renormalise. | Slides only |
| Top-p (nucleus) | Keep the smallest set whose probability reaches p, then renormalise. | Slides only |
| Chunk | A retrievable text unit, with stable ID and metadata. | Part 3 |
| Chunking strategy | The rule that decides chunk boundaries (headings, fixed windows, ...). | Part 3 |
| Vector store | A database holding vectors, IDs, documents, and metadata. | Part 3 |
| Top-k retrieval | Return the k nearest candidates. Always returns k, relevant or not. | Part 3 |
| RAG | Retrieve external chunks, then condition generation on them. Not retraining. | Part 3 |
| Grounding | Restricting answer claims to supplied evidence. | Part 3 |
| Parametric knowledge | Patterns in fitted model weights, distinct from the tutorial corpus. | Part 3 |
| Terminology | An explicit list of concepts, preferred terms, and synonyms. | Part 3 |
| Query expansion | Adding terms to a query before retrieving. | Part 3 |
| Knowledge graph | Subject–predicate–object statements about how concepts relate. | Part 3 |
| Provenance | The record of which source a claim came from. | Part 3 |
| Prompt injection | Untrusted text attempting to change model instructions. | Part 3 |
| Corpus fingerprint | A hash identifying the source, chunking, and model state of an index. | Part 3 |
| Streamlit | A Python library that turns a plain script into a web page. The script reruns from top to bottom on every interaction. | Part 3 §14 |
| Session state | `st.session_state`: the place a Streamlit app keeps anything that must survive a rerun, such as the conversation. | Part 3 §14 |
| `@st.cache_data` | Keeps the result of an expensive function, such as loading the corpus or embedding the chunks, so a rerun does not repeat it. | Part 3 §14 |
| `AppTest` | Streamlit's test harness. It runs the app in memory, types into the chat box, and returns what the page would show. | Part 3 §14 |
| localhost | The network name for your own computer. A server bound to `localhost` can be reached only from that machine. | Part 3 §14 |

The decoding notebook is archived at `archive/Decoding_Temperature_TopK_TopP.ipynb`.

## Formula reference

**Attention** (Part 2):

`S = QKᵀ / √d_k` → `S̃ = mask(S)`, disallowed pairs set to `−∞` → `A = softmax(S̃, dim=-1)` → `O = AV`

For `L` query positions, `S` key positions, key width `d_k`, value width `d_v`:
`Q=(L,d_k)`, `K=(S,d_k)`, `V=(S,d_v)`, weights `(L,S)`, output `(L,d_v)`.
**Rows are queries; columns are keys.** In this tutorial's masks, `True = allowed`.

**From an attention row to a prediction** (Part 2 §8):

`logits_i = head(o_i)` → `p_i = softmax(logits_i)`. For AR, `p_i` predicts token `i + 1`. For MLM, `p_i` at a `[MASK]` position predicts the hidden token.

**Softmax with temperature** (slides only):

`p_i = exp(logit_i / T) / Σ_j exp(logit_j / T)`

**Jaccard overlap** (Part 3 lexical baseline):

`score = |shared unique terms| / |all unique terms in query or chunk|`

**Cosine similarity from distance** (Part 3): `similarity = 1 − cosine distance`.

## Mask patterns (Part 2 §5)

| Pattern | Rule | Where it is used |
|---|---|---|
| Bidirectional (full) | Every token reads every token. | BERT-style encoders, embedding models |
| Causal | Read yourself and earlier tokens. | GPT-style decoders, all chat models |
| Padding | Nobody reads `[PAD]` filler. | Batches of unequal-length texts |
| Causal + padding | Both rules at once. | Decoder training in batches |
| Prefix | The prompt is read both ways; the continuation is causal. | Encoder–decoder style models such as T5 |
| Sliding window | Read only the last `w` tokens. | Long-context models, to cut the `n²` cost |

## Two training objectives (Part 2 §6–9)

| | Autoregressive (AR) | Masked language modelling (MLM) |
|---|---|---|
| Attention mask | Causal | Full |
| Input | The sentence | The sentence with about 15% of tokens replaced by `[MASK]` |
| Target | The next token, at every position | The original token, at masked positions only |
| Scored positions in the 12-token example | 11 of 11 | 2 of 12 |
| Toy result in §8 | 0.25 to each of four symptoms after `reports` | Recovers `cough` from `chest xray` on the right |
| Leak test in §9 | Exactly `0.0` | `12.5` |
| Good at | Generating text | Representing a passage: classification, entity recognition, retrieval embeddings |
| Pretrained example in §10 | `distilgpt2` | `distilbert-base-uncased` |

## Decoding settings: what to reach for

Decoding is taught from the slides only.

| Task | Starting point | Why |
|---|---|---|
| Extracting a value from a document | `temperature=0` | One correct answer; you want the same result each run. |
| Grounded question answering | `0`–`0.3` | Wording should follow the evidence. |
| Drafting text a human will edit | `0.7`, `top_p=0.9` | Variation is useful when a person reviews every output. |
| Anything audited | `temperature=0`, record model name **and version** | Reproducibility is a governance requirement, not a quality claim. |

`temperature=0` gives repeatability, not accuracy.

## Word2Vec settings (Part 1)

| Setting | Value | Effect |
|---|---:|---|
| `vector_size` | 50 | Coordinates per word vector. |
| `window` | 3 | Local context radius. |
| `min_count` | 1 | Keep every fixture token. |
| `sg` | 1 | Use skip-gram. |
| `workers`, `seed` | 1, 42 | Classroom reproducibility. |
| `negative` | 10 | Noise words sampled per positive example. |
| `sample` | `1e-3` | Downsample very frequent words. |

## Stages: what each one does and does not do

| Step | Input | Output | Does **not** do |
|---|---|---|---|
| Tokenization | Text | Token IDs | Assign meaning |
| Embedding | Text or token | Vector | Generate an answer |
| Attention | Q, K, V, mask | Weighted combination | Explain causally |
| Prediction head | One output vector | Logits over the vocabulary | Choose a token |
| Retrieval | Query vector + index | Ranked chunks | Verify relevance or truth |
| Decoding | Logits | One chosen token | Supply missing evidence |
| RAG | Question + chunks | Conditioned response | Retrain model weights |
| Chat interface | A typed question | A reply and an evidence panel | Add intelligence or show the system's limits |

## Part 3 evaluation cases

| Case | Question type | What it tests |
|---|---|---|
| Q1 | single-chunk | Basic retrieval and citation |
| Q2 | multi-chunk | Combining two passages |
| Q3 | unanswerable | Abstention when nothing is relevant |
| Q4 | **partial evidence** | Answering the supported part and naming the gap |

## Citation audit: what automated checking catches

| Fault | Caught automatically? |
|---|---|
| Cited an ID that was never retrieved | **yes** |
| Cited a real chunk but invented an extra fact | no |
| Cited the wrong retrieved chunk for a true claim | no |
| Added confident policy advice with no evidence | no |

One of four. The rest need a person reading the cited text.

## Chat app modes (Part 3 §14)

| Mode | Retrieval | Answer | Key | Cost |
|---|---|---|---|---|
| `retrieval only` (default) | Word overlap, optional terminology expansion | None; shows the evidence passages | No | None |
| `generate` | Embedding similarity | Grounded answer with chunk citations | `OPENAI_API_KEY` in a `.env` file | Paid |

Launch from the notebook, or with `streamlit run rag_chat_app.py --server.address localhost`. Stop it with the `STOP_CHAT_APP = True` cell. Set `HLTH667M_SKIP_APP_LAUNCH=1` to skip the launch cell. Keep the app on `localhost` and never enter PHI.

## Artifact map

Fictional sources live in `data/rag_corpus/`. Terminology and graph fixtures live in `data/terminology/`. The recorded run and audit fixture live in `data/recorded_runs/`. The chat app is `rag_chat_app.py`. Earlier notebooks live in `archive/`, and the notebook-maintenance scripts live in `tools/`. Generated chunks, index metadata, and Chroma persistence live under `artifacts/` and are ignored by Git. Never store secrets or real health data in any of them.
