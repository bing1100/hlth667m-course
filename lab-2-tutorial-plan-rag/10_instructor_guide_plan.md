# 10. Instructor Guide Plan

## Before class

- Install and verify the planned requirements in a clean environment.
- Install `gensim` and `chromadb`; the current environment does not include them.
- Confirm Torch attention comparison works on CPU.
- Confirm `OPENAI_API_KEY` is available only in the instructor environment; do not show `.env` contents.
- Run the RAG notebook once offline and once live with the fixed question set.
- Reset and rebuild the local Chroma collection; record collection count and index metadata.
- Prepare an executed backup of all notebooks.
- Check that source documents are fictional/public and contain no patient information.
- Record model names, approximate API usage, and expected output patterns without promising exact generated wording.
- Confirm the 24-slide deck has exact notebook cell/section transitions and rehearse the interleaved slide/code handoffs.
- Set a hard classroom API-call ceiling: one chunk-embedding batch, three question embeddings, and six generation calls unless intentionally reduced.

## 90-minute run-of-show

| Time | Activity | Check |
|---:|---|---|
| 0–5 | State arc and safety boundary. | Students name the three transformations. |
| 5–12 | Word2Vec corpus and context. | Students identify a target/context pair. |
| 12–20 | Train and inspect vectors. | Students explain one nearest-neighbor result as corpus-dependent. |
| 20–28 | t-SNE map and limitations. | Students state one invalid map interpretation. |
| 28–35 | Attention formula and hand-checkable example. | Students identify score, softmax, and weighted sum. |
| 35–45 | Attention map and mask. | Students read one row and identify blocked future positions. |
| 45–55 | Self-attention and PyTorch comparison. | Students distinguish self- from cross-attention. |
| 55–62 | RAG architecture and local database. | Students name source, chunk, vector, retrieval, and generation stages. |
| 62–70 | Direct response without retrieval. | Students identify what evidence is absent. |
| 70–80 | Embed, index, retrieve, and inspect chunks. | Students trace source IDs and top-k results. |
| 80–87 | Generate with retrieval and compare. | Students mark supported, unsupported, and missing-evidence claims. |
| 87–90 | Exit reflection. | Students write one limitation and one security safeguard. |

Slides and notebooks are interleaved inside these intervals. The per-slide times in Document 09 are pacing maxima for explanation/checks and overlap the associated activity windows; they must not be added to the notebook durations.

## Expected teaching points

- Word2Vec uses local context; it does not use a health ontology.
- t-SNE is sensitive to settings and should not be treated as a validated map.
- Attention weights are computed quantities, not automatic explanations.
- Self-attention uses one sequence for Q/K/V; cross-attention can use different sources.
- RAG changes the prompt context; it does not update model weights.
- Retrieval can improve evidence access but cannot guarantee correct or complete answers.
- Local persistence makes the database inspectable but not secure by itself.
- Retrieval and generation must be evaluated separately; a successful retrieval can still lead to an unsupported answer.
- Retrieved documents are untrusted data and cannot override higher-priority instructions.

## Contingencies

- **No internet/API:** run all Word2Vec and attention cells; run RAG chunking, fallback retrieval, prompt display, and an executed live-output backup.
- **Missing gensim:** stop before Part 1 and use a prepared output image only; do not replace the core objective with an unannounced library.
- **Missing Chroma:** run the retrieval abstraction with a NumPy/lexical fallback and explain that the persistent database cell is unavailable.
- **API failure or cost concern:** disable live calls, display the exact prompts and retrieved evidence, and use the prepared output.
- **Attention shape error:** print batch/sequence/embedding dimensions and return to the formula before changing code.
- **Generated answer is fluent but unsupported:** use it as the primary discussion example; do not edit it into a better-looking answer.

## Instructor answer guidance

For the hand attention example, the answer should show scores, divide by `sqrt(d_k)`, normalize each query row with softmax, and multiply weights by V. For a causal mask, all entries above the diagonal must be zero after softmax. For RAG, a correct answer cites the chunk that contains the relevant fictional policy and states insufficient evidence for the unanswerable question.

## Result statement template

> The notebook demonstrated that [Word2Vec/attention/RAG operation] produces [observed numerical output]. In the RAG comparison, retrieval changed the available context by adding [source chunk IDs], but the generated response remains a model output that requires evidence checking. The tutorial does not establish clinical accuracy, safety, completeness, fairness, privacy compliance, or deployment readiness.
