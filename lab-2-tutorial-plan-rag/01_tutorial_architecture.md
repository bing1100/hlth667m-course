# 1. Tutorial Architecture and Shared Design Decisions

## Narrative arc

Students begin with a word-level representation learned from local context. They then inspect the operation that lets a sequence assign different weights to other sequence positions. Finally, they use embeddings and retrieval to place external text into a generation prompt, comparing what changes when evidence is supplied. Each block answers a question left by the previous block: how can words become vectors; how can vectors interact selectively; and how can retrieved text constrain a generated response?

## Learning outcomes

By the end of the 90-minute walkthrough, students should be able to:

1. Define a word embedding and explain that Word2Vec learns vectors from local co-occurrence patterns.
2. Distinguish CBOW from skip-gram at a high level and identify `window`, `vector_size`, `min_count`, `sg`, and `seed` as important settings.
3. Explain why t-SNE is a nonlinear visualization projection and why its 2-D geometry should not be overinterpreted.
4. Compute scaled dot-product attention from queries, keys, and values, including the softmax and optional mask.
5. Distinguish cross-attention from self-attention by the source of `Q`, `K`, and `V`.
6. Read an attention map with a stated row/column convention and identify the effect of a causal mask.
7. Describe RAG as retrieval followed by prompt construction and generation, not as model retraining.
8. Compare no-retrieval and retrieval-grounded answers for evidence use, unsupported claims, and citation coverage.
9. Identify API-key, data privacy, vector-store persistence, cost, and reproducibility risks.

## Notebook matrix

| Notebook | Main input | Main output | Core implementation | Duration |
|---|---|---|---|---:|
| `Part-1_Word2Vec_and_tSNE.ipynb` | Fictional tokenized sentences | Word vectors, nearest words, labeled 2-D map | `gensim.models.Word2Vec`, `sklearn.manifold.TSNE` | 25 min |
| `Part-2_Attention_and_Self_Attention.ipynb` | Tiny synthetic token/value tensors | Attention outputs and heat maps | PyTorch matrix operations; compare official functional API | 30 min |
| `Part-3_RAG_With_and_Without_Retrieval.ipynb` | Fictional local documents and questions | Retrieved chunks and paired responses | Chroma local persistence, OpenAI embeddings/responses | 35 min |

The notebooks are independently runnable. Part 3 may reuse conceptual terms from Parts 1–2 but must not require their runtime state.

The durations include the associated slide explanations, checks, and notebook demonstrations; slide and notebook times are not additive. Follow the minute-by-minute instructor guide as the authoritative 90-minute schedule.

## Theory-to-practice rhythm

Each major concept follows the same sequence: define it on a slide, predict an output, run the smallest relevant notebook cell group, interpret the result, and state a limitation. Avoid presenting all theory before all code. The slide transitions in Document 09 identify each handoff; final slides must use exact notebook section/cell labels.

## Shared dependency policy

- Python 3.10+.
- `numpy`, `pandas`, `matplotlib`, `seaborn`.
- `gensim` for Word2Vec.
- `scikit-learn` for t-SNE and optional cosine similarity checks.
- `torch` for transparent tensor operations and the official attention comparison.
- `openai` and `python-dotenv` for the live RAG path.
- `chromadb` for local persistent vector storage.
- Optional `tiktoken` for token-count display; it must not be required for the core run.

Do not add LangChain or another orchestration framework to the core notebooks. The goal is to expose the operations rather than hide them behind an abstraction.

## Shared reproducibility rules

- Set `RANDOM_STATE = 42`, `torch.manual_seed(42)`, and `workers=1` for Word2Vec classroom runs.
- Use a fixed corpus, fixed document files, fixed chunking parameters, fixed embedding model, fixed prompt template, and fixed question set.
- Do not promise identical OpenAI generated wording across runs. Record model names, timestamps, retrieval parameters, and source chunk IDs.
- Make the RAG notebook idempotent: an explicit `RESET_VECTOR_STORE = False` switch controls deletion/rebuild, and the notebook reports the collection count.
- Never use patient or confidential data.

## Common terminology

Use **representation** for a numerical encoding, **embedding** for a dense vector representation, **query/key/value** for attention inputs, **attention weights** for post-softmax weights, **chunk** for a retrieved text unit, **retriever** for the component selecting chunks, and **generator** for the language-model response step.

## Safety and health-domain boundary

The fictional documents may use health-services topics such as appointment scheduling, symptom escalation instructions, or data-governance procedures, but every document must be marked fictional. The RAG prompt must instruct the generator to answer only from supplied context, say when context is insufficient, and avoid diagnosis, treatment, or emergency advice.
