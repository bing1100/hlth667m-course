# 12. Implementation Contract and Traceability

## 1. Implementation order

1. Create final directory structure, `.gitignore`, `requirements.txt`, and README.
2. Create and hash the three fictional RAG source documents and manifest.
3. Implement reusable pure helpers in `tutorial_utils.py` with tests.
4. Implement Word2Vec notebook against Document 11 fixtures.
5. Implement attention notebook and numerical assertions.
6. Implement RAG offline path, local storage, and expected-evidence tests.
7. Implement live OpenAI path behind `LIVE_API_CALLS`.
8. Create companion materials and full `slides.md` from Document 09.
9. Execute validation matrix and record versions/runtime in `VALIDATION.md`.

## 2. Shared configuration contract

The final tutorial README documents these environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | unset | Authentication; never printed. |
| `OPENAI_GENERATION_MODEL` | `gpt-4.1-mini` | Override generation model if unavailable/changed. |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Override embedding model. |
| `HLTH667M_LIVE_API_CALLS` | `0` | Explicitly enable paid/network calls when `1`. |
| `HLTH667M_ARTIFACT_DIR` | `<tutorial>/artifacts` | Optional artifact-root override. |

Notebook config cell:

```python
RANDOM_STATE = 42
LIVE_API_CALLS = os.getenv("HLTH667M_LIVE_API_CALLS", "0") == "1"
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
GENERATION_MODEL = os.getenv("OPENAI_GENERATION_MODEL", "gpt-4.1-mini")
TOP_K = 4
RESET_VECTOR_STORE = False
COLLECTION_NAME = "hlth667m_rag_demo_v1"
```

## 3. Path-resolution contract

Notebooks must not rely only on `Path.cwd()`. Resolve the tutorial root by checking, in order:

1. a documented `TUTORIAL_ROOT` environment override, when set and valid;
2. `Path.cwd()` if it contains `data/rag_corpus`;
3. the notebook's known final absolute path for the instructor environment.

Print resolved non-secret paths. Create artifact directories with `mkdir(parents=True, exist_ok=True)`.

## 4. Utility-function contracts

Implement these as short, inspectable functions in `tutorial_utils.py`; notebooks display or explain their core logic rather than hiding conceptual operations.

```python
def sha256_bytes(data: bytes) -> str: ...
def load_manifest(corpus_dir: Path) -> list[dict]: ...
def chunk_markdown_by_h2(path: Path, source_metadata: dict) -> list[dict]: ...
def corpus_fingerprint(chunks: list[dict], embedding_model: str) -> str: ...
def token_overlap_scores(query: str, chunks: list[dict]) -> list[tuple[str, float]]: ...
def format_context(retrieved: list[dict]) -> str: ...
def validate_citations(answer: str, retrieved_ids: set[str]) -> dict: ...
```

Function tests must cover malformed headings, duplicate chunk IDs, empty text, stable ordering, unsupported citation IDs, and fingerprint change after source edits.

## 5. Attention function contract

```python
def scaled_dot_product_attention_from_scratch(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    allowed_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return output, attention weights, and pre-softmax scaled scores."""
```

Validate last dimensions, key/value sequence lengths, mask shape/broadcastability, and fully masked rows. Use `masked_fill(~allowed_mask, -torch.inf)`, then `torch.softmax(scores, dim=-1)`. The core notebook shows this function body.

## 6. OpenAI function contracts

Initialize only in live mode after key-presence validation:

```python
client = OpenAI()
```

Embedding helper:

```python
def embed_texts(client, texts: list[str], model: str, batch_size: int = 32) -> list[list[float]]:
    # reject empty strings; preserve input order; batch requests
    response = client.embeddings.create(model=model, input=batch)
    # sort response.data by item.index before extending
```

Generation helper:

```python
def generate_response(client, question: str, context: str | None, model: str) -> dict:
    response = client.responses.create(
        model=model,
        instructions=BASE_INSTRUCTION + condition_instruction,
        input=rag_or_direct_input,
    )
    return {
        "text": response.output_text,
        "request_id": getattr(response, "_request_id", None),
        "usage": response.usage.model_dump() if response.usage else None,
    }
```

Do not set `temperature` unless the selected model's current documentation confirms support. Hosted model outputs are variable; use fixed prompts and evaluation criteria rather than exact output strings. Log request IDs and usage, not credentials or full environment data.

Handle authentication, rate-limit, timeout, connection, and bad-request exceptions with concise messages. Do not retry indefinitely. Use at most two attempts with bounded exponential backoff for transient rate/connection errors.

## 7. Chroma contract

Pin and test one Chroma version during implementation. The intended API is:

```python
import chromadb
client_db = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client_db.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=None,
    metadata={"hnsw:space": "cosine", "tutorial": "hlth667m"},
)
collection.upsert(
    ids=chunk_ids,
    embeddings=chunk_embeddings,
    documents=chunk_texts,
    metadatas=chunk_metadatas,
)
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=min(TOP_K, collection.count()),
    include=["documents", "metadatas", "distances"],
)
```

If the pinned Chroma version replaces `metadata={"hnsw:space": "cosine"}` with an index `configuration`, update the code and requirements together and record the tested syntax in `VALIDATION.md`. Do not mix APIs from different Chroma versions.

Before reuse, compare `index_metadata.json` with current corpus fingerprint, embedding model, chunk algorithm version (`h2-v1`), collection name, and expected record count (9). Mismatch means rebuild; never combine vectors from different embedding models.

`RESET_VECTOR_STORE=True` may call `delete_collection(COLLECTION_NAME)` only after confirming the exact name. Then recreate it. Use `upsert`, not blind `add`, for idempotent reruns.

## 8. Offline retrieval contract

Tokenize query/chunks with `re.findall(r"[a-z0-9]+", text.lower())`, remove exactly `{"a", "an", "and", "are", "does", "how", "is", "of", "the", "to", "what", "when", "who"}`, and score with Jaccard overlap. Sort by descending score then `chunk_id` for deterministic ties. Label results `LEXICAL FALLBACK`, because their ranking is not equivalent to embedding retrieval.

Offline mode does not fabricate generation. It displays direct and RAG prompts plus a fixture-based expected-evidence statement marked `DEMO FALLBACK — NOT AN OPENAI RESPONSE`.

## 9. Caching and cost contract

- Embed the nine chunks once per corpus/model fingerprint.
- Embed each unique question once per embedding model and cache by SHA-256 of model + question.
- Never cache the API key.
- Live classroom maximum by default: 9 chunk embeddings in one batch, 3 query embeddings, and 6 generation calls (direct + RAG for three questions).
- Before class, consult current pricing and record a conservative estimated maximum in the instructor guide; do not hard-code a dollar amount in student materials.

## 10. Test file plan

```text
tests/
├── test_corpus_and_chunking.py
├── test_attention.py
├── test_offline_retrieval.py
├── test_index_metadata.py
└── test_notebook_structure.py
```

Live API tests are opt-in (`RUN_LIVE_OPENAI_TESTS=1`) and never run in a default test command. Mocked client tests verify batching, response ordering, usage extraction, and exception paths.

## 11. Notebook-to-slide traceability

| Requirement | Notebook section | Slides | Validation |
|---|---|---|---|
| Context-based embeddings | Part 1 §§1–3 | 2–5 | corpus/vector tests |
| t-SNE caution | Part 1 §4 | 6 | plot/settings check |
| Q/K/V and scaling | Part 2 §§1–2 | 8–10 | numeric fixture |
| Mask and self-attention | Part 2 §§4–5 | 12–14 | mask/API assertions |
| RAG boundaries | Part 3 title/§§3–6 | 16–20 | prompt/retrieval checks |
| With/without comparison | Part 3 §7 | 21–22 | expected-evidence table |
| Security/governance | all limits; Part 3 setup | 23 | secret scan/data audit |

## 12. Definition of done

- All planned files exist and are linked from README.
- Unit tests pass offline.
- All notebooks execute from fresh kernels in offline mode.
- Live RAG path executes with the instructor key and current pinned SDK/model settings.
- No secret scan findings occur.
- Exact fixtures and expected-evidence checks pass.
- Slides contain visible theory, formulas, speaker notes, alt text, checks, and notebook transitions.
- Instructor guide records actual tested versions, runtimes, collection count, and live-call count.
