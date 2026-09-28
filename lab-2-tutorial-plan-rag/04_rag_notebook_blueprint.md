# 4. RAG With and Without Retrieval Notebook Blueprint

## Notebook identity

- **Filename:** `Part-3_RAG_With_and_Without_Retrieval.ipynb`
- **Duration:** approximately 35 minutes
- **Question:** How does supplying retrieved evidence change a generated answer?
- **External service:** OpenAI embeddings and Responses API, using an environment-loaded key
- **Local store:** persistent Chroma at `artifacts/rag/chroma/`
- **Primary comparison:** same question and generation model, direct prompt versus retrieved-context prompt

## RAG architecture

```text
fictional source documents
        ↓ chunking + metadata
OpenAI embedding model
        ↓ vectors
local Chroma collection: hlth667m_rag_demo
        ↓ similarity query
top-k chunks + source IDs
        ↓ prompt assembly
OpenAI Responses API
        ↓
answer with source labels or insufficient-evidence statement
```

## Storage decision

Use Chroma's local `PersistentClient(path=...)` for the core demonstration. Resolve the path relative to the notebook/tutorial directory, not the current working directory. The planned layout is:

```text
data/rag_corpus/                 # fictional source Markdown files
artifacts/rag/chunks.jsonl        # reproducible chunk records and metadata
artifacts/rag/chroma/             # persistent Chroma files
artifacts/rag/run_log.jsonl       # question, model names, IDs, timings; no API key
```

Create one named collection, for example `hlth667m_rag_demo`, with cosine distance. Use stable IDs such as `source_filename::chunk_000`. On each run, inspect the collection count. Provide `RESET_VECTOR_STORE = False`; when true, delete/recreate the named collection and rebuild it. Do not delete unrelated collections.

The database should not be hosted in the course Git repository, should not contain secrets, and should not contain real health data. It is a generated local teaching artifact. The notebook must explain that a persistent vector store is a database directory, not a backup or an access-control system.

## Required notebook sequence

| Section | Markdown focus | Code/output requirement |
|---|---|---|
| Title | Define RAG, direct generation, retrieval, chunk, embedding, vector store, and safety boundary. | No code. |
| 0. Setup and key check | Explain environment variables and cost/network behavior. | Load `.env` from documented instructor path only if present; check `OPENAI_API_KEY` without printing it; import OpenAI and Chroma; define model names, paths, question, and `LIVE_API_CALLS=False` default. |
| 1. Create/read fictional corpus | Explain source documents and provenance. | Read the three exact fictional Markdown documents specified in Document 11; verify their SHA-256 manifest; display filenames and short previews. |
| 2. Chunk documents | Define chunk size, overlap, metadata, and why chunking affects retrieval. | Use the heading-aware paragraph chunking contract in Document 11; build records with the required schema; write `chunks.jsonl`; display chunk table. |
| 3. Baseline without retrieval | Define a direct prompt and make its evidence boundary explicit. | If live calls enabled and key exists, call the Responses API with the question and no corpus context; otherwise display the exact prompt and use a clearly labelled deterministic placeholder/fallback. |
| 4. Embed and index | Explain dense embeddings, cosine similarity, batching, and API cost. | Call `client.embeddings.create(model="text-embedding-3-small", input=[...])` in batches; persist/reuse only if source fingerprint and embedding-model metadata match; add records to Chroma with embeddings/documents/metadatas/IDs. |
| 5. Retrieve | Explain nearest-neighbor retrieval and `top_k`. | Embed the question; query Chroma with `query_embeddings`; display IDs, distances, sources, and text; do not hide retrieval results inside a framework. |
| 6. Generate with retrieval | Explain prompt assembly and citation instructions. | Build a context block labelled `[source:chunk_id]`; call Responses API using a short system/developer instruction and the context; require citations by chunk ID and an insufficient-context statement. |
| 7. Compare outputs | Explain that retrieval can help or hurt. | Display direct answer, retrieved chunks, RAG answer, source coverage, unsupported-claim checklist, latency/cost notes, and a student judgement. |
| 8. Ablation / try it | Ask students to change `top_k`, question wording, or remove a source chunk. | Re-run retrieval and inspect whether answer evidence changes; do not use patient data. |
| Conclusion | State what the demonstration supports and does not support. | No code; link to evaluation and security resources. |

## OpenAI integration requirements

Use the official Python client and environment-based authentication:

```python
from openai import OpenAI
client = OpenAI()  # reads OPENAI_API_KEY from the environment
```

Use one configuration cell. Default to `EMBEDDING_MODEL = "text-embedding-3-small"` and `GENERATION_MODEL = os.getenv("OPENAI_GENERATION_MODEL", "gpt-4.1-mini")`. Verify availability and pricing before class and override through the environment if needed. Do not scatter model names, hard-code a key, or print environment values.

The notebook must distinguish:

- **embedding request:** maps text to vectors; no answer is generated;
- **retrieval query:** compares a question vector with stored vectors;
- **generation request:** produces wording from the question and, in the RAG branch, retrieved context.

## Offline fallback requirements

The notebook must run its data and retrieval logic without an API key. If no key is available:

- use a deterministic token-overlap retriever for demonstration only, or load a checked-in local embedding cache with a matching corpus fingerprint;
- display the direct and RAG prompts rather than pretending an API response was generated;
- mark all fallback outputs as `DEMO FALLBACK — NOT AN OPENAI RESPONSE`;
- never silently label a local placeholder as model output.

The fallback is not a substitute for the live OpenAI demonstration. The instructor should run one live, low-cost query before class and keep an executed notebook backup.

## Evaluation requirements

Use a fixed question set containing:

1. a question answered explicitly by one document;
2. a question requiring two chunks;
3. an unanswerable question.

For each, record:

- whether retrieved sources contain the answer;
- whether the answer cites the correct chunk IDs;
- whether unsupported detail is present;
- whether the model states insufficient evidence for the unanswerable question;
- direct versus RAG response differences.

Do not score factual correctness using the language model itself as the only judge. Use a small instructor-authored expected-evidence table.

Document 11 fixes the three questions, expected source chunks, source text, direct/RAG prompts, and evaluation rubric. Document 12 fixes the Chroma and OpenAI function contracts.

## Required RAG theory

The notebook and slides must distinguish four boundaries:

1. **Parametric knowledge:** patterns represented in the generator's fitted parameters.
2. **External corpus:** documents supplied by the tutorial author.
3. **Retrieval:** a ranking operation over indexed chunks; no answer is generated.
4. **Grounded generation:** a new response conditioned on selected chunks; the model weights are not retrained.

Explain precision/recall trade-offs in retrieval qualitatively: a small `top_k` can omit needed evidence, while a large `top_k` can add irrelevant text and consume context. Explain that chunk size and overlap alter the retrieval unit. Include prompt injection as a threat: retrieved text is untrusted data and must not override system/developer instructions.

## RAG cautions

- Retrieval does not guarantee truth; it only selects stored text under a similarity rule.
- Top-k similarity is not the same as relevance, completeness, or authority.
- A generated answer may add unsupported claims even when context is present.
- Source citations in a prompt are not proof that every sentence is supported.
- Embeddings and prompts can transmit text to an external service and create cost, privacy, retention, and governance concerns.
- The fictional corpus is intentionally non-sensitive and must remain so.

## Revision of 21 September 2026

The delivered notebook is now `Part-3_RAG_With_and_Without_Retrieval.ipynb`. Section 14 adds a Streamlit chat interface: `%%writefile rag_chat_app.py`, a headless `AppTest` check, a `localhost` launch cell and a stop cell. The app reuses `tutorial_utils` retrieval functions, defaults to a key-free `retrieval only` mode, and offers a paid `generate` mode when `OPENAI_API_KEY` is set. Learner checks moved to section 15.
