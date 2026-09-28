# Validation Record — Tutorial 2, Parts 0 to 3

## This revision

**Date:** 2026-09-21 (restructure: notebooks renumbered to Parts 0–3, attention notebook rewritten, decoding notebook archived, Streamlit section added)
**Execution environment:** Linux; Python 3.13.5; CPU. Package versions at validation time:

```
numpy==2.2.6      pandas==2.2.3       matplotlib==3.10.0   seaborn==0.13.2
scikit-learn==1.6.1  gensim==4.4.0    torch==2.12.0+cu130  tiktoken==0.13.0
openai==2.43.0    chromadb==1.5.9     transformers==5.8.1  streamlit==1.45.1
```

> **The course target remains Python 3.11.** Run the documented clean 3.11 environment from `requirements.txt` before delivery and record the resulting `pip freeze` here.

## What was verified

| Check | Result |
|---|---|
| Unit and structural tests | Passed: `61 passed` via `python -m pytest -q` |
| Fictional corpus manifest hashes | Passed: 3 sources, all marked fictional, all SHA-256 verified |
| Heading chunking | Passed: 9 stable chunks, 3 per source |
| Fixed-window chunking | Passed: deterministic, no duplicate IDs, disjoint ID namespace from heading chunks |
| Chunk-algorithm fingerprinting | Passed: fingerprint changes with chunk algorithm as well as embedding model |
| Terminology fixture | Passed: 6 concepts, unique codes, version string marked fictional |
| Query expansion | Passed: patient-language query matches `NC-00412` and raises the `results_policy::001` lexical score above its pre-expansion value |
| Knowledge graph | Passed: 18 nodes, 19 edges, no dangling edge, every edge's `evidence` resolves to a real chunk ID |
| Graph hop behaviour | Passed: one hop returns only `results_policy` evidence; two hops strictly supersets it and adds `access_guide::000` |
| Recorded-run replay fixture | Passed: 3 records, all retrieved IDs real, all RAG citations drawn from retrieved chunks, no credential-shaped value |
| Citation audit fixture | Passed: 5 cases, 4 real faults, and the automated checker's behaviour matches each case's declared `automated_check_catches_it` |
| BPE implementation | Passed: deterministic merges; token cost decreases monotonically with corpus frequency (20 occurrences → 1 token; 0 occurrences → 9 tokens) |
| Production tokenizer path | Passed: `cl100k_base` loaded (100,277 entries); the notebook degrades cleanly with a printed message when it cannot |
| Attention fixture | Passed: expected weights and outputs, row sums, causal mask; the from-scratch function matches `torch.nn.functional.scaled_dot_product_attention` under `torch.testing.assert_close` in Part 2 section 3 |
| Attention notebook structure | Passed: weighted average, mask gallery with sliding window, AR and MLM objectives, prediction head, leak test, pretrained section, and "Try it yourself" are all present; at least six figure cells |
| Toy training (Part 2 section 7) | Stored output: 36 sentences, 12 tokens per sentence, 26-word vocabulary; final AR loss `0.326`, final MLM loss `0.13` |
| Prediction from an attention row (Part 2 section 8) | Stored output: AR gives `0.25` to each of four symptoms after `reports`; MLM recovers the hidden symptom at `1.00` for all four tests |
| Leak test (Part 2 section 9) | Passed: largest score change before the edited tokens is exactly `0.0` for the AR model and `12.52` for the MLM model; a test asserts the stored AR value |
| Pretrained models (Part 2 section 10) | Stored output: `distilgpt2` largest weight above the diagonal `0.0`; `distilbert-base-uncased` share of weight above the diagonal `0.382`. The section is optional and degrades cleanly with a printed notice when the download is unavailable |
| Streamlit chat app, offline | Passed: `AppTest` runs `rag_chat_app.py` with no key, the reply note starts with `RETRIEVAL ONLY`, and the first three retrieved chunks are `results_policy::000`, `results_policy::002`, `results_policy::001` |
| Chat app file matches the notebook | Passed: the body of the `%%writefile rag_chat_app.py` cell equals `rag_chat_app.py` on disk |
| Chat app launch settings | Passed: the launch cell binds `--server.address localhost`, and the notebook contains the `STOP_CHAT_APP` stop cell and the PHI upload warning |
| Clean-kernel execution | Passed: all four notebooks are stored executed, with no cell errors |
| Credential-shaped value scan | Passed: no findings in notebooks or data fixtures |
| Markdown-before-code contract | Passed: every code cell in all four notebooks is preceded by a Markdown cell |
| Student notebook generation | Passed: `python ../make_student_notebooks.py --check` reports `student/ is up to date (7 notebooks)`. Tutorial 2 TODO cells: Part 0 = 2, Part 1 = 1, Part 2 = 2, Part 3 = 2. `rag_chat_app.py` is copied into `student/part-2-rag/` |

### Checks retired with this restructure

| Check | Status |
|---|---|
| Positional encoding shuffle test | Retired. The section was removed from the attention notebook. The material is on the "beyond the notebook" slide and in `archive/Attention_full_walkthrough_v1.ipynb`. |
| Decoding arithmetic and decoding repeatability | Retired. Decoding is taught from the slides only. The notebook is in `archive/Decoding_Temperature_TopK_TopP.ipynb` and is outside the test suite. Its last recorded results (2026-09-20): probabilities sum to 1, `top_k=1` equals greedy, greedy gives 1 distinct output over 5 seeds and sampling gives 5. |

Archived notebooks are kept as they were last executed. They are not re-validated.

## Executed notebook outputs

| Notebook | Code cells | With output | Figures |
|---|---:|---:|---:|
| `Part-0_Tokenization_for_Health_Text.ipynb` | 10 | 10 | 1 |
| `Part-1_Word2Vec_and_tSNE.ipynb` | 8 | 8 | 3 |
| `Part-2_Attention_and_Self_Attention.ipynb` | 13 | 13 | 9 |
| `Part-3_RAG_With_and_Without_Retrieval.ipynb` | 25 | 25 | 3 |

## Scope limit of this revision — read before class

**Part 3 was validated in `replay` mode only, and the chat app in `retrieval only` mode only. No live API call was made during this revision.**

Replay mode was exercised end to end and is the default. It reads the verbatim transcript in `data/recorded_runs/rag_recorded_run_2026-09-19.json`, which was captured during the previous revision's live run (details below), and labels every replayed response `RECORDED`.

The `live` code path was **not** re-executed in this revision, and the chat app's `generate` mode has not been exercised with a key. Before teaching in live mode or showing `generate` mode, run the pre-class checklist below.

The stored launch-cell output shows the app starting at `http://localhost:8501` with `key present: False`. Browser rendering of the app was not captured in this record.

## Previous live API validation (2026-09-19)

The RAG notebook, now `Part-3_RAG_With_and_Without_Retrieval.ipynb` (formerly Part 6), was executed from a clean kernel with live calls enabled. That run made four embedding requests (one corpus batch, three questions) and six Responses API generation requests. All completed successfully.

- Q1 retrieved `access_guide::000` first; the RAG response cited it for the 48-hour answer.
- Q2 retrieved both expected evidence chunks in the first two positions and cited both.
- Q3 retrieved low-relevance chunks and correctly abstained.
- Six generation responses used 1,419 tokens total: 473 direct and 946 RAG.
- Chroma collection `hlth667m_rag_demo_v1` contained all 9 records.
- Request IDs and usage are stored in the replay fixture. No credential value is recorded anywhere.

Q4 (`Q4_PARTIAL`) was added **after** that run and therefore has no recorded response. The notebook states this and does not fabricate one. Its retrieval is analysed offline, and the generation risk it represents is taught through the section 13 audit.

This verifies the configured models and API path at one point in time. It does not verify future availability, pricing, or rate limits.

## Before-class checklist

1. Create a clean Python 3.11 environment from `requirements.txt`; record `pip freeze` above.
2. Run `python -m pytest -q` from `part-2-rag/` and confirm all tests pass.
3. Execute all four notebooks from clean kernels. Set `HLTH667M_SKIP_APP_LAUNCH=1` first if the run is automated, so Part 3 section 14 does not start a server.
4. Confirm `tiktoken` can load `cl100k_base` on the room's network. If not, note that Part 0 sections 6–8 will skip themselves.
5. Run Part 2 section 10 once so `distilgpt2` and `distilbert-base-uncased` (about 600 MB in total) are cached. If the download is unavailable, note that section 10 will skip itself.
6. Run Part 3 section 14, open the printed `localhost` address, ask one question, and run the stop cell. Confirm afterwards that the address no longer responds.
7. Decide the Part 3 run mode. **Replay needs nothing.** For live mode only:
   - confirm the configured generation and embedding models are available and within budget;
   - export `OPENAI_API_KEY` and `HLTH667M_RUN_MODE=live` in your shell, never in a cell;
   - run Part 3 once with your key, confirm the collection count is 9, and inspect `artifacts/rag/index_metadata.json`;
   - check request IDs and usage in the ignored `artifacts/rag/run_log.jsonl` without exposing credentials.
8. Regenerate student notebooks if any notebook changed: `python ../make_student_notebooks.py`.

## Known limitations of these materials

- The terminology and knowledge graph are small hand-built fictional fixtures. They inherit their author's errors and model none of the licensing, versioning, or maintenance burden of a real terminology.
- Terminology matching is plain case-insensitive substring matching. It over-matches and under-matches in ways the notebook states explicitly.
- The Part 2 toy model has one head, one layer, 36 template sentences, and a 26-word vocabulary. It demonstrates a mechanism and nothing about medicine.
- `distilgpt2` and `distilbert-base-uncased` are general-purpose models. Their completions are not medical statements, and the attention maps average 12 heads in one layer.
- The chat app is a localhost prototype. It has no authentication, access control, logging policy, or deployment review.
- The generator in the archived decoding notebook is a bigram word-count model and is labelled as such throughout.
- The recorded run is one sample from a non-deterministic service on one date. A fresh live run will word things differently.
- Nothing in these materials evaluates clinical correctness.
