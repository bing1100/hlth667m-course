# Tutorial 2 — Tokenization, Representations, Attention, and RAG

This half of Lab 2 moves from raw text to grounded generation and a local chat interface in four notebooks. Decoding (temperature, top-k, top-p) is taught from the slide deck. It demonstrates mechanisms with fictional health-services text. It is **not** clinical decision support, a health chatbot, or a deployment guide.

> **Data and API boundary:** use only the included fictional corpus. Never put patient data, PHI, credentials, restricted course data, or confidential text into a notebook, prompt, embedding request, vector store, screenshot, or log.

## Notebooks, in order

| File | Teaches | Runtime |
|---|---|---:|
| `Part-0_Tokenization_for_Health_Text.ipynb` | Build byte pair encoding by hand; watch clinical terms fragment; connect token counts to cost and context limits. | 20 min |
| `Part-1_Word2Vec_and_tSNE.ipynb` | Train skip-gram Word2Vec; inspect cosine similarity; read a t-SNE map cautiously. | 25 min |
| `Part-2_Attention_and_Self_Attention.ipynb` | *Attention, Masks, and Training Objectives.* Compute Q/K/V attention explicitly and check it against PyTorch; compare six mask patterns; train one tiny model under the autoregressive and the masked-language-modelling objective; follow an attention row to a prediction; run a leak test; see both objectives in `distilgpt2` and `distilbert-base-uncased`. | 30 min |
| `Part-3_RAG_With_and_Without_Retrieval.ipynb` | Chunk, retrieve, ground, and audit. Compare chunking strategies; add a terminology and a knowledge graph; build, test, and launch a local Streamlit chat app. | 50 min |

Numbering restarts in each tutorial. `../part-1-ml/` is Tutorial 1 (Parts 0, 1A, 1B) and this folder is Tutorial 2 (Parts 0 to 3). In this folder's documents "Part N" means the Tutorial 2 notebook unless Tutorial 1 is named.

### What moved out of the sequence

| Material | Where it lives now |
|---|---|
| Decoding: temperature, top-k, top-p (formerly Part 5) | The decoding section of the slides, labelled "Slides only — no notebook". The notebook is archived at `archive/Decoding_Temperature_TopK_TopP.ipynb`. |
| Softmax zoom, order-blindness and positional encoding, standalone PyTorch comparison (earlier attention notebook) | The "beyond the notebook" slide covers order-blindness and positional encoding. The earlier notebook is archived at `archive/Attention_full_walkthrough_v1.ipynb`. The PyTorch check is now one line in Part 2 section 3. |

Archived notebooks are kept for reference. They are outside the test suite and the student build.

## Supporting material

| File | Purpose |
|---|---|
| `slides.md` | Presenter-ready teaching deck. |
| `instructor_guide.md` | Run-of-show, checkpoints, and contingencies. |
| `glossary_and_quick_reference.md` | Definitions, formulas, settings, and artifact map. |
| `exercises_and_answer_key.md` | Practice questions and instructor answers. |
| `troubleshooting.md` | Setup and interpretation fixes. |
| `VALIDATION.md` | What was verified, when, and in which environment. |
| `../handouts/` | One-page printable references. |
| `rag_chat_app.py` | The Streamlit chat app. Part 3 section 14 writes this file with `%%writefile`, and a test confirms the two copies match. |
| `tools/build_attention_notebook.py` | Regenerates the source of the Part 2 attention notebook. |
| `tools/add_streamlit_section.py` | Inserts the Streamlit section into Part 3. Running it twice changes nothing. |
| `archive/` | Earlier notebooks kept for reference: the decoding notebook and the first attention walkthrough. |
| `../student/part-2-rag/` | Student versions with `# TODO` cells (Part 0 has 2, Part 1 has 1, Part 2 has 2, Part 3 has 2) and a copy of `rag_chat_app.py`. Regenerate with `python ../make_student_notebooks.py`. |

## Setup

Tested target: Python 3.11 (3.10+ where packages support it).

```bash
cd part-2-rag
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
jupyter lab
```

`requirements.txt` includes `transformers` for Part 2 section 10 and `streamlit` for Part 3 section 14. The test suite reports `61 passed`.

Choose the `.venv` kernel and run each notebook from top to bottom. If Jupyter runs from another directory, the notebooks search upward for `data/rag_corpus`; set `TUTORIAL_ROOT` to this folder if that fails.

## Optional downloads

| Notebook | Download | Without it |
|---|---|---|
| Part 0 sections 6–8 | The `cl100k_base` tokenizer vocabulary through `tiktoken`. | Those three sections print a message and skip themselves. |
| Part 2 section 10 | `distilgpt2` and `distilbert-base-uncased` through `transformers`, about 600 MB in total, downloaded once and then cached. | Section 10 prints a notice and skips itself. Sections 1–9 are complete without it, and section 8 shows the same contrast on the toy model. |

## Part 3 run modes

Part 3 runs in one of two modes. **Both teach the whole lesson.** `replay` is the default.

| Mode | Behaviour | Key needed | Cost |
|---|---|---|---|
| `replay` | Shows the verbatim transcript of a real paid run recorded on 2026-09-19, labelled `RECORDED`. | No | None |
| `live` | Makes fresh OpenAI embedding and generation calls. | Yes | Yes |

Replay mode is **not a simulation**. It never invents a response. It replays what the model actually returned, together with the request IDs and token counts from that run. The only thing it cannot show is a *new* answer, and it says so where that applies.

To run live:

```bash
export OPENAI_API_KEY='set-this-in-your-shell-not-in-a-notebook'
export HLTH667M_RUN_MODE=live
# Optional overrides:
export OPENAI_GENERATION_MODEL=gpt-4.1-mini
export OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Do not commit `.env`. The notebook checks that a key is present but never prints its value. A live run costs roughly one nine-chunk embedding batch, four question embeddings, and eight generation requests.

Only an instructor should enable live calls, after checking current model availability and pricing.

## The Streamlit chat app (Part 3 section 14)

Section 14 turns the retrieval functions from the earlier sections into a chat page. It has four cells.

| Cell | What it does |
|---|---|
| Write | `%%writefile rag_chat_app.py` saves the cell body as the app file beside the notebook. Edit the cell and rerun it to change the app. |
| Test | `streamlit.testing.v1.AppTest` runs the app in memory, types a patient's question into the chat box, and prints the reply and the evidence ranking as ordinary cell output. No browser is needed. |
| Launch | Starts Streamlit as a separate process bound to `localhost`, on the first free port from 8501, and prints the address to open in a browser tab. |
| Stop | Set `STOP_CHAT_APP = True` and run the cell before closing the notebook. |

The app has two modes, chosen in its sidebar.

| Mode | Retrieval | Answer | Key needed | Cost |
|---|---|---|---|---|
| `retrieval only` (default) | Word overlap, with optional terminology expansion. | None. The app names the best-matching passage and shows the evidence passages a model would receive. | No | None |
| `generate` | Embedding similarity. | A grounded answer with chunk citations. | Yes | Yes |

`generate` mode stays disabled until the app finds `OPENAI_API_KEY`. Put the key in a `.env` file beside the notebook before launching. Never type a key into a notebook cell or into the app.

To run the app without the notebook:

```bash
cd part-2-rag
streamlit run rag_chat_app.py --server.address localhost
```

To execute the notebook without starting a server, for example in an automated run:

```bash
export HLTH667M_SKIP_APP_LAUNCH=1
```

Keep the app on `localhost`. Making it reachable by other people is a deployment decision with privacy, security, and governance consequences, and it is outside this tutorial. The app accepts uploaded `.txt` and `.md` files, so the PHI rule applies to it as well: upload only fictional or public text.

Learner checks follow in section 15.

## Files and persistence

| Path | Contents | In version control |
|---|---|---|
| `data/rag_corpus/` | Three fictional Markdown sources plus a SHA-256 manifest. | yes |
| `data/terminology/` | Fictional `NORTHSTAR-CT` concepts and a small care knowledge graph. | yes |
| `data/recorded_runs/` | The recorded live transcript and the citation-audit fixture. | yes |
| `artifacts/rag/chunks.jsonl` | Generated chunks. | no |
| `artifacts/rag/chroma/` | Local persistent vector store. | no |
| `artifacts/rag/index_metadata.json` | Model and fingerprint metadata. | no |

A local vector store is not a backup, an access-control system, or a privacy-compliance mechanism. Rebuild an index whenever the corpus fingerprint, embedding model, or chunking algorithm changes.

## Fictional terminology and graph

`data/terminology/` uses an **invented** code system called `NORTHSTAR-CT`. These are not SNOMED CT, LOINC, or ICD-10 codes and must not be reused outside this tutorial. A production system would use a licensed terminology, which adds licensing, versioning, release-cycle, and mapping-maintenance obligations this fixture does not model.

Every knowledge-graph edge carries the corpus chunk it was derived from, so a graph claim can be traced back to source text rather than asserted on the graph's authority.

## Validation and teaching limits

Run `python -m pytest -q` before class, then execute the notebooks from clean kernels. Part 3 evaluates four fixed questions: single-chunk, multi-chunk, unanswerable, and **partial evidence** — the realistic case where retrieval is relevant but incomplete.

A citation is a traceability aid. It is not proof that a response is correct, complete, safe, or clinically valid. Section 13 of Part 3 makes this concrete: of five hand-written answers containing four deliberate faults, automated citation checking catches exactly one.
