# HLTH 667M Lab 2 Tutorial Plan — Representations, Attention, and RAG

> **Status: delivered.** These materials were built and now live at
> `/home/bhux/research/proposals/hlth-667-668/Lab-2-tutorial/part-2-rag/`.
> The delivery extends this plan; see **Delivered structure** at the end of this file.

## Purpose and scope

This planning package specifies a 90-minute, three-notebook tutorial sequence that moves from local word representations to attention and then to retrieval-augmented generation (RAG):

1. **Word2Vec and embedding maps:** train a small Word2Vec model on reproducible example text, inspect nearest words, and project selected vectors with t-SNE.
2. **Attention and self-attention:** implement scaled dot-product attention with explicit tensor operations, inspect synthetic attention maps, and compare self-attention with cross-attention and a causal mask.
3. **RAG with and without retrieval:** compare a direct language-model response with a response grounded in retrieved local documents using OpenAI embeddings and generation.

The sequence teaches representations and mechanisms. It does not teach clinical prediction, medical advice, autonomous decision-making, or deployment of a health chatbot.

> **Data boundary:** Use fictional or public educational text only. Do not place patient information, protected health information, credentials, or restricted course data in prompts, embeddings, vector stores, or notebooks.

## Destination and source locations

| Purpose | Absolute path |
|---|---|
| Planning package | `/home/bhux/research/proposals/hlth667m-course/lab-2-tutorial-plan-rag/` |
| Planned tutorial materials | `/home/bhux/research/proposals/hlth667m-course/lab-2-tutorial-rag/` |
| Existing course-plan model | `/home/bhux/research/proposals/hlth667m-course/lab-2-tutorial-plan/` |
| API-key source for instructor environment only | `/home/bhux/research/proposals/hlth-667-668/.env` |

The final notebooks must load `OPENAI_API_KEY` from an environment variable or documented local `.env` path. They must never contain the key, display it, or commit the `.env` file.

## Planned final deliverables

```text
/home/bhux/research/proposals/hlth667m-course/lab-2-tutorial-rag/
├── README.md
├── requirements.txt
├── .gitignore
├── tutorial_utils.py
├── Part-1_Word2Vec_and_tSNE.ipynb
├── Part-2_Attention_and_Self_Attention.ipynb
├── Part-3_RAG_With_and_Without_Retrieval.ipynb
├── data/
│   └── rag_corpus/
│       ├── fictional_access_guide.md
│       ├── fictional_results_policy.md
│       ├── fictional_privacy_policy.md
│       └── manifest.jsonl
├── artifacts/
│   ├── word2vec/
│   └── rag/
│       ├── chunks.jsonl
│       ├── index_metadata.json
│       ├── run_log.jsonl
│       └── chroma/                      # persistent local vector store
├── slides.md
├── glossary_and_quick_reference.md
├── exercises_and_answer_key.md
├── troubleshooting.md
├── instructor_guide.md
├── VALIDATION.md
└── tests/
    ├── test_corpus_and_chunking.py
    ├── test_attention.py
    ├── test_offline_retrieval.py
    ├── test_index_metadata.py
    └── test_notebook_structure.py
```

`data/` contains source teaching documents and may be versioned if they contain only fictional/public material. `artifacts/` contains generated models, embeddings, and the local Chroma database and should be gitignored unless there is a deliberate educational reason to distribute a pinned artifact.

## Planning documents

| Document | Use |
|---|---|
| `01_tutorial_architecture.md` | Narrative arc, learning outcomes, notebook boundaries, timing, dependencies, and safety decisions. |
| `02_word2vec_notebook_blueprint.md` | Cell-level implementation plan for Word2Vec and t-SNE. |
| `03_attention_notebook_blueprint.md` | Cell-level implementation plan for attention and self-attention. |
| `04_rag_notebook_blueprint.md` | Cell-level implementation plan for RAG with/without retrieval and local database design. |
| `05_visualizations_and_interpretation.md` | Required plots, attention-map interpretation, embedding-map cautions, and RAG comparison criteria. |
| `06_resources_dependencies_and_validation.md` | Official links, package policy, API-key handling, persistence, and acceptance tests. |
| `07_notebook_markdown_style_guide.md` | Required explanation pattern, accessibility, cell size, and safety language. |
| `08_supplementary_materials_plan.md` | Glossary, exercises, reading prompts, extensions, and troubleshooting coverage. |
| `09_slide_deck_helper.md` | Presenter-ready slide sequence aligned to the three notebooks. |
| `10_instructor_guide_plan.md` | 90-minute run-of-show, checks, expected outputs, contingencies, and answer guidance. |
| `11_deterministic_fixtures_and_expected_outputs.md` | Exact corpora, attention tensors, RAG documents/questions, prompts, and expected evidence. |
| `12_implementation_contract_and_traceability.md` | Module contracts, configuration, file schemas, implementation order, and requirement-to-test matrix. |

## Clarifications resolved by this plan

1. **What does “Word2Vec example data” mean?** A small fictional corpus is created inside the notebook from explicit sentences. It is not downloaded and is not patient data.
2. **What does “embedding mapping using t-SNE” mean?** Select a documented vocabulary subset, transform its vectors to two dimensions with fixed `random_state`, label points, and interpret neighborhoods cautiously. Do not imply that distances in the 2-D plot are authoritative semantic measurements.
3. **What does “attention maps” mean?** Display the post-softmax attention weights with rows as queries and columns as keys. Annotate token order, color scale, mask entries, and the direction of interpretation.
4. **Where does the RAG database live?** The primary classroom implementation uses a local persistent Chroma database at `artifacts/rag/chroma/`, relative to the final tutorial directory. Source documents remain in `data/rag_corpus/`; a chunk manifest records source, chunk ID, text, and metadata.
5. **What happens without an API key?** The notebook must still run its chunking, retrieval, evidence-display, and prompt-construction cells using a deterministic local lexical fallback. The OpenAI generation/embedding demonstration is skipped with a clear message, not silently substituted.
6. **Should OpenAI hosted vector stores be used?** No for the core notebook. Hosted retrieval is a useful extension, but local Chroma makes storage visible, reproducible, inspectable, and inexpensive to reset for a classroom.

## Core safety statement

All generated answers must be framed as outputs conditioned on the prompt and retrieved text. Students must inspect source citations and missing evidence. The notebook must state that a retrieved context can be incomplete, stale, or wrong, and that a fluent answer is not a health recommendation or evidence of clinical validity.

## Definition of implementation-ready

No implementer should need to invent tutorial data, tensor values, prompt wording, chunk IDs, model defaults, storage paths, evaluation questions, expected evidence, slide wording, or validation checks. Documents 11 and 12 fix those details. Hosted model availability and pricing are the only intentionally runtime-verified external details.

## Delivered structure

The three notebooks specified here were delivered, renumbered to continue from `part-1-ml/`, and **two more were added** to cover graded rubric criteria this plan did not address.

| Planned | Delivered as |
|---|---|
| — | `Part-2_Tokenization_for_Health_Text.ipynb` *(new)* |
| `Part-1_Word2Vec_and_tSNE.ipynb` | `Part-3_Word2Vec_and_tSNE.ipynb` |
| `Part-2_Attention_and_Self_Attention.ipynb` | `Part-4_Attention_and_Self_Attention.ipynb` *(+ positional encoding)* |
| — | `Part-5_Decoding_Temperature_TopK_TopP.ipynb` *(new)* |
| `Part-3_RAG_With_and_Without_Retrieval.ipynb` | `Part-6_RAG_With_and_Without_Retrieval.ipynb` *(substantially extended)* |

Renumbering removed a collision: the lab previously contained two different notebooks called "Part 1" and two called "Part 2".

### Why the two new notebooks exist

The graded rubric in `../current_course_materials/assessment/03_technical_lab_2/` allocates 4 points to tokenization plus attention and 3 points to a decoding comparison. Neither topic appeared anywhere in the planned materials. Both are also named directly in the syllabus description of Lab 2.

### What was added to Part 6

- **Run modes.** `replay` is the default and needs no API key: it shows the verbatim transcript of a real paid run, labelled `RECORDED`. The previous revision required live paid calls and would `raise` without a key, which would have blocked every student.
- **A chunking comparison** (heading versus fixed-window), for the rubric's chunking criterion.
- **A fourth evaluation case**, `Q4_PARTIAL`, where retrieval is relevant but incomplete — the realistic failure the plan's three cases did not cover.
- **A fictional terminology and knowledge graph** in `data/terminology/`, for the rubric's "terminology, ontology, or graph evidence" criterion. Every graph edge carries the corpus chunk it was derived from.
- **A citation audit exercise** in which four deliberately faulty answers are checked; automated citation validation catches exactly one.

### Corrections to plan assumptions

Document 06 and the earlier delivered docs described an offline mode with labelled fallback responses. That mode had been removed from the notebook while the surrounding documentation still described it. The delivered version resolves the contradiction with recorded replay, which never fabricates a response and never presents replayed text as fresh.

## Revision of 21 September 2026

The student-facing notebooks now live alone in `../../hlth-667-668/Lab-2-tutorial/`. Everything else (complete notebooks with tests, student TODO copies, instructor guides, handouts) lives in `../Lab-2-tutorial-all-content/`, and the Beamer deck in `../latex-slides/`.

| Previous | Now |
|---|---|
| `Part-2_Tokenization_for_Health_Text.ipynb` | `Part-0_Tokenization_for_Health_Text.ipynb` |
| `Part-3_Word2Vec_and_tSNE.ipynb` | `Part-1_Word2Vec_and_tSNE.ipynb` |
| `Part-4_Attention_and_Self_Attention.ipynb` | `Part-2_Attention_and_Self_Attention.ipynb` *(rewritten, see below)* |
| `Part-5_Decoding_Temperature_TopK_TopP.ipynb` | slides only; notebook archived in `Lab-2-tutorial-all-content/part-2-rag/archive/` |
| `Part-6_RAG_With_and_Without_Retrieval.ipynb` | `Part-3_RAG_With_and_Without_Retrieval.ipynb` *(+ Streamlit chat app)* |

Each tutorial folder now numbers its notebooks from Part 0.

### Attention notebook

A full hand replication of every attention matrix taught little beyond the formula. The notebook keeps the short opening (weighted average, the recipe, one hand-checkable example verified against PyTorch, self-attention) and spends the rest of its time on masks:

- a gallery of six mask patterns on one sentence (bidirectional, causal, padding, causal + padding, prefix, sliding window);
- the mapping from the causal mask to the autoregressive objective and from the full mask to masked language modelling, including the two meanings of "mask";
- one tiny architecture trained under each objective on a fictional template corpus;
- a worked path from an attention row, through the output vector and prediction head, to a probability for every word, for an AR prediction and an MLM prediction;
- a leak test showing that the causal mask blocks the future exactly (`0.0`) while the MLM model uses it (`12.5`);
- the same contrast in `distilgpt2` and `distilbert-base-uncased`, with their layer-1 attention maps.

Positional encoding and the `√d_k` demonstration remain in the slide deck. The earlier notebook is archived as `archive/Attention_full_walkthrough_v1.ipynb`. `tools/build_attention_notebook.py` regenerates the notebook source.

### Decoding

Temperature, top-k and top-p stay in the deck as a slides-only section between attention and RAG. The archived notebook remains available to students who want runnable code.

### Assignment

Lab 2 is now a build assignment: students program and evaluate a RAG-powered chatbot on a supplied corpus, plan its path to production with AI, and submit an AI use record and a one-page reflection (`hlth-667-668/Lab-2/lab-2-assignment.md`). The rubric in `../current_course_materials/assessment/03_technical_lab_2/` was rewritten to match, so the earlier worksheet criteria for tokenization, attention and decoding no longer carry points.

### RAG notebook

Section 14 builds a Streamlit chat interface over the same retrieval functions. The notebook writes `rag_chat_app.py` with `%%writefile`, tests it headlessly with `streamlit.testing.v1.AppTest`, and launches it bound to `localhost`. The app runs in `retrieval only` mode without a key and offers a paid `generate` mode when `OPENAI_API_KEY` is present. `tools/add_streamlit_section.py` inserts the section idempotently.

### Slides

Each key notebook section now has an explanation slide followed by a "From the notebook" slide. Figure slides embed the image exported from the executed notebook by `latex-slides/export_notebook_figures.py`; slides that redraw a notebook table natively carry a "From the notebook" tag naming the part and section.
