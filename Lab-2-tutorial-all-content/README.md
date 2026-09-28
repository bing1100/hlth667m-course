# HLTH 667M — Lab 2

Seven notebooks in two tutorials taking you from a raw health CSV to a grounded, audited retrieval system with a local chat interface. Everything runs offline by default; nothing requires an API key.

> **Boundary for the whole lab:** these are teaching materials. They are not clinical decision support, validated tools, causal analyses, or deployment designs. Never put patient data, PHI, credentials, or confidential text into any notebook, prompt, screenshot, or log.

## The notebooks, in order

| Tutorial | Part | Notebook | Teaches | Time |
|---|---|---|---|---:|
| 1 | 0 | [`part-1-ml/Part-0_Healthcare_Data_Processing_and_Analytics.ipynb`](part-1-ml/Part-0_Healthcare_Data_Processing_and_Analytics.ipynb) | Audit a real health CSV; decide what each column is and when it becomes known. | 30 min |
| 1 | 1A | [`part-1-ml/Part-1A_Classification_ML_Pipeline.ipynb`](part-1-ml/Part-1A_Classification_ML_Pipeline.ipynb) | Predicting a category; metric choice under class imbalance; encoding categorical health data. | 60 min |
| 1 | 1B | [`part-1-ml/Part-1B_Regression_ML_Pipeline.ipynb`](part-1-ml/Part-1B_Regression_ML_Pipeline.ipynb) | Predicting a number; residual diagnostics; watching leakage manufacture a result. | 60 min |
| 2 | 0 | [`part-2-rag/Part-0_Tokenization_for_Health_Text.ipynb`](part-2-rag/Part-0_Tokenization_for_Health_Text.ipynb) | How clinical text becomes tokens, and what that costs. | 20 min |
| 2 | 1 | [`part-2-rag/Part-1_Word2Vec_and_tSNE.ipynb`](part-2-rag/Part-1_Word2Vec_and_tSNE.ipynb) | How a token acquires a meaning; reading a t-SNE map cautiously. | 25 min |
| 2 | 2 | [`part-2-rag/Part-2_Attention_and_Self_Attention.ipynb`](part-2-rag/Part-2_Attention_and_Self_Attention.ipynb) | Attention, masks, and training objectives: Q/K/V attention by hand; six mask patterns; autoregressive and masked-language-model training; from an attention row to a prediction. | 30 min |
| 2 | 3 | [`part-2-rag/Part-3_RAG_With_and_Without_Retrieval.ipynb`](part-2-rag/Part-3_RAG_With_and_Without_Retrieval.ipynb) | Chunking, retrieval, terminology, knowledge graphs, auditing citations, and a local Streamlit chat interface. | 50 min |

Numbering restarts in each tutorial, so each folder begins at Part 0. Where a reference could be read either way, this lab writes "Tutorial 1 Part 0" or "Tutorial 2 Part 0".

Decoding (temperature, top-k, top-p) is taught from the slide deck. The former decoding notebook is kept for reference at [`part-2-rag/archive/Decoding_Temperature_TopK_TopP.ipynb`](part-2-rag/archive/Decoding_Temperature_TopK_TopP.ipynb).

## Which folder is which

| Folder | Contents |
|---|---|
| [`part-1-ml/`](part-1-ml/) | Tutorial 1, Parts 0–1B: the supervised-learning workflow. `scikit-learn` only. |
| [`part-2-rag/`](part-2-rag/) | Tutorial 2, Parts 0–3: tokenization through retrieval-augmented generation and a local chat app. Earlier notebook versions are in `part-2-rag/archive/`. |
| [`handouts/`](handouts/) | Four printable one-pagers. |
| [`student/`](student/) | Student versions of all seven notebooks, with `# TODO` cells and outputs cleared, plus a copy of `rag_chat_app.py`. |

Each folder has its own `README.md`, `instructor_guide.md`, `slides.md`, `glossary_and_quick_reference.md`, `exercises_and_answer_key.md`, `troubleshooting.md`, and `VALIDATION.md`.

## Quick start

```bash
# Tutorial 1 (scikit-learn only)
cd part-1-ml
python -m pip install -r requirements.txt
python -m pytest -q          # 27 structural checks
jupyter lab

# Tutorial 2 (adds gensim, torch, tiktoken, transformers, chromadb, openai, streamlit)
cd ../part-2-rag
python -m pip install -r requirements.txt
python -m pytest -q          # 61 unit and structural checks
jupyter lab
```

Open each notebook and run from the top. If a kernel restarts, use **Run All** — a restarted kernel forgets every variable.

## Nothing here requires an API key

Tutorial 2 Part 3 defaults to `replay` mode, which shows the verbatim transcript of a real paid run recorded on 2026-09-19. Every replayed response carries a `RECORDED` badge naming that date. It is not a simulation and it never invents a response.

Instructors who want fresh output can set `HLTH667M_RUN_MODE=live` with a key exported in the shell. See [`part-2-rag/README.md`](part-2-rag/README.md#part-3-run-modes).

Tutorial 2 Part 3 section 14 builds a Streamlit chat app and launches it on `localhost`. The app starts in `retrieval only` mode, which needs no key and makes no network call. Its `generate` mode is paid and stays disabled until an `OPENAI_API_KEY` is found in a `.env` file. See [`part-2-rag/README.md`](part-2-rag/README.md#the-streamlit-chat-app-part-3-section-14).

Two sections download files on first use and skip themselves with a printed message when the download is unavailable. Tutorial 2 Part 0 downloads a tokenizer vocabulary and skips three sections without it. Tutorial 2 Part 2 section 10 downloads two small pretrained models (about 600 MB in total) and skips only that section without them.

## Printable handouts

| Handout | Use it when |
|---|---|
| [`handouts/pipeline_map.md`](handouts/pipeline_map.md) | Building any preprocessing workflow. |
| [`handouts/data_splitting_map.md`](handouts/data_splitting_map.md) | Deciding how to split, and which splitter to use. |
| [`handouts/metric_chooser.md`](handouts/metric_chooser.md) | Choosing and reporting a metric. |
| [`handouts/leakage_checklist.md`](handouts/leakage_checklist.md) | **Before reporting any result.** |

## Student versions

`student/` holds all seven notebooks with selected code cells replaced by `# TODO` stubs and all outputs cleared. The complete notebooks are the instructor answer key.

Regenerate after editing any notebook:

```bash
python make_student_notebooks.py           # rewrite student/
python make_student_notebooks.py --check   # fail if student/ is stale
```

Because the student version is generated from the complete one, the two cannot drift apart. The script also copies `rag_chat_app.py` into `student/part-2-rag/` so the Streamlit section runs from the student folder.

## Assessment

These materials support **Technical Lab 2 (20%)**. The graded rubric lives in `../../hlth667m-course/current_course_materials/assessment/03_technical_lab_2/`.

| Rubric criterion | Points | Where the tutorial prepares students |
|---|---:|---|
| Corpus understanding and use case | 2 | Tutorial 2 Part 3 §2 (verify the sources), §4 (questions and their evidence needs) |
| Retrieval design and comparison | 4 | Tutorial 2 Part 3 §3–3b (chunking comparison), §6–7 (lexical scores), §8 (terminology expansion), §11 (embedding retrieval); Tutorial 2 Part 0 §8 and Tutorial 2 Part 1 for the vocabulary gap |
| Chatbot implementation and grounding | 4 | Tutorial 2 Part 3 §5 and §10 (instructions, citations, abstention, untrusted context), §14 (Streamlit chat app) |
| Evaluation, citation audit, and failure analysis | 4 | Tutorial 2 Part 3 §4 (four evaluation cases), §11 (with and without retrieval), §12 (two gates), §13 (citation audit); Tutorial 1 for evaluation discipline |
| Next-steps plan for production | 2 | Tutorial 2 Part 3 §14 "What do we see?" and learner check 12; slides "From notebook to chat interface" and "Limits to carry forward" |
| AI use record and reflection | 3 | Same expectations as Lab 1 |
| Communication and reproducibility | 1 | Every notebook's run-from-top convention and limits section |

Lab 2 asks students to build and evaluate a RAG-powered chatbot on a supplied corpus (`hlth-667-668/Lab-2/lab-2-assignment.md`). Tutorial 2 Part 3 is the direct model for that work. Tutorial 2 Parts 0 to 2 and the decoding slides explain the mechanisms underneath it, and Tutorial 1 builds the evaluation habits the assignment relies on.

## The three things worth protecting if time runs short

1. **Tutorial 2 Part 0 §6** — `HbA1c` becomes five tokens and `myocardial` splits into fragments. Tokenization stops being abstract.
2. **Tutorial 2 Part 2 §8** — one architecture gives 0.25 to each of four symptoms after `reports` under autoregressive training, and recovers `cough` from `chest xray` on the right under masked language modelling. The mask decides what a model can know.
3. **Tutorial 2 Part 3 §13** — of four deliberately faulty answers, automated citation checking catches one. A citation makes checking possible, and a person still has to do the checking.

## Before teaching

Run `python -m pytest -q` in both folders, execute every notebook from a clean kernel, and read the `VALIDATION.md` in each folder — including its stated scope limits.
