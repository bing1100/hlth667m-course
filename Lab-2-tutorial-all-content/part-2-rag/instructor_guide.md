# Instructor Guide — Tutorial 2, Parts 0 to 3

## Before class

- Create a clean environment from `requirements.txt` and run `python -m pytest -q` (expect all tests to pass).
- Run all four notebooks from fresh kernels. Keep the executed copies as your no-network contingency.
- Decide the Part 3 run mode. **`replay` is the default and needs no key.** Only choose `live` if you have checked current model availability and pricing and are comfortable incurring charges in front of the class.
- If running live, test once with your own key beforehand, without displaying it.
- Confirm `tiktoken` can load `cl100k_base` on the room's network. If it cannot, Part 0 sections 6–8 skip themselves cleanly — say so rather than debugging live.
- Run Part 2 section 10 once on the teaching machine before class. The first run downloads `distilgpt2` and `distilbert-base-uncased` through `transformers` (about 600 MB in total) and caches them. If the download is unavailable in the room, section 10 prints a notice and skips itself, and section 8 carries the same contrast on the toy model.
- Run Part 3 section 14 once and open the printed `localhost` address to confirm the chat app loads. Run the stop cell afterwards. Decide whether the class will see `retrieval only` mode alone (no key, no cost) or also `generate` mode (paid, needs `OPENAI_API_KEY` in a `.env` file beside the notebook).
- Decoding has no notebook. Rehearse the decoding slides, which are labelled "Slides only — no notebook". The former notebook is in `archive/Decoding_Temperature_TopK_TopP.ipynb` if you want a live demonstration.
- Open `slides.md` in a Markdown preview with MathJax/KaTeX so the equations render at presentation size.
- Print `../handouts/` for students who want something to write on.

## 150-minute run-of-show

The four notebooks total roughly 125 minutes of content, and the slides-only decoding section adds about 10. Run them across two sessions or select from them; the table below assumes a single long block with a break.

| Time | Notebook / activity | Learner check |
|---:|---|---|
| 0–5 | Arc, boundaries, fictional-data statement. | Name the five stages: tokens, vectors, attention, decoding, retrieval. |
| 5–13 | **Part 0** §1–5: whitespace splitting fails; build BPE; watch merges. | Explain why frequency, not presence, decides token cost. |
| 13–21 | **Part 0** §6–8: production tokenizer; tokens per word; synonyms share no tokens. | State why a lab panel costs more per word than plain prose. |
| 21–34 | **Part 1**: corpus design, Word2Vec, cosine similarity. | Explain why a neighbour is corpus-dependent. |
| 34–41 | **Part 1** t-SNE and its limits. | Name one unjustified reading of the map. |
| 41–48 | *Break.* | |
| 48–57 | **Part 2** §1–4: weighted average; the five-step recipe with its Mask step; the hand-checkable example matched against PyTorch; self-attention with no mask and with a causal mask. | Read one query row aloud correctly. |
| 57–65 | **Part 2** §5–6: gallery of six mask patterns; AR and MLM as a mask plus a target. | Say which tokens `cough` may read under the causal mask and under the full mask. |
| 65–75 | **Part 2** §7–9: train one tiny model under each objective; attention row → output vector → prediction head → probabilities; leak test. | Explain why the AR model gives 0.25 to each symptom after `reports` while the MLM model recovers `cough`. |
| 75–80 | **Part 2** §10–11: `distilgpt2` and `distilbert-base-uncased`; the triangle and the filled square; learner check. | Name the mask a chat model uses while it writes a reply. |
| 80–90 | **Slides only — decoding:** logits, temperature, greedy vs sampling, top-k, top-p. No notebook. | State what `temperature=0` does and does not guarantee. |
| 90–98 | **Part 3** §1–3b: pipeline, corpus, two chunking strategies. | Say what a fixed window gains and loses versus heading chunks. |
| 98–106 | **Part 3** §4–7: four question types; lexical baseline. | Distinguish the unanswerable case from the partial-evidence case. |
| 106–117 | **Part 3** §8–9: terminology expansion and knowledge-graph walk. | Explain what one hop gave that two hops spoiled. |
| 117–128 | **Part 3** §11–12: direct vs RAG comparison; retrieval and generation gates. | Locate a failure in the correct stage. |
| 128–138 | **Part 3** §13: the citation audit. | Identify which fault the automated check misses and why. |
| 138–148 | **Part 3** §14: write the Streamlit app, test it with `AppTest`, launch it on `localhost`, ask it Q3, stop it. | Name one limit of the system that the chat box hides from its user. |
| 148–150 | **Part 3** §15 learner checks and exit prompt. | |

## Teaching the new attention sections (Part 2)

- **Section 2.** The recipe has five steps and the Mask step is the one the notebook studies. Say that the other four steps are identical in every model students will meet.
- **Section 3.** Read the `clinic` row aloud: scores `[0.707, 0.000]`, weights `[0.670, 0.330]`, output `[6.70, 6.60]`. The `assert_close` line is the whole PyTorch comparison.
- **Section 5.** Follow the `cough` row across the six panels. The padding mask removes a column, and the sliding window (14 of 36 pairs at `w = 3`) exists to cut the `n²` cost.
- **Section 6.** The word *mask* has two meanings. The attention mask is the Boolean table, and the `[MASK]` token is a change to the input. AR scores 11 of 11 positions and MLM scores 2 of 12, which is one reason large generative models train autoregressively.
- **Section 7.** Training takes a few seconds on CPU. The AR loss settles near 0.33, and that floor is correct because four symptoms are equally likely after `reports`.
- **Section 8.** This is the figure to slow down for. Row 1 is the honest 0.25 split, row 2 shows attention carrying `cough` forward eight positions to predict `chest`, and row 3 shows the MLM model reading `chest xray` on the right.
- **Section 9.** The AR model shows a change of exactly `0.0` at every position before the edit, and the MLM model shows up to `12.5`. Connect this to Tutorial 1: a wrong causal mask is data leakage, and the low training loss it produces is worthless.
- **Section 10.** Both pretrained models are general-purpose and their completions are not medical statements. The attention maps average 12 heads in one layer and show where weight went.
- Order-blindness and positional encoding now sit on the "beyond the notebook" slide. The toy model in section 7 adds a position embedding, so you can point at `self.pos` when a student asks how the model knows word order. The worked shuffle test is in `archive/Attention_full_walkthrough_v1.ipynb`.

## Teaching the Streamlit section (Part 3 section 14, about 10 minutes)

| Minutes | Step | Say |
|---:|---|---|
| 2 | Read the three Streamlit ideas: the script reruns on every interaction, `st.session_state` keeps the conversation, `@st.cache_data` keeps expensive results. | "Find where `retrieve` and `generate` are called. Very little of this file is about the interface." |
| 2 | Run the `%%writefile` cell and the `AppTest` cell. | "The ranking is the one from section 8, because the app calls the same functions." |
| 4 | Launch, open the printed address, ask the patient's bloodwork question, open the evidence panel, then ask Q3. | "The app still names a best-matching passage for a question the corpus cannot answer." |
| 2 | Set `STOP_CHAT_APP = True`, run the stop cell, and discuss. | "What does this page hide from the person typing into it?" |

The main point of the section is that a chat box hides the system's limits. Nothing on the page tells a user that the corpus has nine chunks, that top-k always returns k passages, or that a cited answer can still be wrong. The evidence panel under each reply lets a reader do the section 13 check.

Safety rules for this section:

- The app stays on `localhost`. Never expose it beyond your own machine: no `--server.address 0.0.0.0`, no tunnelling service, no shared classroom URL.
- Never enter PHI or any real patient text in the chat box or the file uploader. Upload only fictional or public text.
- Keys go in a `.env` file beside the notebook before launch. Never type a key into a cell or into the app.
- Stay in `retrieval only` mode unless you have budgeted for paid calls. `generate` mode embeds the corpus and calls the generation model on every question.
- Run the stop cell before closing the notebook. The app is a separate process and can outlive the notebook.
- For automated or headless execution, set `HLTH667M_SKIP_APP_LAUNCH=1` so the launch cell skips itself.

## The three moments that matter most

If time runs short, protect these.

1. **Part 0, section 6.** Students see `HbA1c` become five tokens and `myocardial` become two fragments. This is the moment tokenization stops being abstract.
2. **Part 2, section 8.** One architecture, trained twice, gives 0.25 to each of four symptoms after `reports` under the AR objective and recovers `cough` from `chest xray` on the right under the MLM objective. The section 9 leak test (`0.0` for AR, `12.5` for MLM) confirms it in one number. This is the concrete explanation for why a chat model writes left to right and an embedding model reads both ways.
3. **Part 3, section 13.** Case B cites a real retrieved chunk, states one true fact, and appends an invented fee. Every automated gate passes. If students leave with one thing, it should be that a citation makes checking possible and a person still has to do the checking.

The former second moment, the same request producing five different outputs, now lives in the decoding slides. The executed demonstration is section 8 of `archive/Decoding_Temperature_TopK_TopP.ipynb`.

## Teaching reminders

- A tokenizer learns from character frequency, not meaning. Two synonyms can share no tokens.
- Word2Vec is not a health ontology.
- Attention weights are computed contributions, not causal explanations or clinical importance.
- Attention is order-blind until position is injected into the input. This point is now made on the "beyond the notebook" slide.
- A mask decides what a position may read. It is applied before the softmax, so it changes the information available, and the display follows from that.
- The attention mask and the `[MASK]` token are two different things that share a word.
- An AR model generates and an MLM model represents. The embedding models used for retrieval in Part 3 belong to the second family.
- A leak test that returns anything other than exactly `0.0` for a causal model means the mask is wrong and the training loss is meaningless.
- `distilgpt2` and `distilbert-base-uncased` are general-purpose models. Their completions are not medical statements.
- Decoding settings change wording and repeatability. They cannot supply missing evidence.
- `temperature=0` gives repeatability, not accuracy. It will repeat a wrong answer just as reliably.
- RAG changes prompt context; it does not retrain weights.
- Retrieval is ranking, not proof. Top-k always returns k candidates, including when none are relevant.
- A terminology encodes people's decisions about equivalence, maintained version by version, and can be wrong or outdated.
- A knowledge graph inherits every error of whoever built it. Its value here is provenance, not authority.
- A local vector store is not an access-control or privacy mechanism.
- Retrieved text is untrusted data, never instructions.
- A chat interface adds access, and it adds no intelligence. It also hides the limits that the notebook makes visible.

## Contingencies

| Problem | Response |
|---|---|
| No network or no key | Part 3 already defaults to `replay`, and the chat app defaults to `retrieval only`. Nothing is lost except a fresh Q4 answer and `generate` mode. |
| `tiktoken` cannot download | Part 0 sections 6–8 skip themselves. Use the printed fallback message as a teaching point about dependency risk. |
| Pretrained models cannot download | Part 2 section 10 prints a notice and skips itself. Teach the contrast from the section 8 figure and show the two attention maps from the slides. |
| The chat app will not start, or the port is taken | The launch cell picks the first free port from 8501. If it still fails, show the `AppTest` output, which carries the same reply and ranking, and consult `troubleshooting.md` after class. |
| A student asks to share the app with a classmate over the network | Decline. The app stays on `localhost`; use the request to discuss what deployment would require. |
| Live API authentication or rate-limit failure mid-class | Set `HLTH667M_RUN_MODE=replay`, restart the kernel, continue. Do not debug credentials in front of the class. |
| Gensim or Chroma missing | Use the executed backup notebook. Do not silently substitute a different component. |
| Attention shape failure after a student edit | Return to the printed dimensions and the formula. The fixture assertions tell you which step broke. |
| A student asks whether the fictional policies are real | They are not, and every file says so in its first two lines. Use the question to discuss provenance. |
| Running long | Drop Part 1's t-SNE section, Part 2's section 10, and Part 3's section 3b. Shorten section 14 to the `AppTest` cell. Protect the three moments listed above. |

## Assessment alignment

These notebooks support Technical Lab 2 (20%). Mapping to `assessment/03_technical_lab_2/rubric_and_instructor_notes.md`:

| Rubric criterion | Points | Where the tutorial prepares students |
|---|---:|---|
| Corpus understanding and use case | 2 | Part 3 §2 (verify the sources), §4 (questions and their evidence needs) |
| Retrieval design and comparison | 4 | Part 3 §3–3b (chunking comparison), §6–7 (lexical scores), §8 (terminology expansion), §11 (embedding retrieval); Part 0 §8 and Part 1 for the vocabulary gap |
| Chatbot implementation and grounding | 4 | Part 3 §5 and §10 (instructions, citations, abstention, untrusted context), §14 (Streamlit chat app) |
| Evaluation, citation audit, and failure analysis | 4 | Part 3 §4 (four evaluation cases), §11 (with and without retrieval), §12 (two gates), §13 (citation audit); Tutorial 1 for evaluation discipline |
| Next-steps plan for production | 2 | Part 3 §14 "What do we see?" and learner check 12; slides "From notebook to chat interface" and "Limits to carry forward" |
| AI use record and reflection | 3 | Same expectations as Lab 1 |
| Communication and reproducibility | 1 | Every notebook's run-from-top convention and limits section |

Lab 2 asks students to build and evaluate a RAG-powered chatbot on a supplied corpus. Part 3 is the direct model for that work, and students are expected to adapt it to the corpus and write their own application. Parts 0 to 2 and the decoding slides explain the mechanisms underneath it.

`assessment/03_technical_lab_2/student_materials.md` holds optional planning templates for the assignment: the test-question table, the retrieval comparison, the citation audit, the failure analysis, and the production brainstorm. The `../student/` notebooks contain `# TODO` cells for in-class practice: two in Part 0, one in Part 1, two in Part 2 (the attention function; the causal and sliding-window masks), and two in Part 3.
