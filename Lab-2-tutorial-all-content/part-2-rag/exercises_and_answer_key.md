# Exercises and Answer Key — Tutorial 2, Parts 0 to 3

Decoding is taught from the slides, so its questions sit in their own block. The questions on order-blindness and positional encoding moved there too, because that material now lives on the "beyond the notebook" slide and in `archive/Attention_full_walkthrough_v1.ipynb`.

## Student exercises

### Part 0 — Tokenization
1. Why can a subword tokenizer never produce an "unknown" token?
2. A term is split into seven tokens. What does that tell you about the tokenizer's training corpus, and what does it *not* tell you?
3. You estimate an API budget using the word count of an English report. What happens when the real input is a laboratory panel?
4. Name two phrases that mean the same thing clinically but are likely to share no tokens.

### Part 1 — Word2Vec
5. In `clinic schedules followup through portal`, list the skip-gram context tokens for `followup` with a window of 2.
6. State one difference between CBOW and skip-gram.
7. Name one claim you cannot make from the 2-D t-SNE plot.

### Part 2 — Attention, masks, and training objectives
8. For scores `[0, 0]`, what are the two softmax weights?
9. Which cells are blocked above the diagonal of a 3×3 causal mask?
10. If `Q=XW_Q`, `K=XW_K`, and `V=XW_V`, is this self- or cross-attention? Why?
11. List the five steps of the attention recipe in order. Which step differs between a GPT-style and a BERT-style model?
12. In the section 5 sentence `the patient reports cough today [PAD]`, which tokens may `cough` read under the full mask, the causal mask, and the sliding window with `w = 3`?
13. The padding mask removes a column and keeps the `[PAD]` row. Why is that safe?
14. The word *mask* has two meanings in section 6. State both, and say which of them MLM uses.
15. For a 12-token sentence, AR scores 11 positions and MLM scores 2. Explain both numbers.
16. After `the patient reports`, the trained AR model gives 0.25 to each of four symptoms. Is the model under-trained? Explain.
17. The MLM model recovers `cough` at the blank with near certainty. Which tokens did it read to do so, and why could the AR model at the same position not read them?
18. Trace the three steps from an attention row to a probability distribution. Define *logits* and *prediction head* as you go.
19. In the leak test, the AR model shows a change of exactly `0.0` before the edited tokens and the MLM model shows `12.5`. What does each number tell you?
20. You train an AR model with the full mask by mistake. What happens to the training loss, and what does the leak test report?
21. An embedding model turns a whole paragraph into one vector for search. Which objective family suits it, and why?
22. The `distilgpt2` attention map is a lower triangle with a bright first column. Explain both features.

### Slides only — Decoding, and beyond the notebook
23. Does raising the temperature change what the model computed? Explain.
24. Is `top_k=1` different from greedy decoding?
25. Why does top-p keep a different number of tokens in different situations, while top-k does not?
26. A colleague says "we set temperature to 0, so the output is reliable." Correct this in two sentences.
27. Can any decoding setting prevent a model from stating a fact that is not in its input?
28. You shuffle the rows of `X` and the self-attention outputs come back in the same shuffled order, otherwise unchanged. What does that prove, and what fixes it?

### Part 3 — RAG, structured knowledge, and the chat app
29. Which chunk should answer "How long does a portal registration code remain valid?"
30. Give one advantage and one disadvantage of fixed-window chunking versus heading-based chunking.
31. Rewrite "Parking costs $10 [access_guide::000]" as an evidence-bounded answer.
32. List four things that must never be sent in a hosted API request for this tutorial.
33. Why is a persistent local vector store not equivalent to a secure database service?
34. A correct chunk is retrieved but the answer invents a fee. Which stage needs investigation first?
35. Why must retrieved text be treated as untrusted data rather than instructions?
36. What did the terminology supply that the embedding model did not?
37. Why is a two-hop graph walk not automatically better than one hop?
38. Of the five answers in the section 13 audit, which is the most dangerous and why?
39. Q3 and Q4 both fail to produce a complete answer. Explain the difference between them and why Q4 is the more realistic risk.
40. A Streamlit script reruns from top to bottom on every interaction. Where does the chat app keep the conversation, and what would happen without it?
41. The chat app in `retrieval only` mode returns the same ranking as section 8. Why?
42. Name three limits of the system that the chat page hides from the person using it.
43. Why is the launch cell bound to `localhost`, and what would have to be decided before changing that?

## Instructor answer key

1. It can always fall back to single characters, which are all in the vocabulary. Nothing is out of vocabulary; rare text is just expensive.
2. That the term was **rare** in that corpus. It says nothing about whether the term is clinically important or correctly understood.
3. You underestimate the cost. Clinical text runs roughly 2–4 tokens per word against about 1.1 for plain prose. In Part 0 the lab panel reached 4.0.
4. Examples: `laboratory result` / `bloodwork`; `appointment` / `visit`; `proxy access` / `caregiver access`. Part 0 section 8 shows zero shared tokens for two of these.
5. `clinic`, `schedules`, `through`, `portal`.
6. CBOW predicts a centre word from its context; skip-gram predicts nearby context from a centre word.
7. Axis meaning, global distance, cluster size, or that a validated clinical category was discovered.
8. `[0.5, 0.5]`.
9. `(0,1)`, `(0,2)`, and `(1,2)` using zero-based row/column positions.
10. Self-attention: one input sequence supplies all three projected tensors.
11. Match (`QKᵀ`), scale (divide by `√d_k`), mask (disallowed pairs to `−∞`), normalize (row-wise softmax), combine (`weights @ V`). The mask step differs: causal for GPT-style models, full for BERT-style models. The other four steps are identical.
12. Full: `the, patient, reports, cough, today, [PAD]`. Causal: `the, patient, reports, cough`. Sliding window: `patient, reports, cough`.
13. No token reads `[PAD]`, so filler never enters another token's output. The `[PAD]` row is still computed, and its output is ignored when the loss is calculated.
14. The **attention mask** is the Boolean table saying which positions each token may read. The **`[MASK]` token** is a change to the input that hides a word. MLM uses a `[MASK]` token together with a full attention mask.
15. AR predicts the next token at every input position, and the input is the sentence without its last token, so 11 positions are scored. MLM scores only the masked positions, here 2, because an unmasked position can see its own token and predicting it teaches nothing.
16. No. Nothing to the left of the blank says which symptom is coming, and the corpus uses the four symptoms equally often, so 0.25 each is the best possible answer. This is also why the AR loss settles above zero.
17. It read `chest` and `xray`, to the **right** of the blank. The full mask allows that. The AR model at that position has a causal mask, so its row stops at `reports`.
18. **Attend:** the position's query is matched against the keys it may read, giving one row of weights. **Combine:** the weights blend the value vectors into one output vector. **Predict:** the prediction head, a linear layer, turns the output vector into one score per vocabulary word (the logits), and softmax turns the logits into probabilities.
19. `0.0` shows that the causal mask guarantees no later token can influence an earlier prediction, which is what makes left-to-right generation possible. `12.5` shows that the MLM model reads both directions as intended, which also means it cannot simply be run left to right to generate text.
20. The loss collapses toward zero because each position reads its own answer, and the leak test turns non-zero. The low loss is worthless. This is the sequence-model version of the data leakage in Tutorial 1.
21. MLM-style, with a full mask, because every token should be read in the context of the entire passage.
22. The triangle is the causal mask: the largest weight above the diagonal is exactly `0.0`. The bright first column is a known habit of GPT-style models, which park weight on the first token when a head has nothing specific to read.
23. No. The model's scores are already final. Temperature only rescales them before the softmax.
24. No, they are the same rule: one surviving candidate receives all the probability.
25. Top-p keeps the smallest set whose cumulative probability reaches the threshold, so a confident distribution needs fewer tokens and an uncertain one needs more. Top-k keeps a fixed count regardless.
26. `temperature=0` makes the output **repeatable**, not correct — it will return the same wrong answer just as reliably. It is also not reproducible across time unless the model version is pinned and recorded.
27. No. Decoding controls how tokens are selected from the model's distribution. Supplying and checking evidence is the only remedy, which is what Part 3 addresses.
28. It proves attention is **permutation-equivariant** — it sees a set, not a sequence, and has no notion of order. Adding a positional encoding to `X` before the Q/K/V projections fixes it. Accept any mention of learned or rotary encodings as alternatives, and accept a pointer to `self.pos` in the Part 2 section 7 toy model.
29. `access_guide::000`.
30. Advantage: predictable sizes, works on unstructured text with no headings. Disadvantage: a window can begin mid-sentence and mix two topics, and an answer sitting on a boundary can be split across two chunks so neither ranks well.
31. "The supplied context does not contain parking-fee information."
32. Examples: API keys, patient data or PHI, restricted course records, confidential identifiers.
33. Persistence supplies none of authorization, audit policy, backup controls, retention rules, or governance.
34. Generation and evidence use. Record retrieval as having **succeeded**, so the investigation is aimed at the right stage.
35. A chunk can contain irrelevant, incorrect, or deliberately instruction-like text. Higher-priority instructions must stay in control.
36. An explicit, inspectable, maintainable statement that two surface forms name one concept. The embedding model may capture some of this implicitly, but you cannot read it, version it, or correct it.
37. Each hop reaches material that has drifted further from the question. In Part 3 the second hop pulls in `access_guide::000` (registration codes), which is connected through the portal node but irrelevant to a question about results.
38. Case B. It cites a real, retrieved chunk, states one true fact from it, and appends an invented $15 fee. Every automated gate passes, so nothing flags it; only a person reading `access_guide::000` catches it.
39. Q3 is unanswerable: nothing in the corpus is relevant, the scores are all zero, and the failure is obvious. Q4 retrieves a genuinely relevant chunk that answers only part of the question — the consent rule — while saying nothing about minors. That partial relevance is what invites a generator to fill the gap with plausible invented policy, and it is far harder for a reviewer to spot.
40. In `st.session_state`, under the key `history`. Without it every rerun would start with an empty conversation, so each new question would erase the previous turns.
41. The app calls the same functions: `expand_query_with_terminology` and `token_overlap_scores` on the same nine chunks. For the bloodwork question the ranking is `results_policy::000` (0.125), `results_policy::002` (0.107), `results_policy::001` (0.038), `access_guide::000` (0.000). The interface added access and no intelligence.
42. Any three of: the corpus has only nine chunks; top-k always returns k passages, including for an unanswerable question; a cited answer can still be wrong; the corpus is fictional; `retrieval only` mode calls no model; the fourth passage above scored 0.000 and is still shown.
43. `localhost` means only the same machine can reach the app. Making it reachable by others is a deployment decision that needs answers on authentication, access control, logging, privacy review, PHI handling, cost control for paid calls, and governance sign-off. All of that is outside this tutorial.

## Coding extensions

- Retrain the Part 0 BPE with 20 and with 200 merges. Report which clinical terms change from many tokens to one.
- In Part 2 section 5, change `window = 3` to `window = 2` and predict the number of allowed pairs before running. **Answer:** 10 of 36, down from 14.
- In Part 2 section 8, change the AR position from `8` to `9` (after `chest`). Predict the token and where the attention goes. **Answer:** `xray`, because `chest` is always followed by `xray` in this corpus. Students read the attention panel to see which tokens the position used, and compare it with the row for position 8.
- In Part 2 section 10, write your own pair of `MLM_TEXTS` that differ only to the right of `[MASK]`, and report whether the top prediction changes.
- In Part 3 section 14, make `answer()` decline when the top retrieval score is below a threshold you choose. Test it with Q3, the unanswerable question. **Answer sketch:** in the `retrieval only` branch, check `retrieved[0]['score'] < THRESHOLD` and return text such as "No passage in these documents matches that question closely enough to show." with the evidence list still attached. Students should justify the threshold from the section 7 score table, note that word-overlap and embedding scores need different thresholds, and rerun the `AppTest` cell to confirm the bloodwork question still returns `results_policy::000`.
- In the archived decoding notebook, implement a repetition penalty and show its effect on greedy decoding.
- In Part 3, add a third chunking strategy (sentence-based) and compare all three on the same four questions.
- Add a synonym to `health_terminology.json` that causes a **wrong** chunk to rank first, then explain why a terminology needs governance.
- Compare CBOW and skip-gram maps in Part 1 without selecting the nicer plot.
