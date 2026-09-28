# 5. Visualizations and Interpretation Guide

## Required visualization set

| Notebook | Graphic | Required interpretation |
|---|---|---|
| Word2Vec | Token-frequency bar chart | Which tokens occur often enough to learn from? Frequency is not importance. |
| Word2Vec | t-SNE labelled scatter | Which selected points are near one another? The 2-D layout depends on settings and is not a literal semantic map. |
| Attention | Raw score heat map | Which query-key pair has a high dot product before softmax? Scores are not probabilities. |
| Attention | Attention-weight heat map | Which keys receive weight for a stated query? Each unmasked row sums to one. |
| Attention | Causal-mask heat map | Which future positions are blocked? Masked positions should receive zero weight. |
| RAG | Retrieval-result table | Which source chunks were selected, with what distances, and what text do they contain? |
| RAG | Direct-versus-RAG answer table | Which claims are supported, unsupported, or absent? Fluency is not evidence. |

## t-SNE interpretation rules

- Label every point with the token and use a fixed selected vocabulary.
- Set and display `perplexity`, `init`, and `random_state`.
- Do not interpret axis direction or global distances.
- Do not claim a cluster is a validated clinical or linguistic category.
- State that Word2Vec vectors are high-dimensional and t-SNE is a 2-D display transformation.

## Attention-map convention

Use rows for queries and columns for keys. State this directly above every map. Use the same token ordering across raw scores, weights, and masked weights. Include a colorbar. Ask students to read one row aloud: “For query token X, the largest weight is assigned to key token Y.” Do not call a high weight an explanation or cause.

## RAG comparison rubric

For each question, use a compact table with these columns:

- question ID;
- expected source chunk IDs;
- retrieved chunk IDs;
- direct answer available evidence: `none`;
- RAG answer citation coverage;
- unsupported claim observed: yes/no;
- answer says insufficient evidence when appropriate: yes/no;
- instructor note.

A good classroom comparison should include at least one case where direct generation sounds plausible but lacks the fictional policy detail, one case where retrieval supplies the required detail, and one unanswerable case where a safe answer should acknowledge missing evidence.

## Accessibility

Use readable font sizes, high-contrast colorblind-safe palettes, explicit legends, labelled axes, and a written interpretation after each figure. Do not use color alone to distinguish source documents or attention states.
