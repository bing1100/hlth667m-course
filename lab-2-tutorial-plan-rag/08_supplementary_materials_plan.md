# 8. Supplementary Materials Plan

## Glossary and quick reference

Define: token, vocabulary, context window, embedding, Word2Vec, CBOW, skip-gram, vector dimension, cosine similarity, t-SNE, query, key, value, dot product, softmax, scaling, mask, causal mask, self-attention, cross-attention, attention weight, positional encoding, Transformer, chunk, embedding model, vector store, metadata, nearest-neighbor retrieval, top-k, prompt, generation, parametric knowledge, RAG, grounding, citation, unsupported claim, prompt injection, fallback, API key, persistence, and corpus fingerprint.

Include tables for:

1. Word2Vec settings and effects;
2. attention formula and tensor shapes;
3. direct generation versus retrieval versus RAG;
4. local artifact locations;
5. offline/live troubleshooting.

## Exercises and answer key

1. Given two Word2Vec sentences, identify the context window for a target token.
2. Explain one difference between CBOW and skip-gram.
3. Read a t-SNE plot and name one interpretation that is not justified.
4. Compute a two-key attention softmax by hand.
5. Identify which entries a causal mask blocks.
6. Given `X`, `W_Q`, `W_K`, and `W_V`, identify whether the operation is self-attention.
7. Compare retrieved chunk IDs with expected evidence for a RAG question.
8. Rewrite an unsupported RAG answer so it states insufficient evidence.
9. List four things that must not be placed in a hosted API request.
10. Explain why a persistent local vector store is not the same as a secure database service.
11. Diagnose whether a failed RAG answer is primarily a retrieval failure or a generation/evidence-use failure.
12. Explain why retrieved text must be treated as untrusted data rather than higher-priority instructions.

## Reading discussion prompts

- What input, transformation, and output does Word2Vec use?
- What does the attention formula compute before and after softmax?
- What changes when `Q`, `K`, and `V` come from the same sequence?
- Which RAG claims are supported by retrieved text, and which remain unsupported?

## Troubleshooting coverage

Include fixes for missing `gensim`, Torch dtype/shape errors, invalid t-SNE perplexity, nondeterministic Word2Vec results, attention mask convention errors, missing OpenAI key, authentication/rate-limit errors, missing Chroma package, stale index metadata, duplicate IDs, malformed source documents, and offline fallback confusion.

## Optional extensions

- Compare CBOW and skip-gram maps without selecting the nicer plot.
- Add a second attention head conceptually, not as a full Transformer implementation.
- Add cosine retrieval beside Chroma and compare rankings.
- Use a small manually authored expected-answer evaluation set.
- Demonstrate OpenAI-hosted file search only as a separate extension after explaining data residency, retention, cost, and lifecycle controls.
