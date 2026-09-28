# 3. Attention and Self-Attention Notebook Blueprint

## Notebook identity

- **Filename:** `Part-2_Attention_and_Self_Attention.ipynb`
- **Duration:** approximately 30 minutes
- **Question:** How do queries assign different weights to keys and combine values?
- **Framework:** PyTorch tensor operations
- **Main outputs:** score matrices, attention-weight heat maps, masked maps, and output vectors

## Required notebook sequence

| Section | Markdown focus | Code/output requirement |
|---|---|---|
| Title | Define attention, self-attention, and the educational synthetic-data boundary. | No code. State tensor shapes and map convention. |
| 0. Setup | Explain tensors, dimensions, fixed seed, CPU execution, and package versions. | Import torch, NumPy, pandas, matplotlib/seaborn; set seed and print versions. |
| 1. A hand-checkable example | Define query, keys, values, dot products, softmax, and weighted sum. | Use the exact 2-D tensors and expected values in Document 11; calculate scores and output step by step. |
| 2. Scaled dot-product attention | Explain why scores are divided by `sqrt(d_k)`. | Implement `attention_from_scratch(Q, K, V, mask=None)` returning output, weights, and scores. Display each matrix. |
| 3. Attention map | State that rows are queries and columns are keys. | Plot a labelled heat map with a colorbar and token labels; verify each row sums to 1. |
| 4. Masking | Explain additive masking and the purpose of a causal mask. | Apply an upper-triangular causal mask; show that future positions receive zero attention. Display before/after maps. |
| 5. Self-attention | Define self-attention as using the same sequence to produce `Q`, `K`, and `V`; distinguish it from cross-attention. | Use the fixed `X`, `W_Q`, `W_K`, and `W_V` in Document 11; compute `Q=XW_Q`, `K=XW_K`, `V=XW_V`; inspect maps and outputs. |
| 6. Compare official implementation | Link PyTorch documentation and explain shape convention. | Compare custom output with `torch.nn.functional.scaled_dot_product_attention`; use `dropout_p=0.0`; assert close equality. |
| 7. Try it yourself | Ask students to alter one projection or mask and predict map changes before running. | Provide a short modification cell; avoid training a transformer. |
| Conclusion | Summarize weighted information flow and limits. | No code. Transition to RAG: retrieval selects text; generation then uses it as context. |

## Exact mathematical content

For queries `Q`, keys `K`, and values `V`:

```text
scores = Q Kᵀ / sqrt(d_k)
weights = softmax(scores, dimension=-1)
output = weights V
```

For a mask, disallowed score entries receive a large negative value before softmax. The causal mask permits a position to attend to itself and earlier positions but not later positions. Explicitly state whether the notebook uses `True = allowed` or `True = blocked`; do not mix conventions.

## Synthetic example requirements

Use tokens such as `clinic`, `appointment`, `result`, `nurse`, and `portal`, but do not imply the values are learned clinical semantics. Construct values so one or two attention relationships are easy to inspect. Use fixed tensors or fixed projection matrices rather than a training loop.

Document 11 is authoritative for tensor values, labels, mask convention (`True = allowed`), expected weights, tolerance, and official-API shape adaptation.

## Required theory beyond the formula

- Define the roles: a query represents what a position is matching, a key is what can be matched, and a value is the information combined into the output.
- Explain scaling: dot-product variance tends to increase with key dimension; dividing by `sqrt(d_k)` reduces softmax saturation.
- Show that softmax is row-wise and that each valid row sums to one.
- State computational complexity for one dense self-attention head as quadratic in sequence length for the score/weight matrix: `O(n²)` attention entries, with projection and value-multiplication costs depending on dimensions.
- Position encodings are required for token order in a full Transformer; the core fixed example omits them deliberately.
- Briefly locate attention inside a Transformer block: multi-head attention, residual connection, normalization, feed-forward network. Do not implement the full block.

Required assertions:

- `weights.shape == (n_queries, n_keys)` for the simple example.
- Every unmasked row sums approximately to 1.
- Causal-mask entries above the diagonal are approximately 0.
- Custom implementation and PyTorch implementation agree within a documented tolerance when dropout is zero.

## Visualization requirements

Every heat map must include:

- title naming the map and mask state;
- x-axis `Key position` or key token;
- y-axis `Query position` or query token;
- token labels in sequence order;
- colorbar labelled `attention weight`;
- a text explanation of one row.

## Pedagogical cautions

- Attention weights are not automatically explanations, causes, or feature importance.
- A high weight indicates a contribution in this computation, not that the token is clinically important.
- Self-attention does not mean a model has human-like attention or understanding.
- The notebook demonstrates one attention head without training a transformer block. It omits residual connections, layer normalization, feed-forward layers, and multi-head parameterization except as a brief extension.
- The PyTorch function may use optimized kernels; the explicit implementation is for inspection, not performance.

## Revision of 21 September 2026

This blueprint describes the first delivered version, archived as `Lab-2-tutorial-all-content/part-2-rag/archive/Attention_full_walkthrough_v1.ipynb`. The current `Part-2_Attention_and_Self_Attention.ipynb` keeps sections 1 to 4 of this blueprint in shortened form and replaces the remaining matrix walkthrough with mask patterns, the AR and MLM training objectives, a tiny model trained under each, the path from an attention row to a prediction, a leak test, and pretrained `distilgpt2` / `distilbert` examples. See **Revision of 21 September 2026** in `README.md` for the section list. Positional encoding moved to the slide deck.
